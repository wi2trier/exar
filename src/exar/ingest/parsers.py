"""Parsers that turn source files into Markdown with the images they contain.

A parser is a callable from the path of a source file to a `Parsed` result.
Parsers are synchronous and picklable, so the caller decides where they run,
in a thread or a crash-resilient process pool (see `exar.ingest.executors`).
Backends import their library on the first call, each behind an extra of its own.
`Route` picks a parser by the suffix of the file, `WithFallback` recovers from a failed or degraded parse.
"""

import asyncio
import logging
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol

import PIL.Image

from ..helpers import frozendict
from ..subprocess import ProcessLimits, SubprocessError, run, sanitized_env
from ..text import decode, fenced
from .host import Device
from .media import png
from .model import Asset, Parsed, collect_assets

__all__ = [
    "Anydoc",
    "Docling",
    "Kreuzberg",
    "MarkItDown",
    "Pandoc",
    "Parser",
    "PdfInspector",
    "PdfOxide",
    "Pdftotext",
    "PlainText",
    "Route",
    "WithFallback",
    "failed",
    "garbled",
]

logger = logging.getLogger(__name__)


class Parser(Protocol):
    """Parses a source file into Markdown.

    Raises:
        ValueError: If the file cannot be parsed.
    """

    def __call__(self, source: Path, /) -> Parsed: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PlainText(Parser):
    """Reads Unicode or Western text as is.

    Args:
        fence: Whether to wrap the text in a code fence tagged with the suffix of the file,
            so source code and configuration files do not render as Markdown.
    """

    fence: bool = True

    def __call__(self, source: Path, /) -> Parsed:
        return self.parse_bytes(source.read_bytes(), source.suffix)

    def parse_bytes(self, content: bytes, suffix: str, /) -> Parsed:
        """Reads the content of a file with a suffix, such as `".ini"`, which tags the fence.

        Raises:
            ValueError: If the content is not text.

        Examples:
            >>> PlainText().parse_bytes(b"a = 1", ".INI").markdown
            '```ini\\na = 1\\n```\\n'
        """
        decoded = decode(content)

        if decoded is None:
            raise ValueError("The content is not text")

        text = (
            fenced(decoded.text, suffix.lstrip(".").lower())
            if self.fence
            else decoded.text
        )

        return Parsed(markdown=text, encoding=decoded.encoding)


@dataclass(frozen=True, slots=True, kw_only=True)
class Route(Parser):
    """Parses a file with the parser registered for its lowercase suffix.

    Args:
        parsers: The parser of each suffix, such as `".pdf"`.
        default: The parser of every other suffix, files of which are refused if None.

    Examples:
        >>> route = Route(parsers={".txt": PlainText(fence=False)})
        >>> route(Path("notes.docx"))
        Traceback (most recent call last):
        ValueError: No parser for '.docx'
    """

    parsers: Mapping[str, Parser]
    default: Parser | None = None

    def __call__(self, source: Path, /) -> Parsed:
        suffix = source.suffix.lower()
        parser = self.parsers.get(suffix, self.default)

        if parser is None:
            raise ValueError(f"No parser for {suffix!r}")

        return parser(source)


def failed(outcome: Parsed | Exception, /) -> bool:
    """Whether a parser raised."""
    return isinstance(outcome, Exception)


# Three or more glyph name tokens in a row, the signature of a PDF font without a ToUnicode map.
_GLYPH_RUN = re.compile(r"(?:/[A-Za-z][A-Za-z0-9]{0,6}){3,}")


def garbled(outcome: Parsed | str | Exception, /) -> bool:
    """Whether a parser produced mostly raw PDF glyph names instead of text.

    Examples:
        >>> garbled("## /G56/G6F/G6C/G75/G6D/G65/G6E")
        True
        >>> garbled("# Volumenstromangaben in der Drucklufttechnik")
        False
    """
    if isinstance(outcome, Exception):
        return False

    text = outcome if isinstance(outcome, str) else outcome.markdown
    total = sum(not char.isspace() for char in text)
    glyphs = sum(
        not char.isspace()
        for match in _GLYPH_RUN.finditer(text)
        for char in match.group()
    )

    return total > 0 and glyphs / total >= 0.25


@dataclass(frozen=True, slots=True, kw_only=True)
class WithFallback(Parser):
    """Parses with a fallback parser when the outcome of the primary one triggers it.

    The primary outcome is kept if the fallback fails too or yields no text, so its error is raised.
    A recovery is logged as a warning, since it hides the failure of the primary parser.

    Args:
        primary: The parser tried first.
        fallback: The parser tried when the trigger fires.
        trigger: Whether the outcome of the primary parser, its result or its error, needs the fallback.

    Examples:
        >>> parser = WithFallback(primary=Route(parsers={}), fallback=PlainText(fence=False))
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp, "a.txt")
        ...     _ = path.write_text("hello")
        ...     parser(path).markdown
        'hello'
    """

    primary: Parser
    fallback: Parser
    trigger: Callable[[Parsed | Exception], bool] = failed

    def __call__(self, source: Path, /) -> Parsed:
        outcome: Parsed | Exception

        try:
            outcome = self.primary(source)
        except Exception as error:  # noqa: BLE001
            # The trigger decides whether the error is recovered, else it is raised below.
            outcome = error

        if self.trigger(outcome):
            try:
                recovered = self.fallback(source)
            except Exception as error:
                if isinstance(outcome, Exception):
                    raise outcome from error
            else:
                if recovered.markdown.strip():
                    logger.warning(
                        "%s parsed %s where %s failed",
                        type(self.fallback).__name__,
                        source.name,
                        type(self.primary).__name__,
                    )

                    return recovered

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


# Formats whose images pandoc reads from inside the file, so it extracts them.
_ZIP_CONTAINERS = frozenset({".docx", ".epub", ".odt", ".pptx", ".xlsx"})


_MAX_OUTPUT = 256 * 1024**2
"""Bytes of Markdown a converter may write, past which it is killed rather than filling memory."""

_MEMORY = 4 * 1024**3
"""Default of the bytes of memory a converter may use."""


def _run(
    args: list[str],
    *,
    timeout: float | None,
    memory: int | None = None,
    files: int | None = _MAX_OUTPUT,
    cwd: Path | None = None,
) -> bytes:
    """Runs an executable through `exar.subprocess.run` and returns its standard output.

    Blocking, as a parser is, so it runs the command on an event loop of its own.
    The executable sees only the `PATH`, the locale, and the temporary directory of the environment, never the secrets of the server,
    and joins the process group of the caller, so a `ProcessPool` that kills its worker kills it too.
    A file it writes is cut off at *files* bytes, failing it, before it fills the disk.

    Raises:
        ValueError: If it fails, with its standard error and return code, which is negative for a signal.
        TimeoutError: If it runs longer than the timeout, after it is killed.
        exar.subprocess.OutputTooLarge: If it writes more than `_MAX_OUTPUT` bytes, after it is killed.
    """
    limits = ProcessLimits(
        timeout=timeout, cpu=None, memory=memory, files=files, output=_MAX_OUTPUT
    )
    env = sanitized_env("LANG", "LC_ALL", "TMPDIR")

    try:
        return asyncio.run(
            run(args, cwd=cwd, env=env, limits=limits, own_group=False)
        ).stdout
    except SubprocessError as error:
        result = error.result

        raise ValueError(
            f"{result.stderr_text.strip()} (return code {result.returncode})".strip()
        ) from error


@dataclass(frozen=True, slots=True, kw_only=True)
class Pandoc(Parser):
    """Parses the many formats of the `pandoc` executable, such as ODT, RST, EPUB, LaTeX, and Org.

    pandoc runs with `--sandbox`, so the includes and images a file references, such as an RST `include`,
    a LaTeX `\\input`, or an HTML `<img>` of a path or URL, are never read.
    Images are extracted only from the formats that contain them, such as DOCX, EPUB, and ODT.

    Args:
        formats: Input formats of suffixes that pandoc does not infer, such as `{".txt": "markdown"}`.
        timeout: Seconds pandoc may run.
        memory: Bytes of heap pandoc may use, unlimited if None,
            enforced by its runtime since a limit of its address space breaks it.
        max_assets: Bytes of the images extracted from one file, unlimited if None,
            and of each one while pandoc writes it.

    Raises:
        ValueError: If pandoc fails, exceeds its memory, or extracts more than `max_assets` bytes.
    """

    formats: Mapping[str, str] = field(default_factory=frozendict)
    timeout: float | None = 300
    memory: int | None = _MEMORY
    max_assets: int | None = _MAX_OUTPUT

    def __call__(self, source: Path, /) -> Parsed:
        suffix = source.suffix.lower()
        heap = [] if self.memory is None else ["+RTS", f"-M{self.memory}", "-RTS"]
        args = ["pandoc", *heap, "--sandbox", "--to", "markdown"]

        if (format := self.formats.get(suffix)) is not None:
            args += ["--from", format]

        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp).resolve()

            if suffix in _ZIP_CONTAINERS:
                # Relative to the working directory, so pandoc references the images by their asset paths.
                args.append("--extract-media=")

            output = _run(
                [*args, str(source.resolve())],
                cwd=media,
                timeout=self.timeout,
                files=self.max_assets,
            )

            return Parsed(
                markdown=output.decode(errors="replace"),
                assets=collect_assets(media, limit=self.max_assets),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Pdftotext(Parser):
    """Extracts the text of a PDF with poppler's `pdftotext`.

    Poppler maps glyph names to Unicode for fonts without a ToUnicode map,
    so it recovers PDFs that other backends turn into raw glyph names.

    Args:
        layout: Whether to keep the physical layout instead of the reading order.
        timeout: Seconds pdftotext may run.
        memory: Bytes of address space pdftotext may use, unlimited if None, where the platform enforces it.

    Raises:
        ValueError: If the text is empty or still garbled.
    """

    layout: bool = False
    timeout: float | None = 300
    memory: int | None = _MEMORY

    def __call__(self, source: Path, /) -> Parsed:
        args = [
            "pdftotext",
            "-enc",
            "UTF-8",
            "-q",
            *(["-layout"] if self.layout else []),
            str(source.resolve()),
            "-",
        ]
        text = (
            _run(args, timeout=self.timeout, memory=self.memory)
            .decode(errors="replace")
            .replace("\f", "\n\n")
            .strip()
        )

        if not text or garbled(text):
            raise ValueError(f"{source.name} has no readable text")

        return Parsed(markdown=text)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkItDown(Parser):
    """Parses Office documents, PDFs, HTML, and more with Microsoft's MarkItDown."""

    def __call__(self, source: Path, /) -> Parsed:
        from markitdown import MarkItDown as Converter

        return Parsed(markdown=str(Converter().convert(source).text_content))


@dataclass(frozen=True, slots=True, kw_only=True)
class Anydoc(Parser):
    """Parses Office, OpenDocument, RTF, EPUB, and CSV files with anydoc, fast and without models.

    Embedded images are left out, since anydoc renders them as their alternative text only.
    """

    def __call__(self, source: Path, /) -> Parsed:
        import anydoc

        return Parsed(markdown=anydoc.to_markdown(source))


@dataclass(frozen=True, slots=True, kw_only=True)
class PdfInspector(Parser):
    """Parses PDFs with a text layer with pdf-inspector, layout aware and without OCR.

    Args:
        page_markers: Whether to precede every page with a `<!-- Page N -->` comment,
            which sizes headings per page instead of per document.

    Raises:
        ValueError: If the PDF has no text layer.
    """

    page_markers: bool = False

    def __call__(self, source: Path, /) -> Parsed:
        import pdf_inspector

        if self.page_markers:
            pages = pdf_inspector.extract_pages_markdown(str(source)).pages
            markdown = "\n\n".join(
                f"<!-- Page {page.page + 1} -->\n\n{page.markdown}".rstrip()
                for page in pages
            )
        else:
            markdown = pdf_inspector.process_pdf(str(source)).markdown or ""

        if not markdown.strip():
            raise ValueError(f"{source.name} has no text layer")

        return Parsed(markdown=markdown)


@dataclass(frozen=True, slots=True, kw_only=True)
class PdfOxide(Parser):
    """Parses PDFs with pdf_oxide, optionally running OCR on the pages without a text layer.

    Args:
        preserve_layout: Whether to keep the visual columns instead of the reading order.
        detect_headings: Whether to turn detected headings into Markdown headings.
        ocr_models: Paths of the PaddleOCR detection model, recognition model, and dictionary,
            no OCR if None.
        threads: Threads of the OCR engine.
    """

    preserve_layout: bool = False
    detect_headings: bool = True
    ocr_models: tuple[Path, Path, Path] | None = None
    threads: int = 1

    def __call__(self, source: Path, /) -> Parsed:
        from ._pdf_oxide import parse

        return parse(source, self)


@dataclass(frozen=True, slots=True, kw_only=True)
class Kreuzberg(Parser):
    """Parses more than 75 formats with Kreuzberg, running Tesseract OCR where needed.

    Args:
        force_ocr: Whether to run OCR even on files with a text layer.
        ocr_languages: Tesseract language codes, resolved through `TESSDATA_PREFIX`.
        quality_processing: Whether to clean up the extracted text.
    """

    force_ocr: bool = False
    ocr_languages: tuple[str, ...] = ("eng",)
    quality_processing: bool = True

    def __call__(self, source: Path, /) -> Parsed:
        from kreuzberg import (
            ExtractionConfig,
            ImageExtractionConfig,
            OcrConfig,
            TesseractConfig,
            extract_file_sync,
        )

        config = ExtractionConfig(
            force_ocr=self.force_ocr,
            output_format="markdown",
            # Its caches would grow without bound, and inputs are never parsed twice.
            use_cache=False,
            enable_quality_processing=self.quality_processing,
            ocr=OcrConfig(
                backend="tesseract",
                language="+".join(self.ocr_languages),
                tesseract_config=TesseractConfig(use_cache=False),
            ),
            images=ImageExtractionConfig(inject_placeholders=True),  # pyright: ignore[reportCallIssue]  # ty: ignore[unknown-argument]
        )
        result = extract_file_sync(source, config=config)
        markdown = str(result.content)
        assets: dict[str, Asset] = {}

        for image in result.images or []:
            data: bytes = image["data"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
            index = image.get("image_index", len(assets))
            format = image.get("format", "png")
            name = f"image_{index:06}.png"

            if format.lower() != "png":
                data = png(PIL.Image.open(BytesIO(data)))

            page = image.get("page_number")
            assets[name] = Asset(data=data, page=None if page is None else int(page))
            # Kreuzberg references each image by its index and original format.
            markdown = markdown.replace(f"![](image_{index}.{format})", f"![]({name})")

        return Parsed(markdown=markdown, assets=frozendict(assets))


@dataclass(frozen=True, slots=True, kw_only=True)
class Docling(Parser):
    """Parses Office documents, PDFs, images, and HTML with Docling, with layout and table models.

    The two most recent converters are kept per process, since every one holds its own models.

    Args:
        ocr: Whether to run Tesseract OCR on PDFs and images.
        ocr_languages: Tesseract language codes, resolved through `TESSDATA_PREFIX`.
        skip_ocr_with_text: Whether a PDF with a text layer skips OCR.
        classify_pictures: Whether to classify pictures for their `AssetRole`.
        threads: Threads of the models.
        batch_size: Pages fed to the layout, table, and OCR models at once.
        device: Device of the models.
        max_image_pixels: Pixels an image of an image, HTML, or Markdown file may have, the default of Docling if None.
            Images embedded in other formats, such as Office documents, keep the limit of Pillow, see `limit_worker`.
    """

    ocr: bool = True
    ocr_languages: tuple[str, ...] = ("eng",)
    skip_ocr_with_text: bool = True
    classify_pictures: bool = False
    threads: int = 1
    batch_size: int = 8
    device: Device = "cpu"
    max_image_pixels: int | None = None

    def __call__(self, source: Path, /) -> Parsed:
        from ._docling import parse

        return parse(source, self)
