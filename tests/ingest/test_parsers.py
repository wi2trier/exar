import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path

import PIL.Image
import pytest

from exar.ingest import (
    Docling,
    Email,
    Json,
    Kreuzberg,
    Pandoc,
    Parsed,
    Parser,
    PdfOxide,
    Pdftotext,
    PlainText,
    Route,
    WithFallback,
    garbled,
    png,
)


@dataclass(frozen=True, slots=True)
class Fixed:
    markdown: str

    def __call__(self, source: Path, /) -> Parsed:
        return Parsed(markdown=self.markdown)


@dataclass(frozen=True, slots=True)
class Failing:
    def __call__(self, source: Path, /) -> Parsed:
        raise ValueError("unreadable")


def test_plain_text_decodes_legacy_text(tmp_path: Path) -> None:
    source = tmp_path / "config.ini"
    source.write_bytes("key = Grüße".encode("cp1252"))

    parsed = PlainText()(source)

    assert parsed.markdown == "```ini\nkey = Grüße\n```\n"
    assert parsed.encoding == "cp1252"

    source.write_bytes(b"\x89PNG\r\n\x1a\n\x00")

    with pytest.raises(ValueError, match="not text"):
        PlainText()(source)


def test_route_and_fallback(tmp_path: Path) -> None:
    source = tmp_path / "scan.PDF"
    source.touch()
    garbled_pdf = Fixed("/G56/G6F/G6C/G75/G6D/G65")
    parser = Route(
        parsers={
            ".pdf": WithFallback(
                primary=garbled_pdf, fallback=Fixed("Volume"), trigger=garbled
            )
        },
        default=WithFallback(primary=Failing(), fallback=Failing()),
    )

    assert parser(source).markdown == "Volume"
    assert WithFallback(primary=garbled_pdf, fallback=Failing(), trigger=garbled)(
        source
    ) == Parsed(markdown=garbled_pdf.markdown)

    with pytest.raises(ValueError, match="unreadable"):
        parser(tmp_path / "notes.docx")

    with pytest.raises(ValueError, match="unreadable"):
        WithFallback(primary=Failing(), fallback=Fixed(" \n"))(source)


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="needs pandoc")
def test_pandoc(tmp_path: Path) -> None:
    source = tmp_path / "notes.rst"
    source.write_text("Title\n=====\n\nSome *text*.\n")

    assert Pandoc()(source).markdown.strip() == "# Title\n\nSome *text*."


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="needs pandoc")
def test_pandoc_reads_nothing_but_its_file(tmp_path: Path) -> None:
    secret, image = tmp_path / "secret.txt", tmp_path / "secret.png"
    secret.write_text("SECRET")
    image.write_bytes(png(PIL.Image.new("RGB", (1, 1))))
    sources = {
        "a.rst": f"Title\n=====\n\n.. include:: {secret}\n",
        "b.tex": f"\\input{{{secret}}}\n",
        "c.org": f'#+INCLUDE: "{secret}"\n',
        "d.html": f'<img src="{image}"><img src="file://{image}"><img src="{secret}">',
    }
    parser = Pandoc(
        formats={".rst": "rst", ".tex": "latex", ".org": "org", ".html": "html"}
    )

    for name, content in sources.items():
        source = tmp_path / name
        source.write_text(content)
        parsed = parser(source)

        assert "SECRET" not in parsed.markdown
        assert not parsed.assets

    page = tmp_path / "page.md"
    page.write_text(f"![]({image})\n")
    subprocess.run(["pandoc", str(page), "-o", str(tmp_path / "page.docx")], check=True)
    parsed = parser(tmp_path / "page.docx")

    assert len(parsed.assets) == 1
    assert all(path in parsed.markdown for path in parsed.assets)

    for limited in (Pandoc(max_assets=1), Pandoc(memory=1024**2)):
        with pytest.raises(ValueError):
            limited(tmp_path / "page.docx")


def test_markitdown(tmp_path: Path) -> None:
    pytest.importorskip("markitdown")
    from exar.ingest import MarkItDown

    source = tmp_path / "page.html"
    source.write_text("<html><body><h1>Title</h1><p>Body</p></body></html>")

    assert MarkItDown()(source).markdown.strip() == "# Title\n\nBody"


def test_kreuzberg_links_images(tmp_path: Path) -> None:
    pytest.importorskip("kreuzberg")
    source = _docx_with_images(tmp_path / "images.docx", ("PNG", "JPEG"))
    parsed = Kreuzberg()(source)

    assert parsed.markdown.split() == [
        "![](image_000000.png)",
        "![](image_000001.png)",
    ]
    assert list(parsed.assets) == ["image_000000.png", "image_000001.png"]


def _pdf(path: Path, text: str) -> Path:
    """Writes a single page PDF drawing `text` in a standard font, with a real cross-reference table."""
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        b"<</Length %d>>stream\n%s\nendstream" % (len(stream), stream),
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []

    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += b"%d 0 obj%s endobj\n" % (number, body)

    xref = len(output)
    output += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    output += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    output += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(output))

    return path


def test_pdf_inspector(tmp_path: Path) -> None:
    pytest.importorskip("pdf_inspector")
    from exar.ingest import PdfInspector

    source = _pdf(tmp_path / "report.pdf", "Fixture Heading")
    plain = PdfInspector()(source)

    assert "Fixture Heading" in plain.markdown
    assert "<!-- Page" not in plain.markdown
    assert PdfInspector(page_markers=True)(source).markdown.startswith(
        "<!-- Page 1 -->"
    )

    with pytest.raises(ValueError, match="no text layer"):
        PdfInspector()(_pdf(tmp_path / "scan.pdf", ""))


def test_docling_skips_ocr_only_with_a_text_layer(tmp_path: Path) -> None:
    pytest.importorskip("docling")
    from exar.ingest._docling import _has_text_layer

    assert _has_text_layer(_pdf(tmp_path / "report.pdf", "Fixture Heading"))
    assert not _has_text_layer(_pdf(tmp_path / "scan.pdf", ""))
    assert not _has_text_layer(tmp_path / "missing.pdf")


def test_anydoc_converts_structure_to_markdown(tmp_path: Path) -> None:
    pytest.importorskip("anydoc")
    from exar.ingest import Anydoc

    source = tmp_path / "note.rtf"
    source.write_text(r"{\rtf1\ansi \b Heading\b0 \par Plain body text.\par}")
    parsed = Anydoc()(source)

    assert "**Heading**" in parsed.markdown
    assert "Plain body text." in parsed.markdown


def test_pdftotext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "legacy.pdf"
    outputs = iter([b"Volume\fflow\n", b"/G56/G6F/G6C/G75/G6D/G65", b" \n"])
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> bytes:
        calls.append(args)

        return next(outputs)

    monkeypatch.setattr("exar.ingest.parsers._run", fake_run)

    assert Pdftotext()(source).markdown == "Volume\n\nflow"
    assert calls[0][0] == "pdftotext" and calls[0][-2:] == [str(source), "-"]

    for _ in range(2):
        with pytest.raises(ValueError, match="no readable text"):
            Pdftotext()(source)


def _docx_with_images(path: Path, formats: tuple[str, ...] = ("PNG", "PNG")) -> Path:
    """Writes a DOCX of one picture per format, each a solid color, at *path* with its suffix."""
    docx = pytest.importorskip("docx")
    path = path.with_suffix(".docx")
    document = docx.Document()

    for color, format in zip(("red", "blue", "green"), formats, strict=False):
        image = BytesIO()
        PIL.Image.new("RGB", (40, 30), color).save(image, format)
        document.add_picture(image)

    document.save(str(path))

    return path


def _scan_pdf(path: Path) -> Path:
    """Writes a PDF of a single picture at *path* with its suffix."""
    path = path.with_suffix(".pdf")
    PIL.Image.new("RGB", (200, 150), "green").save(path, "PDF")

    return path


@pytest.mark.parametrize(
    ("parser", "backend", "write"),
    [
        pytest.param(
            Pandoc,
            None,
            _docx_with_images,
            marks=pytest.mark.skipif(
                shutil.which("pandoc") is None, reason="needs pandoc"
            ),
        ),
        (Kreuzberg, "kreuzberg", _docx_with_images),
        (Docling, "docling", _docx_with_images),
        (PdfOxide, "pdf_oxide", _scan_pdf),
    ],
)
def test_images_are_dropped_with_their_references_unless_extracted(
    tmp_path: Path,
    parser: Callable[..., Parser],
    backend: str | None,
    write: Callable[[Path], Path],
) -> None:
    if backend is not None:
        pytest.importorskip(backend)

    source = write(tmp_path / "source")
    extracted = parser()(source)
    dropped = parser(extract_assets=False)(source)

    assert extracted.assets
    assert all(path in extracted.markdown for path in extracted.assets)
    assert not dropped.assets
    assert "![" not in dropped.markdown


def test_json_lines_of_differing_keys_share_a_table(tmp_path: Path) -> None:
    source = tmp_path / "rows.JSONL"
    source.write_text('{"a": 1, "b": {"c": "x|y"}}\n\n{"a": 2, "d": [1, 2]}\n')

    assert Json()(source).markdown == (
        "| a | b.c | d |\n| --- | --- | --- |\n| 1 | x\\|y |  |\n| 2 |  | [1, 2] |\n"
    )

    source.write_text("{")

    with pytest.raises(ValueError):
        Json()(source)


def test_email_prefers_plain_text_and_lists_attachments() -> None:
    message = EmailMessage()
    message["Subject"] = "Report"
    message["From"] = "a@example.org"
    message["To"] = "b@example.org"
    message.set_content("Plain body.")
    message.add_alternative("<p>HTML body.</p>", subtype="html")
    message.add_attachment(
        b"%PDF", maintype="application", subtype="pdf", filename="r.pdf"
    )

    assert Email().parse_bytes(message.as_bytes()).markdown == (
        "# Report\n\n- **From:** a@example.org\n- **To:** b@example.org\n\n"
        "Plain body.\n\n## Attachments\n\n- r.pdf\n"
    )
