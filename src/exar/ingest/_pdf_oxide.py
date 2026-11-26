"""The pdf_oxide backend of `exar.ingest.parsers.PdfOxide`."""

import tempfile
from functools import cache
from pathlib import Path
from typing import Protocol, TypedDict

from pdf_oxide import OcrConfig, OcrEngine, PdfDocument
from pydantic import TypeAdapter

from .model import Parsed, collect_assets


class Settings(Protocol):
    """What the backend reads of the configuration of `PdfOxide`."""

    @property
    def preserve_layout(self) -> bool: ...

    @property
    def detect_headings(self) -> bool: ...

    @property
    def extract_assets(self) -> bool: ...

    @property
    def ocr_models(self) -> tuple[Path, Path, Path] | None: ...

    @property
    def threads(self) -> int: ...


class _Options(TypedDict):
    preserve_layout: bool
    detect_headings: bool
    include_images: bool
    embed_images: bool
    image_output_dir: str


class _Classification(TypedDict, total=False):
    pages_needing_ocr: list[int]


_CLASSIFICATION = TypeAdapter(_Classification)


@cache
def _engine(models: tuple[Path, Path, Path], threads: int) -> OcrEngine:
    return OcrEngine(*map(str, models), OcrConfig(num_threads=threads))


def parse(source: Path, config: Settings) -> Parsed:
    """Parses a PDF, running OCR on the pages without a text layer if OCR models are given."""
    document = PdfDocument(str(source))
    engine = (
        None
        if config.ocr_models is None
        else _engine(config.ocr_models, config.threads)
    )
    scanned: set[int] = (
        set()
        if engine is None
        else set(
            _CLASSIFICATION.validate_json(document.classify_document()).get(
                "pages_needing_ocr", []
            )
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        images = Path(tmp).resolve()
        options: _Options = {
            "preserve_layout": config.preserve_layout,
            "detect_headings": config.detect_headings,
            "include_images": config.extract_assets,
            "embed_images": False,
            "image_output_dir": str(images),
        }

        if engine is not None and scanned:
            markdown = "\n\n".join(
                document.extract_text_ocr(page, engine)
                if page in scanned
                else document.to_markdown(page, **options)
                for page in range(document.page_count())
            )
        else:
            markdown = document.to_markdown_all(**options)

        # pdf_oxide references the images by their absolute path.
        markdown = markdown.replace(f"{images.as_posix()}/", "")

        return Parsed(markdown=markdown, assets=collect_assets(images))
