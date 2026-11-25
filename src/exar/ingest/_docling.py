"""The Docling backend of `exar.ingest.parsers.Docling`."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypedDict

import pdf_inspector
import PIL.PngImagePlugin
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.backend_options import (
    HTMLBackendOptions,
    MarkdownBackendOptions,
    PdfBackendOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    TesseractOcrOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument, PictureItem

from .host import Device
from .media import png
from .model import Asset, AssetRole, BBox, Parsed, link_images

logger = logging.getLogger(__name__)


class Settings(Protocol):
    """What the backend reads of the configuration of `Docling`, which it hashes as the key of its converter."""

    @property
    def ocr(self) -> bool: ...

    @property
    def ocr_languages(self) -> tuple[str, ...]: ...

    @property
    def skip_ocr_with_text(self) -> bool: ...

    @property
    def classify_pictures(self) -> bool: ...

    @property
    def threads(self) -> int: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def device(self) -> Device: ...

    @property
    def max_image_pixels(self) -> int | None: ...

    def __hash__(self) -> int: ...


_PDF_FORMATS = frozenset({InputFormat.PDF, InputFormat.IMAGE, InputFormat.METS_GBS})
# Backends that bound the pixels of an image by an option of their own instead of the limit of Pillow.
_PIXEL_BOUNDED = {
    InputFormat.IMAGE: PdfBackendOptions,
    InputFormat.HTML: HTMLBackendOptions,
    InputFormat.MHTML: HTMLBackendOptions,
    InputFormat.MD: MarkdownBackendOptions,
}
_DECORATIVE = frozenset(
    {"icon", "logo", "signature", "stamp", "bar_code", "qr_code", "page_thumbnail"}
)
_UNKNOWN = frozenset({"other", "picture_group"})


class _Update(TypedDict, total=False):
    pipeline_options: ConvertPipelineOptions
    backend_options: PdfBackendOptions | HTMLBackendOptions | MarkdownBackendOptions


# Office documents embed images with text chunks beyond the 1 MB default of Pillow.
# The derived constants are set since PngImagePlugin reads SAFEBLOCK only on import.
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 32 * 1024 * 1024  # ty: ignore[invalid-assignment]
PIL.PngImagePlugin.MAX_TEXT_MEMORY = 64 * 1024 * 1024


def _has_text_layer(source: Path) -> bool:
    """Whether fewer than half of the pages of a PDF need OCR, false if it cannot be classified."""
    try:
        classification = pdf_inspector.classify_pdf(str(source))
    except Exception:  # noqa: BLE001
        # The bindings document no errors, and a needless OCR pass beats a failed parse.
        return False

    pages = classification.page_count

    return pages > 0 and 2 * len(classification.pages_needing_ocr) < pages


# Every converter holds its own models, so only the configurations in use stay.
@lru_cache(maxsize=2)
def _converter(config: Settings, ocr: bool) -> DocumentConverter:
    accelerator = AcceleratorOptions(
        num_threads=config.threads,
        device=AcceleratorDevice(config.device),
    )
    batch_size = config.batch_size
    pdf = ThreadedPdfPipelineOptions(
        do_ocr=ocr,
        do_picture_classification=config.classify_pictures,
        generate_picture_images=True,
        ocr_options=TesseractOcrOptions(lang=list(config.ocr_languages)),
        accelerator_options=accelerator,
        ocr_batch_size=batch_size,
        layout_batch_size=batch_size,
        table_batch_size=batch_size,
    )
    convert = ConvertPipelineOptions(accelerator_options=accelerator)
    converter = DocumentConverter()

    for format, default in converter.format_to_options.items():
        update: _Update = {
            "pipeline_options": pdf if format in _PDF_FORMATS else convert
        }
        backend = _PIXEL_BOUNDED.get(format)

        if backend is not None and config.max_image_pixels is not None:
            update["backend_options"] = backend.model_validate(
                {"max_image_pixels": config.max_image_pixels}
            )

        converter.format_to_options[format] = default.model_copy(update=update)

    return converter


def _role(item: PictureItem) -> AssetRole:
    meta = item.meta

    if (
        meta is None
        or meta.classification is None
        or not meta.classification.predictions
    ):
        return AssetRole.UNKNOWN

    label = str(meta.classification.get_main_prediction().class_name)

    if label in _DECORATIVE:
        return AssetRole.DECORATIVE

    return AssetRole.UNKNOWN if label in _UNKNOWN else AssetRole.INFORMATIVE


def _bbox(item: PictureItem, document: DoclingDocument) -> BBox | None:
    if not item.prov:
        return None

    prov = item.prov[0]
    page = document.pages.get(prov.page_no)

    if page is None or page.size is None:
        return None

    box = prov.bbox.to_top_left_origin(page_height=page.size.height).normalized(
        page.size
    )

    return BBox(
        x_min=float(box.l), y_min=float(box.t), x_max=float(box.r), y_max=float(box.b)
    )


def parse(source: Path, config: Settings) -> Parsed:
    """Parses a file, replacing every image placeholder of Docling with a reference to its asset."""
    ocr = config.ocr and not (
        config.skip_ocr_with_text
        and source.suffix.lower() == ".pdf"
        and _has_text_layer(source)
    )
    result = _converter(config, ocr).convert(source)
    document = result.document

    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        logger.warning(
            "Docling converted %s only in part: %s",
            source.name,
            "; ".join(
                f"page {error.page_no or '?'} {error.component_type}: {error.error_message}"
                for error in result.errors
            )
            or "no errors reported",
        )

    # One entry per placeholder, so a picture without an image keeps the later ones aligned.
    assets: list[Asset | None] = []

    for item, _ in document.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        image = item.get_image(document)
        assets.append(
            None
            if image is None
            else Asset(
                data=png(image),
                role=_role(item),
                bbox=_bbox(item, document),
                page=int(item.prov[0].page_no) if item.prov else None,
                caption=item.caption_text(document).strip() or None,
            )
        )

    # Compact tables, since padding would turn into chunks of whitespace.
    # The placeholders follow the document order of the pictures.
    return link_images(
        str(document.export_to_markdown(compact_tables=True)), "<!-- image -->", assets
    )
