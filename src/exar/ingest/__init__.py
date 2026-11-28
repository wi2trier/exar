"""Ingestion of external documents into the knowledge an agent retrieves.

The building blocks follow the path of a document:

- `model`: Parsed documents, their images, and their chunks.
- `parsers`: Turn source files into Markdown with their images, routed by suffix and with fallbacks.
- `repair`: Repair the prose of parsed Markdown, such as words hyphenated at line ends and footnotes.
- `assets`: Decide which images to describe and caption them with an injected vision model.
- `media`: Render PDF pages, sample video and animation frames, and sanitize images.
- `chunkers`: Split Markdown and tables into chunks with their character spans and line ranges.
- `coref`: Join chunks that depend on the chunk before them, as an injected `exar.typing.Judge` decides (`CorefMerge`).
- `pipeline`: Chunk a document into a store and stamp its digest and the fingerprint of the chunker last (`index_document`).
- `executors`: Run crash-prone native code in worker processes that survive crashes, hangs, and leaks.
- `host`: Build components from the options of an operator, fingerprint them, and run them off the event loop
  (`configure`, `fingerprint`, `offloaded`), where every component can `prepare` the models it would download.

Parsers, chunkers, and media functions are synchronous and picklable, so the caller picks where they run.
Configured components are frozen dataclasses whose fields are their configuration.
Every backend imports its library on the first call and needs the extra of the same name,
and every name of this package imports its module on first access.
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_exports

if TYPE_CHECKING:
    from .assets import (
        ANIMATION_INSTRUCTIONS,
        IMAGE_INSTRUCTIONS,
        Describe,
        Media,
        caption,
        caption_assets,
        contexts,
        drop_assets,
        informative,
        perceptual_key,
        relink,
    )
    from .chunkers import (
        Chunker,
        MarkdownChunker,
        RecursiveChunker,
        SemanticChunker,
        SentenceChunker,
        TableChunker,
        TokenChunker,
        Whole,
        annotate,
    )
    from .coref import COREF_INSTRUCTIONS, CorefMerge
    from .executors import (
        WORKER_TMP_PREFIX,
        ProcessPool,
        WorkerCrashed,
        limit_threads,
        limit_worker,
    )
    from .host import (
        RUNTIME,
        AsyncChunker,
        AsyncParser,
        Device,
        Host,
        Offloaded,
        Revised,
        SyncComponent,
        configure,
        fingerprint,
        offloaded,
    )
    from .media import (
        downscale,
        extract_pages,
        parse_pages,
        png,
        render_pages,
        sample_animation,
        sample_video,
        sanitize_image,
        still,
    )
    from .model import (
        ANIMATED_MEDIA_TYPES,
        MAX_DIMENSION,
        MAX_FRAMES,
        MAX_PAGES,
        VISION_MEDIA_TYPES,
        Asset,
        AssetRole,
        BBox,
        Chunk,
        Frame,
        PageImage,
        Parsed,
        Sample,
        collect_assets,
        link_images,
        vision_media_type,
    )
    from .parsers import (
        Anydoc,
        Docling,
        Email,
        Json,
        Kreuzberg,
        MarkItDown,
        Pandoc,
        Parser,
        PdfInspector,
        PdfOxide,
        Pdftotext,
        PlainText,
        Repaired,
        Route,
        WithFallback,
        failed,
        garbled,
    )
    from .pipeline import ChunkStore, Ingested, index_document
    from .repair import dehyphenate, inline_footnotes

__all__ = [
    "ANIMATED_MEDIA_TYPES",
    "ANIMATION_INSTRUCTIONS",
    "COREF_INSTRUCTIONS",
    "IMAGE_INSTRUCTIONS",
    "MAX_DIMENSION",
    "MAX_FRAMES",
    "MAX_PAGES",
    "RUNTIME",
    "VISION_MEDIA_TYPES",
    "WORKER_TMP_PREFIX",
    "Anydoc",
    "Asset",
    "AssetRole",
    "AsyncChunker",
    "AsyncParser",
    "BBox",
    "Chunk",
    "ChunkStore",
    "Chunker",
    "CorefMerge",
    "Describe",
    "Device",
    "Docling",
    "Email",
    "Frame",
    "Host",
    "Ingested",
    "Json",
    "Kreuzberg",
    "MarkItDown",
    "MarkdownChunker",
    "Media",
    "Offloaded",
    "PageImage",
    "Pandoc",
    "Parsed",
    "Parser",
    "PdfInspector",
    "PdfOxide",
    "Pdftotext",
    "PlainText",
    "ProcessPool",
    "RecursiveChunker",
    "Repaired",
    "Revised",
    "Route",
    "Sample",
    "SemanticChunker",
    "SentenceChunker",
    "SyncComponent",
    "TableChunker",
    "TokenChunker",
    "Whole",
    "WithFallback",
    "WorkerCrashed",
    "annotate",
    "caption",
    "caption_assets",
    "collect_assets",
    "configure",
    "contexts",
    "dehyphenate",
    "downscale",
    "drop_assets",
    "extract_pages",
    "failed",
    "fingerprint",
    "garbled",
    "index_document",
    "informative",
    "inline_footnotes",
    "limit_threads",
    "limit_worker",
    "link_images",
    "offloaded",
    "parse_pages",
    "perceptual_key",
    "png",
    "relink",
    "render_pages",
    "sample_animation",
    "sample_video",
    "sanitize_image",
    "still",
    "vision_media_type",
]

lazy_exports(__name__)
