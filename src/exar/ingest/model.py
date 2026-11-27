"""Data of ingestion: parsed documents, the images they contain, their chunks, the images decoded from media, and the defaults and media types of decoding."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

from ..helpers import frozendict

__all__ = [
    "ANIMATED_MEDIA_TYPES",
    "MAX_DIMENSION",
    "MAX_FRAMES",
    "MAX_PAGES",
    "VISION_MEDIA_TYPES",
    "Asset",
    "AssetRole",
    "BBox",
    "Chunk",
    "Frame",
    "PageImage",
    "Parsed",
    "Sample",
    "collect_assets",
    "link_images",
    "vision_media_type",
]

MAX_PAGES = 16
"""Default of the pages rendered from one PDF."""

MAX_FRAMES = 8
"""Default of the frames sampled from one video or animation."""

MAX_DIMENSION = 1024
"""Default of the longest side of a rendered page or sampled frame in pixels."""

VISION_MEDIA_TYPES: frozendict[str, str] = frozendict(
    {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".avi": "video/x-msvideo",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }
)
"""Media types by suffix of the files a vision model is shown, declared so they cannot vary with the registry of the host."""

ANIMATED_MEDIA_TYPES = frozenset({"image/gif", "image/png", "image/webp"})
"""The media types of images that may be animated, an APNG for PNG, whose frames are sampled."""


def vision_media_type(file_path: str, /) -> str | None:
    """The media type of a file that a vision model can be shown, `None` for any other.

    Examples:
        >>> vision_media_type("a/Scan.PDF"), vision_media_type("a.md")
        ('application/pdf', None)
    """
    return VISION_MEDIA_TYPES.get(PurePosixPath(file_path).suffix.lower())


@dataclass(frozen=True, slots=True, kw_only=True)
class Frame:
    """A still PNG image at a point of a timeline in seconds."""

    timestamp: float
    data: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class Sample:
    """Frames sampled evenly from a video or animation, with its duration in seconds."""

    frames: tuple[Frame, ...]
    duration: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PageImage:
    """A PNG image of a page of a document, with its 1-based number."""

    number: int
    data: bytes


class AssetRole(StrEnum):
    """What an extracted image is for, mapped by each backend from its own labels."""

    DECORATIVE = "decorative"
    """Icons, logos, signatures, stamps, and page chrome, rarely worth describing."""

    INFORMATIVE = "informative"
    """Charts, diagrams, photographs, screenshots, and maps, usually worth describing."""

    UNKNOWN = "unknown"
    """The backend has no signal for the image."""


@dataclass(frozen=True, slots=True, kw_only=True)
class BBox:
    """Bounding box of an image on its page, normalized to `[0, 1]` from the top left corner."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Asset:
    """An image extracted from a source file.

    Args:
        data: The encoded image.
        role: What the image is for, if the backend knows.
        bbox: Where it lies on its page, if the backend knows.
        page: The 1-based page it lies on, if the backend knows.
        caption: Its caption in the source, or a description added later.
    """

    data: bytes
    role: AssetRole = AssetRole.UNKNOWN
    bbox: BBox | None = None
    page: int | None = None
    caption: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Parsed:
    """The Markdown of a source file and the images it references.

    Args:
        markdown: The content of the source as Markdown.
        assets: The extracted images by the relative path the Markdown references them with.
        encoding: The encoding the source was decoded from, if it was not UTF-8.
    """

    markdown: str
    assets: frozendict[str, Asset] = field(default_factory=frozendict)
    encoding: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    """A passage of a document, its text the span of the document between a prefix and a suffix.

    The prefix and the suffix are context the chunker adds, so the text is `prefix + document[start:end] + suffix`.

    Args:
        text: The passage.
        start: Offset of its first character in the document.
        end: Offset after its last character in the document.
        tokens: Its length in tokens of the chunker.
        start_line: The 1-based line of the document its first character lies on.
        end_line: The 1-based line of the document its last character lies on.
        hierarchy: The titles of the parts of the document it lies in, outermost first, such as the headings of its sections,
            where the chunker knows them.
        prefix: What its text starts with before its span, such as the headings above it or the header of its table.
        suffix: What its text ends with after its span, such as the start of the next chunk as overlap.
    """

    text: str
    start: int
    end: int
    tokens: int
    start_line: int = 0
    end_line: int = 0
    hierarchy: tuple[str, ...] = ()
    prefix: str = ""
    suffix: str = ""


def collect_assets(root: Path, *, limit: int | None = None) -> frozendict[str, Asset]:
    """Reads every regular file below a directory as an asset keyed by its relative POSIX path.

    Symbolic links are skipped, since one left by a backend could name any file of the host.

    Raises:
        ValueError: If the files hold more than *limit* bytes, checked before any is read.
    """
    files = [
        file
        for file in sorted(root.rglob("*"))
        if file.is_file() and not file.is_symlink()
    ]

    if (
        limit is not None
        and (size := sum(file.stat().st_size for file in files)) > limit
    ):
        raise ValueError(f"The assets hold {size} bytes, more than {limit}")

    return frozendict(
        (file.relative_to(root).as_posix(), Asset(data=file.read_bytes()))
        for file in files
    )


def link_images(
    markdown: str, placeholder: str, assets: Sequence[Asset | None], /
) -> Parsed:
    """Names images in document order and replaces their placeholders with references, one per image.

    *assets* has one entry per placeholder, None for an image that could not be extracted,
    whose placeholder is dropped, so every later reference still names its own image.

    Examples:
        >>> parsed = link_images("a <img> b <img> c <img>", "<img>", [Asset(data=b"x"), None, Asset(data=b"y")])
        >>> parsed.markdown, list(parsed.assets)
        ('a ![](image_000000.png) b  c ![](image_000001.png)', ['image_000000.png', 'image_000001.png'])
    """
    named = [asset for asset in assets if asset is not None]
    names = [f"image_{index:06}.png" for index in range(len(named))]
    unused = iter(names)
    references = [None if asset is None else next(unused) for asset in assets]
    head, *tail = markdown.split(placeholder, len(assets))

    return Parsed(
        markdown=head
        + "".join(
            part if name is None else f"![]({name}){part}"
            for name, part in zip(references, tail, strict=False)
        ),
        assets=frozendict(zip(names, named, strict=True)),
    )
