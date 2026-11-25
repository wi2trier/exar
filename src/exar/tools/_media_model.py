"""The configuration, arguments, and results of `exar.tools.media.View`, which import no decoder."""

from abc import ABC
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from ..ingest.model import MAX_DIMENSION, MAX_FRAMES, MAX_PAGES, vision_media_type
from ..memory.mounts import Mounts
from ..schema import PythonOnly
from . import BaseTool
from .batch import batch_field
from .common import Hint, no_hint

__all__ = [
    "MAX_BYTES",
    "BinaryRead",
    "BinaryReadResult",
    "BinaryReadsArg",
    "ViewFields",
]

MAX_BYTES = 20 * 1024 * 1024
"""Default cap on the bytes of an image or PDF read off disk."""


@dataclass(slots=True, frozen=True, kw_only=True)
class ViewFields(BaseTool, ABC):
    """The configuration of `exar.tools.media.View`, which its calls read."""

    mounts: Mounts
    pdf: Literal["images", "native"] = "images"
    max_bytes: int = MAX_BYTES
    max_pages: int = MAX_PAGES
    max_frames: int = MAX_FRAMES
    max_dimension: int = MAX_DIMENSION
    max_images: int | None = None
    hint: PythonOnly[Hint] = no_hint
    media_type: PythonOnly[Callable[[str], str | None]] = vision_media_type
    executor: PythonOnly[Executor | None] = None


BinaryFilePathArg = Annotated[
    str,
    Field(description="Full workspace path of the image, PDF, or video to read."),
]
PagesArg = Annotated[
    str | None,
    Field(
        description=(
            "PDF only: pages to extract, 1-indexed. Accepts a single page "
            "(`3`), a range (`2-5`), or a comma-separated list (`1,3,5-7`). "
            "Omit to send the whole document. Rejected for non-PDF inputs."
        ),
    ),
]


@dataclass(slots=True, frozen=True)
class BinaryRead:
    """One image, PDF, or video to attach, with the PDF pages to take."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    file_path: BinaryFilePathArg
    pages: PagesArg = None

    @property
    def key(self) -> str:
        """The path, plus the page selection when one was made.

        Examples:
            >>> BinaryRead("~/a.pdf", pages="2-3").key
            '~/a.pdf (pages=2-3)'
        """
        pages = f" (pages={self.pages})" if self.pages else ""

        return f"{self.file_path}{pages}"


BinaryReadsArg = Annotated[
    list[BinaryRead],
    batch_field(
        "Files to attach. Several files share one call's image budget, so ask "
        "for the pages and files you need rather than whole documents."
    ),
]


@dataclass(slots=True, frozen=True)
class BinaryReadResult:
    """Summary of a binary document read."""

    file_path: str
    media_type: str
    size: int
    pages: tuple[int, ...] = ()
    frames: int = 0
    duration: float | None = None
