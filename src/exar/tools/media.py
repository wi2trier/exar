"""A tool that shows images, PDFs, and videos below mounts to a model that sees, configured as a serializable class.

`View` attaches the visuals to its result, so a model can inspect what a textual conversion of a document lost:
images pass through as images, and videos and animations are sampled to still frames, which no chat model takes natively.
A PDF is rendered to one image per page for servers that only take images, or forwarded as a PDF for providers that take documents.
The decoders are those of `exar.ingest.media`, which need the `media` extra besides the `pydantic-ai` extra
only when the tool is built or called,
and run in the `executor` of the tool, such as an `exar.ingest.ProcessPool` that isolates them,
since a decoder crashing on untrusted input belongs in a process of its own.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic_ai import ToolReturn

from .._lazy import extra_hint
from ..schema import serializable
from . import BatchData
from ._media_model import (
    MAX_BYTES,
    BinaryRead,
    BinaryReadResult,
    BinaryReadsArg,
    ViewFields,
)

__all__ = [
    "MAX_BYTES",
    "BinaryRead",
    "BinaryReadResult",
    "BinaryReadsArg",
    "View",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class View(ViewFields):
    """Attaches images, PDFs, and videos below mounts to its result, a tool of effect `"read"` for models that see.

    The image cap of a request bounds the whole call, since every attachment reaches the model in one request,
    so the files that become images divide it between them before any of them is rendered.
    A PDF forwarded natively spends none of it.

    Args:
        mounts: The mounts of the files, a sequence or a function of the run.
        pdf: Whether a PDF is rendered to an image per page or forwarded as a PDF.
        max_bytes: Most bytes of an image or PDF read off disk.
        max_pages: Most pages rendered from a PDF.
        max_frames: Most frames sampled from a video or animation.
        max_dimension: Longest side of a rendered page or sampled frame in pixels.
        max_images: The image cap of the request, which bounds the pages and frames of a call, none without one.
        hint: What to add to a refusal of a directory or of a file that cannot be shown, a tool to use instead.
        media_type: The media type of a file that can be shown, `None` for one that cannot, `exar.ingest.vision_media_type` by default.
        executor: Runs the decoders of PDFs, videos, and animations, such as an `exar.ingest.ProcessPool` that isolates them,
            the default executor of the event loop if None.
            A crash or a timeout of a worker is refused like input the decoders cannot read.
    """

    extra: ClassVar[str | None] = "media"
    kind: Literal["view"] = field(default="view", repr=False)
    name: str = "view"

    async def __call__(
        self, files: BinaryReadsArg
    ) -> ToolReturn[BatchData[BinaryReadResult]]:
        """Read images, PDFs, or videos and attach them to the tool result.

        Use this when the textual conversion of a document is missing
        information that only the original visual (chart, diagram,
        layout, photo, animation) can convey.  The visuals are attached
        inline with the tool return for the model to inspect.

        Supported types: PDF, PNG, JPEG, GIF, WebP, MP4, WebM, MOV, MKV.
        Videos and animated images are represented by a bounded set of
        still frames sampled evenly across their timeline, each labeled
        with its timestamp.  For PDFs, pass ``pages`` to select a subset
        (e.g. ``"3"``, ``"2-5"``, ``"1,3,5-7"``); omit it to read the
        whole document.  ``pages`` is rejected for non-PDF inputs.
        """
        with extra_hint("View", "media"):
            from ._media import view_files

        return await view_files(self, files)
