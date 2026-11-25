"""The calls of `View`, which need the `media` extra and are imported only when the tool is built or called."""

import asyncio
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from pydantic_ai import BinaryContent, ToolReturn

from ..ingest.executors import WorkerCrashed
from ..ingest.media import (
    downscale,
    extract_pages,
    render_pages,
    sample_animation,
    sample_video,
    sanitize_image,
)
from ..ingest.model import ANIMATED_MEDIA_TYPES, Sample
from ..memory.mounts import Mount, current
from . import BatchData, Retry
from ._media_model import BinaryRead, BinaryReadResult, ViewFields
from .batch import BatchShare, serve_batch, tool_return
from .common import read_bytes_or_retry, resolve_file_or_retry


async def _decode[R](view: ViewFields, failure: str, call: Callable[[], R]) -> R:
    """Runs a decoder in the executor of the view, refusing input it cannot read and a worker that crashed or timed out."""
    try:
        return await asyncio.get_running_loop().run_in_executor(view.executor, call)
    except (ValueError, WorkerCrashed, TimeoutError) as error:
        raise Retry(f"{failure}: {error}") from error


def _sampled(
    canonical: str, media_type: str, sample: Sample
) -> ToolReturn[BinaryReadResult]:
    """The result of frame-sampled media, one PNG attachment per frame stamped with its timestamp."""
    attachments = [
        BinaryContent(
            data=frame.data,
            media_type="image/png",
            identifier=f"{canonical}#t={frame.timestamp:.1f}s",
        )
        for frame in sample.frames
    ]
    size = sum(len(a.data) for a in attachments)

    return tool_return(
        BinaryReadResult(
            file_path=canonical,
            media_type=media_type,
            size=size,
            frames=len(attachments),
            duration=sample.duration,
        ),
        f"attached {len(attachments)} frames sampled evenly from "
        f"{canonical} ({media_type}, duration {sample.duration:.1f}s); "
        "each frame is labeled with its timestamp",
        attachments,
    )


def _renders_images(view: ViewFields, media_type: str | None) -> bool:
    """Whether a file of a media type reaches the model as images, spending the image cap."""
    return media_type is not None and (
        media_type != "application/pdf" or view.pdf == "images"
    )


def _bounded(budget: int, cap: int | None) -> int:
    """A per-file attachment budget bounded by the share of the image cap."""
    return budget if cap is None else min(budget, cap)


async def _pdf_rendered(
    view: ViewFields, canonical: str, raw: bytes, pages: str | None, cap: int | None
) -> ToolReturn[BinaryReadResult]:
    """Renders the requested PDF pages into one image attachment each."""
    images = await _decode(
        view,
        "PDF could not be read",
        partial(
            render_pages,
            raw,
            pages,
            max_dimension=view.max_dimension,
            max_pages=_bounded(view.max_pages, cap),
        ),
    )

    attachments = [
        BinaryContent(
            data=image.data,
            media_type="image/png",
            identifier=f"{canonical}#page={image.number}",
        )
        for image in images
    ]
    size = sum(len(a.data) for a in attachments)
    selected = tuple(image.number for image in images)
    numbers = ",".join(str(page) for page in selected)

    return tool_return(
        BinaryReadResult(
            file_path=canonical,
            media_type="application/pdf",
            size=size,
            pages=selected,
        ),
        f"rendered {len(attachments)} page(s) ({numbers}) from {canonical} "
        f"as images ({size} bytes)",
        attachments,
    )


async def _pdf_native(
    view: ViewFields, canonical: str, raw: bytes, pages: str | None
) -> ToolReturn[BinaryReadResult]:
    """Forwards a PDF, or a subset of its pages, as a document."""
    selected: tuple[int, ...] = ()

    if pages is not None:
        raw, selected = await _decode(
            view, "invalid pages", partial(extract_pages, raw, pages)
        )

    numbers = f" pages {','.join(str(page) for page in selected)}" if selected else ""

    return tool_return(
        BinaryReadResult(
            file_path=canonical,
            media_type="application/pdf",
            size=len(raw),
            pages=selected,
        ),
        f"attached {canonical}{numbers} (application/pdf, {len(raw)} bytes)",
        [BinaryContent(data=raw, media_type="application/pdf", identifier=canonical)],
    )


async def _video(
    view: ViewFields, canonical: str, absolute: Path, media_type: str, cap: int | None
) -> ToolReturn[BinaryReadResult]:
    """Samples a video into timestamped frame attachments."""
    sample = await _decode(
        view,
        "video could not be decoded",
        partial(
            sample_video,
            absolute,
            max_frames=_bounded(view.max_frames, cap),
            max_dimension=view.max_dimension,
        ),
    )

    return _sampled(canonical, media_type, sample)


async def _animation(
    view: ViewFields, canonical: str, raw: bytes, media_type: str, cap: int | None
) -> ToolReturn[BinaryReadResult] | None:
    """Samples an animated GIF, WebP, or PNG into timestamped frame attachments, None for a still image."""
    sample = await _decode(
        view,
        "animation rejected",
        partial(
            sample_animation,
            raw,
            max_frames=_bounded(view.max_frames, cap),
            max_dimension=view.max_dimension,
        ),
    )

    return None if sample is None else _sampled(canonical, media_type, sample)


async def _view(
    view: ViewFields,
    mounts: Sequence[Mount],
    item: BinaryRead,
    media_type: str | None,
    share: BatchShare,
) -> ToolReturn[BinaryReadResult]:
    """Attaches one file, within its share of the image cap."""
    cap = None if view.max_images is None else share.of(view.max_images)
    mount, local, absolute = resolve_file_or_retry(mounts, item.file_path, view.hint)
    canonical = mount.render(local)

    if media_type is None:
        raise Retry(
            f"'{item.file_path}' is not a format a vision model can be shown"
            f"{view.hint('media', canonical) or '.'}"
        )

    if item.pages is not None and media_type != "application/pdf":
        raise Retry(f"pages= is only valid for PDF inputs, got {media_type}.")

    if media_type.startswith("video/"):
        return await _video(view, canonical, absolute, media_type, cap)

    def check(size: int) -> None:
        if size > view.max_bytes:
            raise Retry(f"file too large: exceeds {view.max_bytes} byte limit.")

    raw = await asyncio.to_thread(read_bytes_or_retry, mount, local, canonical, check)

    if (
        media_type in ANIMATED_MEDIA_TYPES
        and (sampled := await _animation(view, canonical, raw, media_type, cap))
        is not None
    ):
        return sampled

    if media_type == "application/pdf":
        if view.pdf == "native":
            return await _pdf_native(view, canonical, raw, item.pages)

        return await _pdf_rendered(view, canonical, raw, item.pages, cap)

    # A quirky but storable image is returned verbatim, so a read never fails on sanitisation.
    shrunk = await _decode(
        view, "image rejected", partial(downscale, raw, view.max_dimension)
    )
    raw, media_type = shrunk or (sanitize_image(raw, media_type), media_type)

    return tool_return(
        BinaryReadResult(file_path=canonical, media_type=media_type, size=len(raw)),
        f"attached {canonical} ({media_type}, {len(raw)} bytes)",
        [BinaryContent(data=raw, media_type=media_type, identifier=canonical)],
    )


async def view_files(
    view: ViewFields, files: Sequence[BinaryRead]
) -> ToolReturn[BatchData[BinaryReadResult]]:
    """Attaches the files, which divide the image cap between those that become images."""
    media_types = {item.key: view.media_type(item.file_path) for item in files}
    images = sum(_renders_images(view, kind) for kind in media_types.values())

    if view.max_images is not None and images > view.max_images:
        raise Retry(
            f"At most {view.max_images} files that become images can be attached "
            "in one call, since each needs at least one image."
        )

    mounts = current(view.mounts)

    async def run(item: BinaryRead, share: BatchShare) -> ToolReturn[BinaryReadResult]:
        return await _view(view, mounts, item, media_types[item.key], share)

    return await serve_batch(
        files,
        run,
        key=lambda item: item.key,
        concurrency=view.concurrency,
        shares=lambda item: _renders_images(view, media_types[item.key]),
    )
