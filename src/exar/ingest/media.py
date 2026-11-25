"""Decoding images, PDFs, and videos into the still images a vision model reads.

The PDF and video functions run native code that crashes or hangs on some malformed input,
so run them on untrusted input in an `exar.ingest.executors.ProcessPool` with `max_tasks=1` and a timeout,
which contains their crashes and hangs but is no security boundary, see `exar.ingest.executors`.
They are plain module level functions of picklable arguments for that reason.
`pypdfium2` and `opencv-python-headless` are imported on the first call.
"""

import io
import logging
from pathlib import Path

import PIL.Image
from PIL import ImageOps, ImageSequence

from .model import (
    ANIMATED_MEDIA_TYPES,
    MAX_DIMENSION,
    MAX_FRAMES,
    MAX_PAGES,
    Frame,
    PageImage,
    Sample,
)

__all__ = [
    "downscale",
    "extract_pages",
    "parse_pages",
    "png",
    "render_pages",
    "sample_animation",
    "sample_video",
    "sanitize_image",
    "still",
]

logger = logging.getLogger(__name__)


def png(image: PIL.Image.Image) -> bytes:
    """Encodes a Pillow image as PNG, converting a mode PNG cannot store first.

    Examples:
        >>> import PIL.Image
        >>> png(PIL.Image.new("CMYK", (1, 1)))[:8]
        b'\\x89PNG\\r\\n\\x1a\\n'
    """
    if image.mode == "I":
        image = image.convert("I;16")
    elif image.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I;16"}:
        image = image.convert("RGBA" if image.has_transparency_data else "RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def still(image: PIL.Image.Image, max_dimension: int) -> bytes:
    """Encodes an image as an opaque PNG whose longer side is at most `max_dimension`.

    Transparent pixels are composited onto white,
    since vision models read no alpha channel and light content on black is unreadable.

    Examples:
        >>> data = still(PIL.Image.new("RGBA", (400, 100)), 200)
        >>> PIL.Image.open(io.BytesIO(data)).size
        (200, 50)
    """
    rgba = image.convert("RGBA")
    background = PIL.Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    flat = PIL.Image.alpha_composite(background, rgba).convert("RGB")
    flat.thumbnail((max_dimension, max_dimension))

    return png(flat)


_LOSSY = {"JPEG": "image/jpeg", "WEBP": "image/webp"}
"""The formats a downscaled image keeps, by their media type, while any other becomes a PNG."""


def downscale(
    data: bytes, max_dimension: int = MAX_DIMENSION
) -> tuple[bytes, str] | None:
    """Shrinks a still image whose longer side exceeds `max_dimension`, returning it with its media type.

    A JPEG, an MPO photo as a JPEG of its primary frame, or WebP stays one, any other format becomes a PNG, and the orientation its EXIF data records is applied,
    since the metadata is dropped.
    None for an image within the bound, an animated one, or one Pillow cannot read, which stay as they are.

    Raises:
        ValueError: For an image of more pixels than Pillow decodes.

    Examples:
        >>> buffer = io.BytesIO()
        >>> PIL.Image.new("RGB", (4000, 1000)).save(buffer, format="JPEG")
        >>> data, media_type = downscale(buffer.getvalue(), 1000)
        >>> media_type, PIL.Image.open(io.BytesIO(data)).size
        ('image/jpeg', (1000, 250))
        >>> downscale(data, 1000) is None
        True
    """
    try:
        with PIL.Image.open(io.BytesIO(data)) as image:
            # An MPO photo is a JPEG whose extra frames are views, of which the primary one is kept.
            kind = "JPEG" if image.format == "MPO" else image.format

            if max(image.size) <= max_dimension or _frame_count(image) > 1:
                return None

            # Pillow resizes these modes with NEAREST, which aliases screenshots and diagrams.
            if image.mode in {"P", "1"}:
                image = image.convert("RGBA" if image.has_transparency_data else "RGB")

            image.thumbnail((max_dimension, max_dimension))
            shrunk = ImageOps.exif_transpose(image)

    except PIL.Image.DecompressionBombError as error:
        raise ValueError(str(error)) from error

    except OSError:
        return None

    if kind not in _LOSSY:
        return png(shrunk), "image/png"

    buffer = io.BytesIO()
    shrunk.convert("RGB" if kind == "JPEG" else "RGBA").save(
        buffer, format=kind, quality=85
    )

    return buffer.getvalue(), _LOSSY[kind]


def parse_pages(spec: str, total: int) -> tuple[int, ...]:
    """Parses a selection of 1-based pages such as `"1,3,5-7"`, in order and without duplicates.

    Raises:
        ValueError: For a malformed token or a page beyond `total`.

    Examples:
        >>> parse_pages("1,3,5-7,3", 10)
        (1, 3, 5, 6, 7)
    """
    pages: dict[int, None] = {}

    for token in spec.split(","):
        start, _, end = token.strip().partition("-")

        # Bounded digits keep a huge token from allocating a huge integer.
        if not (
            start.isdecimal()
            and len(start) <= 9
            and (not end or (end.isdecimal() and len(end) <= 9))
        ):
            raise ValueError(f"Invalid page token: {token.strip()!r}")

        first, last = int(start), int(end or start)

        if first < 1 or last < first:
            raise ValueError(f"Invalid page range: {token.strip()!r}")

        if last > total:
            raise ValueError(f"Page {last} exceeds the {total} pages of the document")

        pages.update(dict.fromkeys(range(first, last + 1)))

    return tuple(pages)


def render_pages(
    pdf: bytes,
    spec: str | None = None,
    *,
    max_dimension: int = MAX_DIMENSION,
    max_pages: int = MAX_PAGES,
) -> tuple[PageImage, ...]:
    """Renders pages of a PDF to PNG images whose longer side is near `max_dimension`.

    Args:
        pdf: The PDF.
        spec: The pages as for `parse_pages`, all if None.
        max_dimension: Pixels of the longer side of a page.
        max_pages: Pages that may be rendered at once.

    Returns:
        The images of the pages in the order of the selection.

    Raises:
        ValueError: For an invalid selection, too many pages, or an unreadable PDF.
    """
    import pypdfium2 as pdfium

    try:
        with pdfium.PdfDocument(pdf) as document:
            total = len(document)
            pages = (
                tuple(range(1, total + 1)) if spec is None else parse_pages(spec, total)
            )

            if len(pages) > max_pages:
                raise ValueError(f"{len(pages)} pages exceed the limit of {max_pages}")

            images: list[PageImage] = []

            for number in pages:
                page = document[number - 1]
                scale = max_dimension / max(page.get_size())
                data = still(page.render(scale=scale).to_pil(), max_dimension)
                images.append(PageImage(number=number, data=data))

            return tuple(images)
    except pdfium.PdfiumError as error:
        raise ValueError(f"PDF could not be opened: {error}") from error


def extract_pages(pdf: bytes, spec: str) -> tuple[bytes, tuple[int, ...]]:
    """Copies pages of a PDF into a new PDF.

    Returns:
        The new PDF and the 1-based pages it holds.

    Raises:
        ValueError: For an invalid selection or an unreadable PDF.
    """
    import pypdfium2 as pdfium

    try:
        with pdfium.PdfDocument(pdf) as source, pdfium.PdfDocument.new() as target:
            pages = parse_pages(spec, len(source))
            target.import_pages(source, pages=[number - 1 for number in pages])
            buffer = io.BytesIO()
            target.save(buffer)

            return buffer.getvalue(), pages
    except pdfium.PdfiumError as error:
        raise ValueError(f"PDF could not be opened: {error}") from error


def _midpoints(count: int, wanted: int) -> list[int]:
    """Up to `wanted` indices spread evenly over `range(count)`, the midpoints of equal segments.

    Examples:
        >>> _midpoints(4, 8)
        [0, 1, 2, 3]
        >>> _midpoints(100, 4)
        [12, 37, 62, 87]
    """
    if count <= wanted:
        return list(range(count))

    return [int((i + 0.5) * count / wanted) for i in range(wanted)]


def _frame_count(image: PIL.Image.Image) -> int:
    """The frames of an animated image, 1 for a still image or one whose frames fail to count.

    The pages of a multi-page TIFF or the views of an MPO photo are frames to Pillow too,
    but stills to a reader, so only the animated formats are counted.
    """
    if PIL.Image.MIME.get(image.format or "") not in ANIMATED_MEDIA_TYPES:
        return 1

    try:
        return getattr(image, "n_frames", 1)
    except Exception:  # noqa: BLE001
        # Pillow raises well beyond OSError on malformed input.
        return 1


def sample_animation(
    data: bytes, *, max_frames: int = MAX_FRAMES, max_dimension: int = MAX_DIMENSION
) -> Sample | None:
    """Samples frames evenly from an animated GIF, WebP, or APNG, None for a still image.

    An image that fails to open or to count its frames counts as still,
    so the caller takes the path of still images, where decoding reports the error.

    Raises:
        ValueError: If a frame of an animation cannot be decoded.

    Examples:
        >>> sample_animation(png(PIL.Image.new("RGB", (1, 1)))), sample_animation(b"no image")
        (None, None)
    """
    try:
        image = PIL.Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        return None

    with image:
        if (count := _frame_count(image)) <= 1:
            return None

        frames: list[Frame] = []
        elapsed = 0.0
        wanted = set(_midpoints(count, max_frames))

        try:
            # Frames are decoded in order, since each one is composited onto the previous ones.
            for index, frame in enumerate(ImageSequence.Iterator(image)):
                if index in wanted:
                    frames.append(
                        Frame(
                            timestamp=elapsed / 1000, data=still(frame, max_dimension)
                        )
                    )

                elapsed += frame.info.get("duration") or 100.0
        except OSError as error:
            raise ValueError(f"Image could not be decoded: {error}") from error

    return Sample(frames=tuple(frames), duration=elapsed / 1000)


def sample_video(
    path: Path, *, max_frames: int = MAX_FRAMES, max_dimension: int = MAX_DIMENSION
) -> Sample:
    """Samples about one frame per second from a video, at most `max_frames` spread evenly.

    Raises:
        ValueError: If the video cannot be decoded or has no duration.
    """
    import cv2

    # One decoding thread per video bounds the CPU use of many concurrent videos.
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise ValueError(f"Video {path.name!r} has no duration")

        duration = count / fps
        wanted = max(1, min(max_frames, round(duration)))
        frames: list[Frame] = []

        for i in range(wanted):
            timestamp = duration * (i + 0.5) / wanted
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()

            if not ok:
                raise ValueError(
                    f"Video {path.name!r} has no frame at {timestamp:.3f}s"
                )

            image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(Frame(timestamp=timestamp, data=still(image, max_dimension)))

        return Sample(frames=tuple(frames), duration=duration)
    finally:
        capture.release()


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The critical chunks and the transparency, every other chunk is metadata that may trip decoders.
_PNG_KEPT = frozenset({b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"})


def sanitize_image(data: bytes, media_type: str) -> bytes:
    """Drops the metadata chunks of a PNG, which trip the decompression limits of some inference servers.

    Other formats and PNGs that cannot be parsed are returned unchanged.

    Examples:
        >>> from PIL.PngImagePlugin import PngInfo
        >>> info, buffer = PngInfo(), io.BytesIO()
        >>> info.add_text("author", "someone")
        >>> PIL.Image.new("RGB", (1, 1)).save(buffer, format="PNG", pnginfo=info)
        >>> clean = sanitize_image(buffer.getvalue(), "image/png")
        >>> b"tEXt" in buffer.getvalue(), b"tEXt" in clean, PIL.Image.open(io.BytesIO(clean)).size
        (True, False, (1, 1))
        >>> truncated = buffer.getvalue()[:40]
        >>> sanitize_image(truncated, "image/png") == truncated
        True
    """
    if media_type != "image/png" or not data.startswith(_PNG_SIGNATURE):
        return data

    output = bytearray(_PNG_SIGNATURE)
    offset = len(_PNG_SIGNATURE)

    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length

        if end > len(data):
            break

        if kind in _PNG_KEPT:
            output += data[offset:end]

        if kind == b"IEND":
            return bytes(output)

        offset = end

    logger.warning("Keeping a PNG unchanged, its chunks could not be parsed")

    return data
