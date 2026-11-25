import io
from pathlib import Path

import PIL.Image
import pytest

from exar.ingest import (
    extract_pages,
    render_pages,
    sample_animation,
    sample_video,
)
from exar.ingest.media import downscale


def test_pdf_pages() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    document = pdfium.PdfDocument.new()

    for _ in range(3):
        document.new_page(200, 100)

    buffer = io.BytesIO()
    document.save(buffer)
    pdf = buffer.getvalue()

    images = render_pages(pdf, "2-3", max_dimension=50)

    assert [image.number for image in images] == [2, 3]
    assert PIL.Image.open(io.BytesIO(images[0].data)).size == (50, 25)

    extracted, pages = extract_pages(pdf, "3,1")

    assert pages == (3, 1)
    assert len(pdfium.PdfDocument(extracted)) == 2

    with pytest.raises(ValueError, match="limit"):
        render_pages(pdf, max_pages=2)

    with pytest.raises(ValueError, match="could not be opened"):
        render_pages(b"no pdf")


def test_sample_animation() -> None:
    frames = [PIL.Image.new("RGB", (64, 32), (i * 20, 0, 0)) for i in range(10)]
    buffer = io.BytesIO()
    frames[0].save(
        buffer, format="GIF", save_all=True, append_images=frames[1:], duration=200
    )
    pages = io.BytesIO()
    frames[0].save(pages, format="TIFF", save_all=True, append_images=frames[1:])

    # The pages of a TIFF are a still to a reader, not the frames of an animation.
    assert sample_animation(pages.getvalue()) is None

    sample = sample_animation(buffer.getvalue(), max_frames=2, max_dimension=16)

    assert sample is not None
    assert sample.duration == pytest.approx(2.0)
    assert [frame.timestamp for frame in sample.frames] == pytest.approx([0.4, 1.4])
    assert PIL.Image.open(io.BytesIO(sample.frames[0].data)).size == (16, 8)


def test_sample_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"MJPG"), 10, (64, 32))

    for i in range(30):
        writer.write(numpy.full((32, 64, 3), i * 8, dtype=numpy.uint8))

    writer.release()
    sample = sample_video(source, max_frames=2, max_dimension=16)

    assert sample.duration == pytest.approx(3.0)
    assert [frame.timestamp for frame in sample.frames] == pytest.approx([0.75, 2.25])
    assert PIL.Image.open(io.BytesIO(sample.frames[0].data)).size == (16, 8)


def test_sample_video_refuses_undecodable_files(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"no video")

    with pytest.raises(ValueError, match="no duration"):
        sample_video(source)


def test_downscale_mpo_as_jpeg() -> None:
    buffer = io.BytesIO()
    first, second = PIL.Image.new("RGB", (400, 100)), PIL.Image.new("RGB", (400, 100))
    first.save(buffer, format="MPO", save_all=True, append_images=[second])

    assert PIL.Image.open(io.BytesIO(buffer.getvalue())).format == "MPO"

    result = downscale(buffer.getvalue(), 200)

    assert result is not None
    data, media_type = result
    image = PIL.Image.open(io.BytesIO(data))

    assert (media_type, image.format, image.size) == ("image/jpeg", "JPEG", (200, 50))


def test_downscale_palette_resamples() -> None:
    source = PIL.Image.new("RGB", (400, 100), "white")
    source.paste("black", (0, 0, 200, 100))
    buffer = io.BytesIO()
    source.convert("P").save(buffer, format="PNG")

    result = downscale(buffer.getvalue(), 200)

    assert result is not None
    image = PIL.Image.open(io.BytesIO(result[0]))

    assert image.mode == "RGB"
    assert image.size == (200, 50)
