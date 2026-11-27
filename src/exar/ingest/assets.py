"""Deciding which extracted images to describe, and describing them with a vision model.

The vision model is injected as a `Describe` callable, so any framework can provide it,
such as an async function returning the output of a pydantic-ai agent for the prompt.
`caption_assets` describes every informative image of a parsed document once,
grounded in the text around each of its occurrences.
"""

import bisect
import io
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import PIL.Image

from ..helpers import bounded_gather, frozendict
from ._markdown import Image, images
from .media import sanitize_image
from .model import Asset, AssetRole, Parsed, Sample, vision_media_type

__all__ = [
    "ANIMATION_INSTRUCTIONS",
    "IMAGE_INSTRUCTIONS",
    "Describe",
    "Media",
    "caption",
    "caption_assets",
    "contexts",
    "drop_assets",
    "informative",
    "perceptual_key",
    "relink",
]

logger = logging.getLogger(__name__)

# Errors of Pillow on malformed images, DecompressionBombError derives from neither.
_DECODE_ERRORS = (
    PIL.UnidentifiedImageError,
    PIL.Image.DecompressionBombError,
    OSError,
    ValueError,
)

IMAGE_INSTRUCTIONS = (
    "You are writing the canonical, reusable description of an image from a document. "
    "It is stored once and used both as alternative text and as a standalone search result, "
    "so it must stand on its own.\n\n"
    "If the image is a screenshot, photo, or illustration, describe what it shows in one or two sentences, "
    "grounded in the context below.\n"
    "If the image is a table, chart, diagram, or other data figure, transcribe the information it carries, "
    "its values, labels, axes, units, and structure, so nothing that lives only in the figure is lost.\n\n"
    "Be factual and specific. Do not invent details that are not visible. "
    "Do not start with 'This image shows' or similar."
)
"""Asks to describe illustrative images and to transcribe data figures."""

ANIMATION_INSTRUCTIONS = (
    "You are writing the canonical, reusable description of a video or animation from a document, "
    "based on still frames sampled evenly across its timeline, each labeled with its timestamp. "
    "It is stored once and used both as alternative text and as a standalone search result, "
    "so it must stand on its own.\n\n"
    "Describe what it shows and how it progresses over time, the setting, the actions or steps, "
    "any text on screen, and the outcome. Transcribe the steps of a workflow in order.\n\n"
    "Be factual and specific. Do not invent details that are not visible. "
    "Do not start with 'This video shows' or similar."
)
"""Asks to describe how sampled frames progress over time."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Media:
    """An encoded image or document in a prompt to a vision model."""

    data: bytes
    media_type: str


class Describe(Protocol):
    """Answers a prompt of text and media with a vision model."""

    async def __call__(self, prompt: Sequence[str | Media], /) -> str: ...


def _decorative_shape(data: bytes) -> bool:
    """Whether an image is tiny or extremely thin, which is almost always page chrome."""
    if len(data) < 2048:
        return True

    try:
        with PIL.Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except _DECODE_ERRORS:
        return False

    return (
        min(width, height) < 48
        or width * height < 4096
        or max(width, height) / max(min(width, height), 1) > 15
    )


def informative(asset: Asset, /) -> bool:
    """Whether an image is worth describing, by its role or else by its size and shape.

    Examples:
        >>> informative(Asset(data=b"", role=AssetRole.INFORMATIVE))
        True
        >>> informative(Asset(data=b"tiny"))
        False
    """
    if asset.role is not AssetRole.UNKNOWN:
        return asset.role is AssetRole.INFORMATIVE

    return not _decorative_shape(asset.data)


def perceptual_key(data: bytes, size: int = 8) -> int | None:
    """A difference hash of an image that near duplicates share.

    Returns None for undecodable or uniform images,
    which would otherwise all share the same key.

    Examples:
        >>> perceptual_key(b"not an image") is None
        True
        >>> buffer = io.BytesIO()
        >>> PIL.Image.linear_gradient("L").save(buffer, format="PNG")
        >>> perceptual_key(buffer.getvalue()) is None
        True
    """
    try:
        with PIL.Image.open(io.BytesIO(data)) as image:
            pixels = (
                image.convert("L")
                .resize((size + 1, size), PIL.Image.Resampling.BILINEAR)
                .tobytes()
            )
    except _DECODE_ERRORS:
        return None

    bits = 0

    for row in range(size):
        for column in range(size):
            left = pixels[row * (size + 1) + column]
            bits = (bits << 1) | (left > pixels[row * (size + 1) + column + 1])

    return None if bits in (0, (1 << size * size) - 1) else bits


def relink(markdown: str, link: Callable[[str], str | None], /) -> str:
    """Points every image reference at the path *link* returns for its own, dropping it where that is None.

    Only the image syntax of Markdown is rewritten, so prose naming a file and code showing the syntax stay.

    Examples:
        >>> relink("a ![x](a.png) b ![](b.png)", lambda path: None if path == "b.png" else f"img/{path}")
        'a ![x](img/a.png) b '
        >>> relink("![x](a.png) `![x](a.png)`\\n```md\\n![x](a.png)\\n```", lambda _: None)
        ' `![x](a.png)`\\n```md\\n![x](a.png)\\n```'
    """

    def rewrite(image: Image) -> str:
        target = link(image.path)

        return "" if target is None else f"![{image.alt}]({target})"

    return _splice(markdown, images(markdown), rewrite, 0, len(markdown))


def _splice(
    markdown: str, images: Iterable[Image], rewrite: Callable[[Image], str], start: int, end: int
) -> str:
    """The text of *markdown* from *start* to *end* with every one of *images* that lies wholly in it rewritten."""
    parts: list[str] = []
    position = start

    for image in images:
        if start <= image.start and image.end <= end:
            parts.append(markdown[position : image.start])
            parts.append(rewrite(image))
            position = image.end

    parts.append(markdown[position:end])

    return "".join(parts)


def drop_assets(parsed: Parsed, /) -> Parsed:
    """A parsed document without its assets and without any image reference, extracted or not.

    Examples:
        >>> dropped = drop_assets(Parsed(markdown="a ![](a.png) b ![](https://x/b.png)", assets=frozendict(a=Asset(data=b""))))
        >>> dropped.markdown, dict(dropped.assets)
        ('a  b ', {})
    """
    return replace(parsed, markdown=relink(parsed.markdown, lambda _: None), assets=frozendict())


def contexts(markdown: str, window: int = 400) -> dict[str, list[str]]:
    """The alternative text and surrounding prose of every image reference, by referenced path.

    Args:
        markdown: The document.
        window: Characters of prose taken on either side of a reference.

    Examples:
        >>> contexts("Click ![the gear](ui.png) to open settings.")["ui.png"]
        ['Alternative text: the gear\\nSurrounding text: Click to open settings.']
    """
    found = images(markdown)
    starts = [image.start for image in found]
    result: dict[str, list[str]] = {}

    def prose(start: int, end: int) -> str:
        # The images of the document are reused, so no window is parsed again.
        near = found[bisect.bisect_left(starts, start) : bisect.bisect_left(starts, end)]

        return _splice(markdown, near, lambda _: " ", start, end)

    for image in found:
        around = f"{prose(max(0, image.start - window), image.start)} {prose(image.end, image.end + window)}"
        text = " ".join(around.split())
        alt = image.alt.strip()
        parts = [
            *([f"Alternative text: {alt}"] if alt else []),
            *([f"Surrounding text: {text}"] if text else []),
        ]
        result.setdefault(image.path, []).append("\n".join(parts))

    return result


async def caption(
    describe: Describe,
    content: Media | Sample,
    contexts: Sequence[str] = (),
    *,
    instructions: str | None = None,
) -> str:
    """Describes an image, or the frames of a video or animation, once for all the places it appears.

    Args:
        describe: The vision model.
        content: The image, or the frames sampled from a video or animation.
        contexts: The text around each occurrence, such as from `contexts`.
        instructions: The task, `IMAGE_INSTRUCTIONS` or `ANIMATION_INSTRUCTIONS` if None.

    Examples:
        >>> import asyncio
        >>> async def echo(prompt):
        ...     return f"{len(prompt)} parts"
        >>> asyncio.run(caption(echo, Media(data=b"", media_type="image/jpeg"), ["Intro"]))
        '2 parts'
    """
    task = instructions or (
        ANIMATION_INSTRUCTIONS if isinstance(content, Sample) else IMAGE_INSTRUCTIONS
    )
    unique = dict.fromkeys(text.strip() for text in contexts if text.strip())

    if unique:
        task += "\n\nContext where this content appears:\n" + "\n\n---\n\n".join(unique)

    prompt: list[str | Media]

    if isinstance(content, Media):
        prompt = [
            task,
            replace(content, data=sanitize_image(content.data, content.media_type)),
        ]
    else:
        prompt = [f"{task}\n\nTotal duration: {content.duration:.1f}s."]

        for frame in content.frames:
            prompt += [
                f"Frame at {frame.timestamp:.1f}s:",
                Media(data=frame.data, media_type="image/png"),
            ]

    return (await describe(prompt)).strip()


async def caption_assets(
    parsed: Parsed,
    describe: Describe,
    *,
    select: Callable[[Asset], bool] = informative,
    concurrency: int = 8,
) -> Parsed:
    """Captions every selected image of a parsed document, near duplicates once together.

    Args:
        parsed: The document.
        describe: The vision model.
        select: Whether an image is worth describing.
        concurrency: Most images described at once.

    Returns:
        The document with the captions of the selected images replaced by their descriptions.
    """
    occurrences = contexts(parsed.markdown)
    groups: dict[int | str, list[str]] = {}

    for name, asset in parsed.assets.items():
        if select(asset):
            groups.setdefault(perceptual_key(asset.data) or name, []).append(name)

    async def describe_group(names: list[str]) -> str:
        first = names[0]
        media_type = vision_media_type(first) or "image/png"
        texts = [text for name in names for text in occurrences.get(name, [])]

        return await caption(
            describe,
            Media(data=parsed.assets[first].data, media_type=media_type),
            texts,
        )

    captions = await bounded_gather(groups.values(), describe_group, limit=concurrency)
    updated = dict(parsed.assets)

    for names, text in zip(groups.values(), captions, strict=True):
        for name in names:
            updated[name] = replace(updated[name], caption=text)

    return replace(parsed, assets=frozendict(updated))
