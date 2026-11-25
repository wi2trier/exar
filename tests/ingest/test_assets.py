import io
from collections.abc import Sequence

import PIL.Image

from exar.helpers import frozendict
from exar.ingest import Asset, AssetRole, Media, Parsed, caption_assets


def noise() -> bytes:
    buffer = io.BytesIO()
    PIL.Image.effect_noise((300, 200), 64).save(buffer, format="PNG")

    return buffer.getvalue()


async def test_caption_assets_describes_duplicates_once() -> None:
    prompts: list[Sequence[str | Media]] = []

    async def describe(prompt: Sequence[str | Media], /) -> str:
        prompts.append(prompt)

        return f" caption {len(prompts)} "

    photo = noise()
    parsed = Parsed(
        markdown="Intro ![a](a.png) middle ![b](b.png) end ![logo](c.png)",
        assets=frozendict(
            {
                "a.png": Asset(data=photo),
                "b.png": Asset(data=photo),
                "c.png": Asset(data=noise(), role=AssetRole.DECORATIVE),
            }
        ),
    )

    result = await caption_assets(parsed, describe)

    ((task, image),) = prompts

    assert isinstance(task, str) and isinstance(image, Media)
    assert "Alternative text: a" in task and "Alternative text: b" in task
    assert image.media_type == "image/png"
    assert [asset.caption for asset in result.assets.values()] == [
        "caption 1",
        "caption 1",
        None,
    ]
