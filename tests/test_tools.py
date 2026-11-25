import inspect

import pytest

from exar.helpers import get_name
from exar.tools import ItemFailure, Retry, as_tool, effect, effect_of, run_batch


async def test_run_batch() -> None:
    async def read(path: str) -> str:
        if path.startswith("/missing"):
            raise Retry(f"{path} not found")

        return path.upper()

    assert await run_batch(["/a", "/missing", "/a"], read, concurrency=2) == {
        "/a": "/A",
        "/missing": ItemFailure(item="/missing", reason="/missing not found"),
    }

    with pytest.raises(Retry, match="Every item failed"):
        await run_batch(["/missing-1", "/missing-2"], read)

    with pytest.raises(Retry, match="^/missing: /missing not found$"):
        await run_batch(["/missing"], read)


def test_effect() -> None:
    def tool() -> None: ...

    assert effect_of(tool) == "execute"
    assert effect_of(effect("read")(tool)) == "read"

    vars(tool)["effect"] = "delete"

    with pytest.raises(ValueError):
        effect_of(tool)


def test_as_tool() -> None:
    class Echo:
        async def __call__(self, text: str) -> str:
            """Repeats the text."""
            return text

    tool = as_tool(
        Echo().__call__, name="say", effect="read", arguments={"text": "What to say."}
    )
    plain = as_tool(Echo().__call__, name="say", effect="read", description="Says it.")

    assert (get_name(tool), tool.__doc__, effect_of(tool)) == (
        "say",
        "Repeats the text.",
        "read",
    )
    assert plain.__doc__ == "Says it."
    assert (
        inspect.signature(tool)
        .parameters["text"]
        .annotation.__metadata__[0]
        .description
        == "What to say."
    )

    with pytest.raises(ValueError, match="no parameters other"):
        as_tool(Echo().__call__, name="say", effect="read", arguments={"other": "x"})
