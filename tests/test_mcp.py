import types

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitRequestParams
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken
from mcp.types import ElicitResult, ImageContent, InputRequiredResult, TextContent
from pydantic_ai import BinaryContent

from exar import mcp
from exar.mcp import Confirmations, server
from exar.policy import EffectPolicy
from exar.tools import Retry, SchemaTool, effect
from exar.tools.batch import tool_return
from exar.typing import Tool

CONFIRMATIONS = Confirmations(key=b"test")


@effect("read")
async def search(query: str, limit: int = 10) -> str:
    """Searches the notes."""
    if not query:
        raise Retry("Pass a non-empty query.")

    return f"{limit} hits for {query}"


@effect("write")
async def save(text: str) -> str:
    """Saves a note."""
    return text


@effect("execute")
async def send(to: str) -> str:
    """Sends the notes."""
    return to


@effect("read")
async def look(name: str) -> object:
    """Looks at a file."""
    return tool_return(
        {"name": name}, "seen", [BinaryContent(b"png", media_type="image/png")]
    )


async def test_tools_carry_schema_and_annotations() -> None:
    async with Client(
        server(
            [search, save],
            name="notes",
            policy=EffectPolicy(),
            confirmations=CONFIRMATIONS,
        )
    ) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    assert tools["search"].description == "Searches the notes."
    assert tools["search"].input_schema["required"] == ["query"]
    assert tools["search"].annotations is not None
    assert tools["search"].annotations.read_only_hint is True
    assert tools["save"].annotations is not None
    assert tools["save"].annotations.destructive_hint is True


async def test_call_and_retry() -> None:
    mcp = server(
        [search], name="notes", policy=EffectPolicy(), confirmations=CONFIRMATIONS
    )

    async with Client(mcp) as client:
        result = await client.call_tool("search", {"query": "cbr"})

        with pytest.raises(ToolError, match="non-empty query"):
            await client.call_tool("search", {"query": ""})

    assert result.structured_content == {"result": "10 hits for cbr"}


async def test_policy_hides_denied_tools_and_asks_the_client() -> None:
    policy = EffectPolicy(write="ask", execute="deny")
    asked: list[str] = []

    async def describe(name: str, arguments: object, /) -> str:
        return f"{name} {arguments}"

    async def answer(
        message: str, _type: object, _params: ElicitRequestParams, _ctx: object
    ) -> bool:
        asked.append(message)

        return len(asked) > 1

    mcp = server(
        [search, save, send],
        name="notes",
        policy=policy,
        describe=describe,
        confirmations=CONFIRMATIONS,
    )

    async with Client(mcp, elicitation_handler=answer) as client:
        names = {tool.name for tool in await client.list_tools()}

        with pytest.raises(ToolError, match="denied by the user"):
            await client.call_tool("save", {"text": "hi"})

        result = await client.call_tool("save", {"text": "hi"})

    assert names == {"search", "save"}
    assert asked == ["save {'text': 'hi'}"] * 2
    assert result.structured_content == {"result": "hi"}


async def test_factory_builds_the_tools_of_each_request() -> None:
    callers: list[AccessToken | None] = []

    async def tools(token: AccessToken | None) -> list[Tool]:
        callers.append(token)

        return [search]

    mcp = server(
        tools, name="notes", policy=EffectPolicy(), confirmations=CONFIRMATIONS
    )

    async with Client(mcp) as client:
        listed = await client.list_tools()
        result = await client.call_tool("search", {"query": "cbr", "limit": 2})

    assert [tool.name for tool in listed] == ["search"]
    assert result.structured_content == {"result": "2 hits for cbr"}
    assert callers and set(callers) == {None}


def _answered(monkeypatch: pytest.MonkeyPatch, state: str | None) -> None:
    """Retries the current request with a yes and *state*."""
    ctx = types.SimpleNamespace(
        input_responses={
            "confirmation": ElicitResult(action="accept", content={"value": True})
        },
        request_state=state,
    )
    monkeypatch.setattr(mcp, "get_context", lambda: ctx)


async def test_a_confirmation_counts_once_and_only_for_the_state_issued_for_its_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    confirm = Confirmations(key=b"test")
    args = {"path": "a.md"}
    _answered(monkeypatch, None)
    asked = await confirm("delete", args, "Delete a.md?")
    assert isinstance(asked, InputRequiredResult) and asked.request_state

    _answered(monkeypatch, asked.request_state)

    assert await confirm("delete", args, "Delete a.md?") is None
    assert await confirm("delete", args, "Delete a.md?") is not None
    assert await confirm("delete", {"path": "b.md"}, "Delete a.md?") is not None
    assert await Confirmations(key=b"other")("delete", args, "Delete a.md?") is not None

    def caller(sub: str, client_id: str) -> None:
        token = AccessToken(
            token="t", client_id=client_id, scopes=[], claims={"sub": sub}
        )
        monkeypatch.setattr(mcp, "get_access_token", lambda: token)

    caller("a@b", "c")
    _answered(monkeypatch, None)
    asked_a = await confirm("delete", args, "Delete a.md?")
    assert isinstance(asked_a, InputRequiredResult) and asked_a.request_state
    _answered(monkeypatch, asked_a.request_state)
    caller("a", "b@c")

    assert await confirm("delete", args, "Delete a.md?") is not None

    _answered(monkeypatch, "0" * 64)

    assert await confirm("delete", args, "Delete a.md?") is not None

    nonce, expiry, mac = asked.request_state.split(".")
    _answered(monkeypatch, f"{nonce}.{int(expiry) + 60}.{mac}")

    assert await confirm("delete", args, "Delete a.md?") is not None

    _answered(monkeypatch, f"{nonce}.²{expiry}.{mac}")

    assert await confirm("delete", args, "Delete a.md?") is not None


async def test_rich_result_maps_to_content_and_structured_content() -> None:
    mcp = server(
        [look], name="notes", policy=EffectPolicy(), confirmations=CONFIRMATIONS
    )

    async with Client(mcp) as client:
        result = await client.call_tool("look", {"name": "a"})

    assert [type(block) for block in result.content] == [TextContent, ImageContent]
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "seen"
    assert result.structured_content == {"name": "a"}


async def test_a_schema_tool_serves_its_schema() -> None:
    async def echo(**arguments: object) -> object:
        return arguments["text"]

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    tool = SchemaTool(
        name="echo", description="Echoes.", schema=schema, effect="read", call=echo
    )

    async with Client(
        server([tool], name="echo", policy=EffectPolicy(), confirmations=CONFIRMATIONS)
    ) as client:
        [listed] = await client.list_tools()
        result = await client.call_tool("echo", {"text": "a"})

    assert (listed.description, listed.input_schema) == ("Echoes.", schema)
    assert result.structured_content == {"result": "a"}
