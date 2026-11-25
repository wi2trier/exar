import asyncio
from collections.abc import AsyncIterator
from types import NoneType

import pytest
from fastmcp import FastMCP
from mcp import MCPError
from mcp.types import CONNECTION_CLOSED, ToolAnnotations
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from exar.mcp_client import McpClient


def _app() -> FastMCP:
    app = FastMCP("crm", instructions="Look a record up before changing it.")

    @app.tool(annotations=ToolAnnotations(read_only_hint=True))
    def lookup(key: str) -> str:
        """Look a record up."""
        return f"record {key}"

    @app.tool
    def delete(key: str) -> str:
        """Delete a record."""
        return f"deleted {key}"

    @app.tool(annotations=ToolAnnotations(read_only_hint=True))
    def export() -> str:
        """Export every record."""
        return "everything"

    return app


@pytest.fixture
async def connected() -> AsyncIterator[McpClient[None]]:
    client = McpClient[None](
        name="crm",
        toolset=MCPToolset(_app(), include_instructions=True),
        blocked={"export"},
        trusted=True,
    )

    async with client.supervised():
        while client.tools is None:
            await asyncio.sleep(0.01)

        yield client


async def _turn(
    client: McpClient[None], *calls: ToolCallPart
) -> tuple[set[str], str, list[ModelMessage]]:
    """One turn offering *client*, making *calls* first, with the tool names and instructions the model got."""
    seen: list[tuple[set[str], str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        params = info.model_request_parameters
        seen.append(
            (
                {tool.name for tool in params.function_tools},
                "\n".join(part.content for part in params.instruction_parts or []),
            )
        )

        return ModelResponse(
            parts=list(calls) if len(messages) == 1 and calls else [TextPart("ok")]
        )

    agent = Agent(
        FunctionModel(respond), deps_type=NoneType, toolsets=[client.agent_toolset()]
    )
    result = await agent.run("hi")

    return *seen[0], result.all_messages()


async def test_tools_are_prefixed_and_their_effects_declared(
    connected: McpClient[None],
) -> None:
    names, instructions, _ = await _turn(connected)

    assert names == {"crm_lookup", "crm_delete"}
    assert "Look a record up before" in instructions
    assert {
        name: connected.effect_of(tool)
        for name, tool in (connected.tools or {}).items()
    } == {
        "lookup": "read",
        "delete": "execute",
    }


async def test_an_untrusted_server_executes_but_the_listed_reads() -> None:
    client = McpClient[None](
        name="crm", toolset=MCPToolset(_app()), effects={"lookup": "read"}
    )
    client.offer(await client.toolset.list_tools())

    assert {
        name: client.effect_of(tool) for name, tool in (client.tools or {}).items()
    } == {
        "lookup": "read",
        "delete": "execute",
        "export": "execute",
    }

    connected_instructions = (await _turn(client))[1]
    assert "Look a record up" not in connected_instructions


async def test_a_lost_connection_keeps_the_tools_and_fails_the_calls(
    connected: McpClient[None], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable(*_args: object) -> None:
        raise ModelRetry("Connection closed") from MCPError(
            CONNECTION_CLOSED, "Connection closed"
        )

    before = await _turn(connected)
    monkeypatch.setattr(connected.toolset, "call_tool", unreachable)
    lost = await _turn(connected, ToolCallPart("crm_lookup", {"key": "a"}, "call"))
    down = await _turn(connected, ToolCallPart("crm_lookup", {"key": "a"}, "call"))

    assert not connected.connected
    assert lost[:2] == down[:2] == before[:2]
    assert "could not be reached" in _returned(lost[2])
    assert "crm are unavailable" in _returned(down[2])


def _returned(messages: list[ModelMessage]) -> str:
    return "\n".join(
        part.model_response_str()
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


async def test_a_server_is_asked_for_its_tools_again() -> None:
    app = _app()
    client = McpClient[None](name="crm", toolset=MCPToolset(app), refresh_seconds=0.01)

    async with client.supervised():
        while client.tools is None:
            await asyncio.sleep(0.01)

        @app.tool
        def merge(key: str) -> str:
            """Merge two records."""
            return f"merged {key}"

        async with asyncio.timeout(5):
            while "merge" not in (client.tools or {}):
                await asyncio.sleep(0.01)
