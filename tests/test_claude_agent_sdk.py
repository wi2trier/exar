from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from mcp import Client
from mcp.types import ImageContent, TextContent
from pydantic import TypeAdapter
from pydantic_ai import BinaryContent

from exar.agents import RunConfig, run_async
from exar.claude_agent_sdk import ClaudeAgent, can_use_tool, tools
from exar.policy import EffectPolicy
from exar.tools import Retry, effect
from exar.tools.batch import tool_return
from exar.typing import Action, Effect


async def search(query: str, limit: int = 10) -> str:
    """Searches the notes."""
    if not query:
        raise Retry("Pass a non-empty query.")

    return f"{limit} hits for {query}"


@effect("write")
async def save(text: str) -> str:
    """Saves a note."""
    return text


@effect("read")
async def look(name: str) -> object:
    """Looks at a file."""
    return tool_return(
        {"name": name}, "seen", [BinaryContent(b"png", media_type="image/png")]
    )


def text(content: object) -> str:
    assert isinstance(content, TextContent)

    return content.text


async def test_tools_serve_schema_and_retry() -> None:
    async with Client(tools([search, save, look], name="notes")["instance"]) as client:
        listed = {tool.name: tool for tool in (await client.list_tools()).tools}
        result = await client.call_tool("search", {"query": "cbr", "limit": 3})
        retry = await client.call_tool("search", {"query": ""})
        seen = await client.call_tool("look", {"name": "a"})

    assert listed["search"].description == "Searches the notes."
    assert listed["search"].input_schema["required"] == ["query"]
    assert listed["save"].annotations is not None
    assert listed["save"].annotations.destructive_hint is True
    assert text(result.content[0]) == "3 hits for cbr"
    assert retry.is_error
    assert text(retry.content[0]) == "Pass a non-empty query."
    assert [type(block) for block in seen.content] == [TextContent, ImageContent]


async def test_claude_agent_serves_its_tools_with_their_effects() -> None:
    options = ClaudeAgent(name="claude", servers={"notes": [search, save]})._options

    assert isinstance(options.mcp_servers, dict) and set(options.mcp_servers) == {
        "notes"
    }
    assert options.can_use_tool is not None
    assert options.output_format is None
    assert ClaudeAgent(
        name="claude", output=TypeAdapter(list[int])
    )._options.output_format == {
        "type": "json_schema",
        "schema": {"type": "array", "items": {"type": "integer"}},
    }

    decide = options.can_use_tool

    async def decisions(task: object, /) -> list[object]:
        return [
            await decide(f"mcp__notes__{name}", {}, ToolPermissionContext())
            for name in ("search", "save")
        ]

    trace = await run_async(
        decisions, None, RunConfig(policy=EffectPolicy(write="deny", execute="allow"))
    )

    assert [type(decision) for decision in trace.output] == [
        PermissionResultAllow,
        PermissionResultDeny,
    ]


async def test_can_use_tool_maps_the_policy_of_the_run() -> None:
    asked: list[Action] = []

    async def approve(action: Action, /) -> bool:
        asked.append(action)

        return action.name == "save"

    effects: dict[str, Effect] = {"search": "read", "save": "write", "drop": "write"}
    context = ToolPermissionContext()
    decide = can_use_tool(effects)

    async def decisions(task: object, /) -> list[object]:
        return [
            await decide(name, {"table": "notes"}, context)
            for name in ("search", "save", "Bash", "drop")
        ]

    trace = await run_async(
        decisions,
        None,
        RunConfig(policy=EffectPolicy(write="ask", execute="deny"), approve=approve),
    )

    assert [type(decision) for decision in trace.output] == [
        PermissionResultAllow,
        PermissionResultAllow,
        PermissionResultDeny,
        PermissionResultDeny,
    ]
    assert [(a.name, a.details) for a in asked] == [
        ("save", {"table": "notes"}),
        ("drop", {"table": "notes"}),
    ]
