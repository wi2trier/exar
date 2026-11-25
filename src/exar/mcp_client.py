"""Connects to external MCP servers once for the whole process and offers their tools to pydantic-ai runs.

`McpClient` holds one `MCPToolset` entered while `supervised` runs, reconnecting with a growing backoff
after a failure.
A run never connects itself: the toolset of `McpClient.agent_toolset` offers the tools the server listed
while it is reachable, none and a short notice while it is not,
and a call the connection fails returns a failed result,
so an unreachable server withholds its own tools and fails no turn.

Every tool declares the effect the host derives its offering and approval from (`exar.pydantic_ai.EFFECT`):
the one `effects` names for it, else what the hints of a `trusted` server say (`exar.mcp.annotated_effect`),
or `execute` when it is not trusted.
The server's tool list is asked for again every `refresh_seconds`, since not every server announces a change.

Examples:
    >>> from fastmcp import FastMCP
    >>> client = McpClient(name="notes", toolset=MCPToolset(FastMCP("notes")), blocked={"drop"})
    >>> client.tools is None
    True
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Collection, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, NoReturn, Self

from mcp import MCPError
from mcp import types as mcp_types
from mcp.types import CONNECTION_CLOSED
from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool, WrapperToolset

from .mcp import annotated_effect
from .pydantic_ai import EFFECT
from .text import truncate_line
from .tools.batch import as_return
from .typing import Effect

__all__ = ["McpClient"]

logger = logging.getLogger(__name__)

_MIN_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_STABLE_SECONDS = _MAX_BACKOFF_SECONDS
"""How long a connection stays up before a loss of it starts the backoff over."""
_MAX_ERROR_CHARS = 200


def _lost_connection(exc: Exception) -> Exception | None:
    """The failure of the connection behind *exc*, `None` for a failed call.

    pydantic-ai reports a call the server refused to the model already, but
    as a retry for a closed connection as well, which no retry can fix.
    """
    if not isinstance(exc, ModelRetry | ToolFailed):
        return exc

    cause = exc.__cause__

    if isinstance(cause, MCPError) and cause.error.code == CONNECTION_CLOSED:
        return cause

    return None


@dataclass(slots=True, eq=False, kw_only=True)
class McpClient[D]:
    """One server, its connection, and what it offers while connected.

    Args:
        name: Name of the server, which prefixes its tools in a run.
        toolset: The connection to the server.
        allowed: The tools offered by their names on the server, all when `None`.
        blocked: The tools never offered.
        effects: The effects of tools by their names on the server, overriding the hints of the server.
        trusted: Whether the hints and instructions of the server count.
        refresh_seconds: Seconds between asking the server for its tools again.
        caller: What a call connects through instead, one connection each, for a server told who calls.
        unavailable: Words the notice of a server without tools for a run.

    Attributes:
        tools: The offered tools as the server lists them by name, `None` while the server is unreachable.
        error: Why the server last failed.
    """

    name: str
    toolset: MCPToolset[D]
    allowed: Collection[str] | None = None
    blocked: Collection[str] = ()
    effects: Mapping[str, Effect] = field(default_factory=dict[str, Effect])
    trusted: bool = False
    refresh_seconds: float = 300.0
    caller: MCPToolset[D] | None = None
    unavailable: Callable[[RunContext[D]], str] | None = None
    tools: Mapping[str, mcp_types.Tool] | None = field(default=None, init=False)
    error: str | None = field(default=None, init=False)
    _lost: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def agent_toolset(self) -> AbstractToolset[D]:
        """The server's tools as a run offers them, prefixed by its name."""
        return _McpToolset(self.toolset, self).prefixed(self.name)

    def effect_of(self, tool: mcp_types.Tool, /) -> Effect:
        """The effect of a tool of the server, see the module."""
        if (declared := self.effects.get(tool.name)) is not None:
            return declared

        # Hints are the server's claims, which count only for a trusted one.
        return annotated_effect(tool.annotations) if self.trusted else "execute"

    def _offers(self, name: str) -> bool:
        return (
            self.allowed is None or name in self.allowed
        ) and name not in self.blocked

    def offer(self, tools: Sequence[mcp_types.Tool]) -> None:
        """Offer what the server listed, naming the configured tools it lacks."""
        listed = {tool.name for tool in tools}
        named = {*(self.allowed or ()), *self.blocked, *self.effects}

        if missing := sorted(named - listed):
            logger.warning(
                "MCP server %s lists none of the configured tools %s",
                self.name,
                ", ".join(missing),
            )

        self.tools = {tool.name: tool for tool in tools if self._offers(tool.name)}
        self.error = None
        logger.info("MCP server %s connected with %d tools", self.name, len(self.tools))

    def lose(self, exc: Exception) -> None:
        """Withhold the tools after *exc* and have the connection rebuilt."""
        error = truncate_line(f"{type(exc).__name__}: {exc}", _MAX_ERROR_CHARS)

        if error != self.error:
            logger.warning("MCP server %s is unreachable: %s", self.name, error)

        self.tools = None
        self.error = error
        self._lost.set()

    async def _until_stale(self) -> None:
        """Wait until the connection is lost or the server lists other tools.

        A change is taken up by connecting again, which lists the tools afresh.
        """
        offered = await self.toolset.list_tools()

        while True:
            try:
                async with asyncio.timeout(self.refresh_seconds):
                    _ = await self._lost.wait()

                return

            except TimeoutError:
                if await self.toolset.client.list_tools() != offered:
                    logger.info("MCP server %s changed its tools", self.name)
                    return

    async def supervise(self) -> NoReturn:
        """Hold the connection open, reconnecting with a growing backoff once lost.

        The backoff starts over only after a connection stayed up for `_STABLE_SECONDS`,
        so a server that accepts and then drops every connection backs off too.
        """
        loop = asyncio.get_running_loop()
        delay = _MIN_BACKOFF_SECONDS

        while True:
            self._lost.clear()
            connected: float | None = None

            try:
                async with self.toolset:
                    self.offer(await self.toolset.list_tools())
                    connected = loop.time()
                    await self._until_stale()
            except Exception as exc:  # noqa: BLE001
                self.lose(exc)

            if connected is not None and loop.time() - connected >= _STABLE_SECONDS:
                delay = _MIN_BACKOFF_SECONDS

            await asyncio.sleep(delay)
            delay = min(2 * delay, _MAX_BACKOFF_SECONDS)

    @asynccontextmanager
    async def supervised(self) -> AsyncGenerator[None]:
        """Supervise the connection in the background, never waiting for it to be up."""
        task = asyncio.create_task(self.supervise())

        try:
            yield
        finally:
            _ = task.cancel()
            _ = await asyncio.gather(task, return_exceptions=True)


@dataclass
class _McpToolset[D](WrapperToolset[D]):
    """A server's tools in a run, each declaring its effect.

    The connection is the client's own, so entering this toolset enters
    nothing, and a run never waits on a server that is down.
    """

    client: McpClient[D]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        return None

    async def get_instructions(
        self, ctx: RunContext[D]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        """The instructions of a trusted server, or why a server offers no tools.

        A trusted server's instructions are operator text, appended as a
        change when a later connection brings others, while an untrusted
        server's never reach the model.
        """
        client = self.client

        if client.tools is None:
            if client.unavailable is not None:
                return client.unavailable(ctx)

            return (
                f"The tools of the MCP server {client.name} are unavailable right now."
            )

        if not client.trusted:
            return None

        return await self.wrapped.get_instructions(ctx)

    async def get_tools(self, ctx: RunContext[D]) -> dict[str, ToolsetTool[D]]:
        """The offered tools the server lists, none while it is unreachable."""
        offered = self.client.tools

        if offered is None:
            return {}

        try:
            listed = await self.wrapped.get_tools(ctx)
        except Exception as exc:  # noqa: BLE001
            self.client.lose(exc)
            return {}

        return {
            name: replace(
                tool,
                toolset=self,
                tool_def=replace(
                    tool.tool_def,
                    metadata={
                        **(tool.tool_def.metadata or {}),
                        EFFECT: self.client.effect_of(offered[name]),
                    },
                ),
            )
            for name, tool in listed.items()
            if name in offered
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[D],
        tool: ToolsetTool[D],
    ) -> ToolReturn[Any]:
        """Call the tool, a failed connection failing the call rather than the run, and its result a `ToolReturn`.

        A server with a `caller` is called on a connection of the call's own.
        """
        try:
            if (caller := self.client.caller) is None:
                return as_return(
                    await self.wrapped.call_tool(name, tool_args, ctx, tool)
                )

            connection = await caller.for_run(ctx)

            async with connection:
                return as_return(await connection.call_tool(name, tool_args, ctx, tool))

        except Exception as exc:
            if (lost := _lost_connection(exc)) is None:
                raise

            self.client.lose(lost)
            raise ToolFailed(
                f"The MCP server {self.client.name} could not be reached, the call did not complete."
            ) from exc
