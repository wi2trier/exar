"""Serves exar tools over MCP with FastMCP, such as exar's memory tools, the operations of an API, or cbrkit's operations.

MCP serves tools to clients in other processes, such as Claude Code, Claude Desktop, or IDEs.
Agents in the same process call the functions directly through their framework instead.
The other side, connecting to MCP servers of others, is `exar.mcp_client`.

The tools are a fixed list, or built for each request from the access token of the caller.
A policy hides the exar tools it denies, and a call it asks about waits for the person at the client,
while FastMCP tools apply their own policy, such as tools that run as an agent runs them.
Since the MCP revision of 2026-07-28 removed requests initiated by servers (SEP-2577),
the call asks over the round trip of SEP-2322 that the client drives (`Confirmations`).
It answers with an elicitation of a yes, the client collects the answer and retries the same call with it,
and the retry runs only if it carries the request state the server issued:
an HMAC under a key of the server over the caller, the tool, the arguments, the description shown,
a nonce, and an expiry.
A client cannot forge it, a client whose arguments or description drifted between the two legs is asked again,
and an answer counts once, for its caller, until it expires, so a replayed answer is asked again
rather than running the call twice.
The gate is for the person at the client, not a defense against a hostile client answering its own elicitation,
and a client that ignores the request never reaches the tool, so it fails closed.

Examples:
    >>> from exar.policy import EffectPolicy
    >>> async def add(a: int, b: int) -> int:
    ...     return a + b
    >>> confirmations = Confirmations(key=b"shared by the replicas")
    >>> mcp = server([add], name="calculator", policy=EffectPolicy(), confirmations=confirmations)
"""

import base64
import hashlib
import hmac
import secrets
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, override

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken, AuthProvider
from fastmcp.server.dependencies import get_access_token, get_context
from fastmcp.server.elicitation import handle_elicit_accept, parse_elicit_response_type
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool as FastMCPTool
from fastmcp.tools import ToolResult
from fastmcp.tools.base import InputRequiredToolResult
from fastmcp.utilities.versions import VersionSpec, version_sort_key
from mcp.types import (
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
    ToolAnnotations,
)
from pydantic import Field, ValidationError
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import to_json, to_jsonable_python

from ._mcp_content import tool_content
from .helpers import get_name
from .text import as_text
from .tools import SchemaTool, effect_hints, schema_tool, tool_action, translate_retry
from .tools.batch import unpack
from .typing import Effect, Policy, Tool

__all__ = [
    "Confirmations",
    "Describe",
    "InMemoryNonces",
    "Nonces",
    "ToolFactory",
    "annotated_effect",
    "annotations",
    "describe_call",
    "server",
    "tool_content",
]

type ToolFactory = Callable[
    [AccessToken | None], Awaitable[Iterable[Tool | FastMCPTool]]
]
"""Builds the tools of a request from the access token of the caller, `None` without authentication."""

_RESPONSE_KEY = "confirmation"

_CONFIRMATION = parse_elicit_response_type(
    bool, response_title="Allow", response_description="Approve this call."
)


class Describe(Protocol):
    """Describes a call for the person who approves it, such as the changes it would make."""

    async def __call__(self, name: str, arguments: Mapping[str, object], /) -> str: ...


async def describe_call(name: str, arguments: Mapping[str, object], /) -> str:
    """Asks about a call by its tool and arguments, the default `Describe`.

    Examples:
        >>> import asyncio
        >>> asyncio.run(describe_call("send", {"to": "ada"}))
        'Allow send with {"to":"ada"}?'
    """
    return f"Allow {name} with {as_text(dict(arguments))}?"


def annotations(effect: Effect, /) -> ToolAnnotations:
    """The MCP annotations a tool of an effect is announced with, see `exar.tools.effect_hints`.

    Examples:
        >>> annotations("read").read_only_hint, annotations("execute").open_world_hint
        (True, True)
    """
    return ToolAnnotations.model_validate(effect_hints(effect))


def annotated_effect(hints: ToolAnnotations | None, /) -> Effect:
    """The effect of a tool of another server from its annotations, the inverse of `annotations`.

    A read-only tool reads, one neither destructive nor reaching beyond its server writes,
    and any other executes, the hints defaulting to the pessimistic values of the specification.
    Hints are claims of the server, so only those of a trusted server should count.

    Examples:
        >>> annotated_effect(ToolAnnotations(read_only_hint=True)), annotated_effect(None)
        ('read', 'execute')
        >>> annotated_effect(ToolAnnotations(destructive_hint=False, open_world_hint=False))
        'write'
    """
    if hints is None:
        return "execute"

    if hints.read_only_hint:
        return "read"

    if hints.destructive_hint is False and hints.open_world_hint is False:
        return "write"

    return "execute"


def _allowed(answer: ElicitResult, /) -> bool:
    if answer.action != "accept":
        return False

    try:
        return handle_elicit_accept(_CONFIRMATION, answer.content).data is True
    except (ValidationError, ValueError) as error:
        raise ToolError("The confirmation response was malformed.") from error


def _principal() -> tuple[str, str] | None:
    """Who calls, the subject and client of the access token, `None` without authentication."""
    token = get_access_token()

    if token is None:
        return None

    return str(token.claims.get("sub", "")), token.client_id


class Nonces(Protocol):
    """Spends the nonce of each answered confirmation once, see `Confirmations`."""

    async def spend(self, nonce: str, expiry: int, /) -> bool:
        """Marks a nonce spent until its expiry, a Unix time in seconds.

        Returns:
            Whether the nonce was unspent, atomically, so of two concurrent spends one wins.
        """
        ...


@dataclass(slots=True, eq=False)
class InMemoryNonces:
    """The spent nonces of one process, forgotten as they expire, for a server of a single process.

    Examples:
        >>> import asyncio
        >>> nonces = InMemoryNonces()
        >>> expiry = int(time.time()) + 60
        >>> asyncio.run(nonces.spend("n", expiry)), asyncio.run(nonces.spend("n", expiry))
        (True, False)
    """

    _spent: dict[str, int] = field(default_factory=dict[str, int], init=False)

    async def spend(self, nonce: str, expiry: int, /) -> bool:
        """Marks a nonce spent until its expiry, see `Nonces`."""
        now = time.time()
        self._spent = {key: until for key, until in self._spent.items() if until > now}

        if nonce in self._spent:
            return False

        self._spent[nonce] = expiry

        return True


@dataclass(frozen=True, slots=True, kw_only=True)
class Confirmations:
    """Asks the client to approve calls over the round trip of SEP-2322, each answer bound to its call and used once.

    The request state is an HMAC under `key` over the caller, the tool, the arguments, the description,
    a nonce, and an expiry, so a client can neither forge it nor carry an answer over to another call,
    caller, or description, and an answer expires after `ttl` seconds.
    An answer counts once, as `nonces` spends its nonce, so repeating it asks again rather than running the call twice.
    The state holds across the replicas and restarts of a server that share the key,
    and the spent nonces as far as they share the store of `nonces`, such as a database.

    Args:
        key: The key of the HMAC, shared by the replicas of a server that serve one client.
        ttl: Seconds an answer counts for.
        nonces: The spent nonces, those of this process by default.
    """

    key: bytes = field(repr=False)
    ttl: float = 600
    nonces: Nonces = field(default_factory=InMemoryNonces)

    def _mac(
        self,
        name: str,
        arguments: Mapping[str, object],
        description: str,
        nonce: str,
        expiry: int,
    ) -> str:
        message = to_json(
            (_principal(), name, dict(arguments), description, nonce, expiry),
            fallback=str,
        )

        return hmac.new(self.key, message, hashlib.sha256).hexdigest()

    def _issue(
        self, name: str, arguments: Mapping[str, object], description: str
    ) -> str:
        nonce = base64.urlsafe_b64encode(secrets.token_bytes(16)).decode()
        expiry = int(time.time() + self.ttl)
        mac = self._mac(name, arguments, description, nonce, expiry)

        return f"{nonce}.{expiry}.{mac}"

    def _verified(
        self,
        state: str | None,
        name: str,
        arguments: Mapping[str, object],
        description: str,
    ) -> tuple[str, int] | None:
        """The nonce and expiry of a state issued for the call that has not expired, `None` otherwise."""
        match (state or "").split("."):
            case [nonce, expiry, mac] if expiry.isascii() and expiry.isdigit():
                expected = self._mac(name, arguments, description, nonce, int(expiry))

                if int(expiry) > time.time() and hmac.compare_digest(mac, expected):
                    return nonce, int(expiry)

                return None
            case _:
                return None

    async def __call__(
        self, name: str, arguments: Mapping[str, object], description: str, /
    ) -> InputRequiredResult | None:
        """Asks the client to approve a call of the current request.

        A tool returns the request to ask, FastMCP passes it to the client,
        and the client retries the same call with the answer, which this function then reads and spends.

        Args:
            name: Name of the tool.
            arguments: Arguments of the call.
            description: What the person approves, such as the changes the call would make.

        Returns:
            The request to return from the tool, or `None` once the person approved the call.

        Raises:
            fastmcp.exceptions.ToolError: If the person refused, or the answer is malformed.
        """
        ctx = get_context()
        answer = (ctx.input_responses or {}).get(_RESPONSE_KEY)

        if (
            isinstance(answer, ElicitResult)
            and (
                verified := self._verified(
                    ctx.request_state, name, arguments, description
                )
            )
            is not None
            and await self.nonces.spend(*verified)
        ):
            if _allowed(answer):
                return None

            raise ToolError("The call was denied by the user.")

        return InputRequiredResult(
            input_requests={
                _RESPONSE_KEY: ElicitRequest(
                    params=ElicitRequestFormParams(
                        message=description, requested_schema=_CONFIRMATION.schema
                    )
                )
            },
            request_state=self._issue(name, arguments, description),
        )


class _Served(FastMCPTool):
    """An exar tool served over MCP: the text and files of its result as content and its data as the structured content."""

    call: Annotated[
        SkipJsonSchema[Callable[..., Awaitable[object]]], Field(exclude=True)
    ]

    @override
    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        result = await self.call(**arguments)

        if isinstance(result, InputRequiredResult):
            return InputRequiredToolResult(result)

        unpacked = unpack(result)
        data = to_jsonable_python(unpacked.data, fallback=str)

        return ToolResult(
            content=tool_content(unpacked),
            structured_content=data
            if data is None or isinstance(data, dict)
            else {"result": data},
        )


def _asking(
    tool: SchemaTool, describe: Describe, confirm: Confirmations, /
) -> Callable[..., Awaitable[object]]:
    async def call(**arguments: object) -> object:
        if (
            ask := await confirm(
                tool.name, arguments, await describe(tool.name, arguments)
            )
        ) is not None:
            return ask

        return await tool.call(**arguments)

    return call


def _served(
    tool: Tool | FastMCPTool,
    policy: Policy,
    describe: Describe,
    confirm: Confirmations,
    /,
) -> FastMCPTool | None:
    if isinstance(tool, FastMCPTool):
        return tool

    served = schema_tool(tool)

    match policy(tool_action(served.name, served.effect)):
        case "deny":
            return None

        case "ask":
            call = _asking(served, describe, confirm)

        case "allow":
            call = served.call

    return _Served(
        name=served.name,
        description=served.description,
        parameters=dict(served.schema),
        annotations=annotations(served.effect),
        call=translate_retry(call, ToolError),
    )


class _PerRequest(Provider):
    """Serves the tools a factory builds for each request."""

    def __init__(
        self,
        tools: ToolFactory,
        serve: Callable[[Tool | FastMCPTool], FastMCPTool | None],
        /,
    ) -> None:
        super().__init__()
        self.tools = tools
        self.serve = serve

    async def _built(self, name: str | None = None) -> list[FastMCPTool]:
        """The served tools of the request, only those of a name if given, since serving a tool builds its schema."""
        built = await self.tools(get_access_token())

        return [
            served
            for tool in built
            if name is None or _name(tool) == name
            if (served := self.serve(tool)) is not None
        ]

    @override
    async def _list_tools(self) -> Sequence[FastMCPTool]:
        return await self._built()

    @override
    async def _get_tool(
        self, name: str, version: VersionSpec | None = None
    ) -> FastMCPTool | None:
        matching = [
            tool
            for tool in await self._built(name)
            if not version or version.matches(tool.version)
        ]

        return max(matching, key=version_sort_key, default=None)


def _name(tool: Tool | FastMCPTool, /) -> str:
    return tool.name if isinstance(tool, FastMCPTool) else get_name(tool)


def server(
    tools: Iterable[Tool | FastMCPTool] | ToolFactory,
    /,
    *,
    name: str,
    policy: Policy,
    describe: Describe = describe_call,
    confirmations: Confirmations,
    auth: AuthProvider | None = None,
    instructions: str | None = None,
) -> FastMCP:
    """Serves exar tools as an MCP server, each as its `exar.tools.schema_tool`.

    The input schema and the description come from the signature and the docstring of a plain tool as pydantic-ai reads them,
    or from the data of a `exar.tools.SchemaTool`, the annotations from the effect,
    and `exar.tools.Retry`, such as for invalid arguments, becomes a tool error the model can react to.
    A result is answered with its text and files as the content (`tool_content`) and its data as the structured content.
    FastMCP tools are served as they are, such as tools of a JSON schema,
    their effects and approvals are their own, which they can ask for with `Confirmations`.

    Args:
        tools: exar tools or FastMCP tools,
            or a factory that builds them for each request from the access token of the caller.
        name: Name of the server.
        policy: Hides the tools it denies, and asks the person at the client to approve each call it asks about.
        describe: Describes a call the policy asks about for the person who approves it.
        confirmations: Asks the person at the client.
        auth: Authenticates the callers, whose access token the factory receives.
        instructions: How to use the server, for the model of the client.

    Examples:
        >>> from exar.policy import EffectPolicy
        >>> from exar.tools import effect
        >>> @effect("execute")
        ... async def send_mail(to: str) -> str:
        ...     return f"Sent to {to}."
        >>> confirmations = Confirmations(key=b"shared by the replicas")
        >>> mcp = server(
        ...     [send_mail], name="mail", policy=EffectPolicy(execute="deny"), confirmations=confirmations
        ... )
        >>> async def per_caller(token: AccessToken | None) -> list[Tool]:
        ...     return [] if token is None else [send_mail]
        >>> mcp = server(per_caller, name="mail", policy=EffectPolicy(), confirmations=confirmations)
    """

    def serve(tool: Tool | FastMCPTool, /) -> FastMCPTool | None:
        return _served(tool, policy, describe, confirmations)

    if isinstance(tools, Iterable):
        served = [tool for tool in map(serve, tools) if tool is not None]

        return FastMCP(name, instructions, auth=auth, tools=served)

    return FastMCP(name, instructions, auth=auth, providers=[_PerRequest(tools, serve)])
