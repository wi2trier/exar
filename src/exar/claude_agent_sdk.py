"""Natively configured Claude Agent SDK sessions as exar agents, with exar tools and policies.

`ClaudeAgent` runs a task as a Claude Code session of natively configured `ClaudeAgentOptions`, with text or structured output,
`tools` serves exar tools as an in-process MCP server, which `ClaudeAgent` does for its `servers` with their effects,
and `can_use_tool` maps the policy of the run to the SDK's permission callback,
asking the approver of the run (`exar.agents.approve`) about the calls the policy asks about.
The SDK emits no OpenTelemetry spans of its own here, so `ClaudeAgent` records each session as one `chat` span
with the usage and cost the SDK reports, and counts them toward the usage and the limits of the run (`exar.agents.spend`).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from claude_agent_sdk import (
    CanUseTool,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    McpSdkServerConfig,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SdkMcpTool,
    ToolAnnotations,
    ToolPermissionContext,
    create_sdk_mcp_server,
)
from pydantic import TypeAdapter

from ._mcp_content import tool_content
from .agents import approve, decide, spend
from .errors import AgentError, Refused
from .helpers import get_name
from .otel import Usage, handler
from .policy import authorize
from .text import TEXT
from .tools import effect_hints, effect_of, schema_tool, tool_action
from .tools.batch import unpack
from .typing import Effect, Tool

__all__ = ["ClaudeAgent", "can_use_tool", "tools"]


def _tool(tool: Tool, /) -> SdkMcpTool[Any]:
    served = schema_tool(tool)

    # The SDK answers an exception, such as `exar.tools.Retry`, with an error result the model can react to.
    async def handler(args: dict[str, Any], /) -> dict[str, Any]:
        result = unpack(await served.call(**args))
        content = [
            block.model_dump(mode="json", by_alias=True, exclude_none=True)
            for block in tool_content(result)
        ]

        return {"content": content}

    return SdkMcpTool(
        name=served.name,
        description=served.description,
        input_schema=dict(served.schema),
        handler=handler,
        annotations=ToolAnnotations.model_validate(effect_hints(served.effect)),
    )


def tools(funcs: Iterable[Tool], /, *, name: str) -> McpSdkServerConfig:
    """Serves exar tools as an in-process MCP server for `ClaudeAgentOptions.mcp_servers`, each as its `exar.tools.schema_tool`.

    The annotations come from the effect, a result is answered with its text and images (`exar.mcp.tool_content`),
    and `exar.tools.Retry`, such as for invalid arguments, with an error result the model can react to.
    Claude calls the tools as `mcp__<name>__<function name>`.

    Args:
        funcs: The tools.
        name: Name of the server, the key in `mcp_servers`.
    """
    return create_sdk_mcp_server(name, tools=[_tool(func) for func in funcs])


def can_use_tool(effects: Mapping[str, Effect] | None = None, /) -> CanUseTool:
    """Maps the decisions of the policy of the current run to the SDK's permission callback.

    Claude Code consults the callback only for calls its own permission settings do not allow already.
    A decision to ask goes to the approver of the run with the arguments as details,
    a refusal becomes a message to the model, see `exar.policy.authorize`.

    Args:
        effects: Effects by tool name as Claude reports it,
            calls of unlisted tools are assumed to change the environment.
    """
    known = effects or {}

    async def decide_call(
        name: str, input: dict[str, Any], context: ToolPermissionContext, /
    ) -> PermissionResult:
        action = tool_action(name, known.get(name, "execute"), input)

        try:
            await authorize(decide(action), action, approve)
        except Refused as error:
            return PermissionResultDeny(message=str(error))

        return PermissionResultAllow()

    return decide_call


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaudeAgent[O = str]:
    """Runs a prompt as a Claude Code session, whose result is the output.

    The output is the text of the result by default, and any other `output` asks the session
    for structured output of its JSON Schema (`ClaudeAgentOptions.output_format`) and validates it.
    The permission callback of the options applies the policy of the run, see `can_use_tool`,
    by the effects the exar tools of `servers` declare and those of `effects`.
    The session is recorded as one `chat` span with the usage and cost the SDK reports for all its model calls.

    Args:
        name: Name of the agent in traces.
        output: Validates the output, the text of the result by default.
        options: Options of the sessions, such as the model, tools, MCP servers, and permissions.
        servers: exar tools by the name of the in-process MCP server that serves them (`tools`),
            beside the MCP servers of the options, which must then be a mapping.
        effects: Effects by tool name as Claude reports it for the policy, for the other tools, such as those of Claude Code,
            calls of tools without an effect are assumed to change the environment.
        effect: What a session may change, which policies check when a workflow invokes it.
    """

    name: str
    output: TypeAdapter[O] = TEXT
    options: ClaudeAgentOptions = field(default_factory=ClaudeAgentOptions)
    servers: Mapping[str, Sequence[Tool]] = field(
        default_factory=dict[str, Sequence[Tool]]
    )
    effects: Mapping[str, Effect] = field(default_factory=dict[str, Effect])
    effect: Effect = "execute"
    _options: ClaudeAgentOptions = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Builds the options of the sessions with the output format, the `servers`, and the permission callback of the effects of every tool.

        Raises:
            TypeError: If there are `servers` and the MCP servers of the options are a file.
        """
        options = self.options
        effects: dict[str, Effect] = {}

        if self.output is not TEXT:
            options = replace(
                options,
                output_format={
                    "type": "json_schema",
                    "schema": self.output.json_schema(),
                },
            )

        if self.servers:
            configured = options.mcp_servers

            if not isinstance(configured, Mapping):
                raise TypeError(
                    "The servers of a ClaudeAgent need the MCP servers of its options as a mapping"
                )

            effects = {
                f"mcp__{server}__{get_name(tool)}": effect_of(tool)
                for server, funcs in self.servers.items()
                for tool in funcs
            }
            servers = {
                server: tools(funcs, name=server)
                for server, funcs in self.servers.items()
            }
            options = replace(options, mcp_servers={**configured, **servers})

        object.__setattr__(
            self,
            "_options",
            replace(options, can_use_tool=can_use_tool({**effects, **self.effects})),
        )

    async def __call__(self, task: str, /) -> O:
        result: ResultMessage | None = None
        options = self._options

        with handler().inference("anthropic", request_model=options.model) as chat:
            async with ClaudeSDKClient(options) as client:
                await client.query(task)

                async for message in client.receive_response():
                    if isinstance(message, ResultMessage):
                        result = message

            if result is not None:
                tokens = result.usage or {}
                cached = _count(tokens, "cache_read_input_tokens")
                written = _count(tokens, "cache_creation_input_tokens")
                cost = result.total_cost_usd
                usage = Usage(
                    input_tokens=_count(tokens, "input_tokens") + cached + written,
                    output_tokens=_count(tokens, "output_tokens"),
                    cache_read_input_tokens=cached,
                    cache_write_input_tokens=written,
                    requests=1,
                    cost=None if cost is None else Decimal(str(cost)),
                )
                chat.input_tokens = usage.input_tokens
                chat.output_tokens = usage.output_tokens
                chat.cache_read_input_tokens = cached
                chat.cache_write_input_tokens = written

                if cost is not None:
                    chat.attributes["operation.cost"] = cost

                spend(usage)

        if result is None:
            raise AgentError("Claude ended without a result")

        if result.is_error:
            raise AgentError(
                f"Claude ended with {result.subtype}: {result.errors or result.result}"
            )

        return self.output.validate_python(
            result.result or "" if self.output is TEXT else result.structured_output
        )


def _count(usage: Mapping[str, Any], key: str, /) -> int:
    value = usage.get(key, 0)

    return int(value) if isinstance(value, int | float) else 0
