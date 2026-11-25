"""exar tools as a pydantic-ai toolset with their effects as tool metadata, a gate that applies a policy to them,
and the declared effects of tools built elsewhere."""

import functools
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace

from pydantic_ai import (
    ApprovalRequired,
    FunctionToolset,
    ModelRetry,
    RunContext,
    ToolCallPart,
    ToolDenied,
    ToolReturn,
)
from pydantic_ai import Tool as PydanticAITool
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
)
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.tools import ToolDefinition

from exar._effect import EFFECTS, declared
from exar.agents import decide
from exar.tools import SchemaTool, effect_of, tool_action, translate_retry
from exar.tools.batch import as_return
from exar.typing import Action, Effect, Policy, Tool

__all__ = [
    "EFFECT",
    "DeclaredEffects",
    "EffectGate",
    "definition_action",
    "tool_effect",
    "toolset",
]

EFFECT = "effect"
"""Metadata key of the effect of a tool, as `toolset` declares it and tool selectors such as reviewers match it."""


def tool_effect(metadata: Mapping[str, object] | None, /) -> Effect:
    """The effect the metadata of a pydantic-ai tool declares under `EFFECT`, `"execute"` if it declares none.

    As `exar.tools.effect_of`, it fails closed.
    Declare the effects of third-party tools on their toolset, such as `server.with_metadata(effect="read")`,
    or with `DeclaredEffects` for tools a capability adds, such as those of pydantic-ai-harness,
    not with the `SetToolMetadata` capability of pydantic-ai, whose wrapper toolset applies its metadata
    only after the `prepare_tools` hooks of capabilities such as `EffectGate` ran.

    Raises:
        ValueError: If it declares an unknown effect.

    Examples:
        >>> tool_effect({"effect": "write"}), tool_effect(None)
        ('write', 'execute')
    """
    return declared(
        (metadata or {}).get(EFFECT, "execute"), EFFECTS, "effect", metadata
    )


def _returning[**P](
    func: Callable[P, Awaitable[object]], /
) -> Callable[P, Awaitable[ToolReturn[object]]]:
    """Wraps a tool so it returns a `ToolReturn`, as pydantic-ai reads a plain value, so hooks and history see one shape."""

    @functools.wraps(func)
    async def call(*args: P.args, **kwargs: P.kwargs) -> ToolReturn[object]:
        return as_return(await func(*args, **kwargs))

    return call


def _pydantic_ai_tool[D](tool: Tool, /) -> PydanticAITool[D]:
    """A pydantic-ai tool of an exar tool, its schema the signature of a plain tool or the data of a `SchemaTool`."""
    if isinstance(tool, SchemaTool):
        wrapped = PydanticAITool[D].from_schema(
            _returning(translate_retry(tool.call, ModelRetry)),
            name=tool.name,
            description=tool.description,
            json_schema=dict(tool.schema),
        )
        wrapped.metadata = {EFFECT: tool.effect}

        return wrapped

    return PydanticAITool[D](
        _returning(translate_retry(tool, ModelRetry)),
        metadata={EFFECT: effect_of(tool)},
    )


def toolset[D](
    tools: Iterable[Tool], /, *, instructions: str | None = None
) -> FunctionToolset[D]:
    """Turns exar tools into a pydantic-ai toolset, which carries the `instructions` for the model.

    `exar.tools.Retry` becomes `ModelRetry`, the effect of a tool its `EFFECT` metadata,
    and a plain result a `ToolReturn` (`exar.tools.batch.as_return`), so hooks read every result with `exar.tools.batch.unpack`.
    A plain tool is a native function tool, which pydantic-ai validates with the typed arguments its hooks see,
    and its schema and description are those of `exar.tools.schema_tool`, which builds them the same way for the other integrations.
    A `SchemaTool`, such as an operation of `exar.tools.http.OpenApi`, serves its schema as it is,
    and pydantic-ai does not validate the arguments against it.

    Examples:
        >>> from exar.tools import effect
        >>> @effect("execute")
        ... async def mail(to: str) -> str:
        ...     return to
        >>> toolset([mail]).tools["mail"].metadata
        {'effect': 'execute'}
    """
    return FunctionToolset(
        [_pydantic_ai_tool(tool) for tool in tools], instructions=instructions
    )


def definition_action(tool_def: ToolDefinition, args: object = None, /) -> Action:
    """The `exar.tools.tool_action` of calling a pydantic-ai tool with the arguments, as `EffectGate` hands it to its policy.

    It carries the effect of the tool (`tool_effect`), its metadata, and the arguments as details.

    Examples:
        >>> action = definition_action(ToolDefinition(name="mail", metadata={"effect": "read", "trusted": True}))
        >>> action.effect, action.metadata["trusted"]
        ('read', True)
    """
    return tool_action(
        tool_def.name, tool_effect(tool_def.metadata), args, tool_def.metadata
    )


def _run_policy[D](ctx: RunContext[D], /) -> Policy:
    """The policy of the exar run the agent runs in, see `exar.agents.decide`."""
    return decide


@dataclass
class EffectGate[D](AbstractCapability[D]):
    """Applies the policy of each run to the tools of a pydantic-ai agent by their effects, with native approvals.

    The policy of a run is that of the exar run it runs in, or else comes from its context, so it can depend on the state of the run,
    such as its deps, and it decides on the `definition_action` of each tool, so it can depend on what the tool declares in its metadata.
    It hides the tools the policy denies, denies each call the policy denies by its arguments,
    and raises `ApprovalRequired` for the calls it asks about,
    so pydantic-ai ends the run with `DeferredToolRequests` or hands them to an approver,
    such as the `ApprovalReviewer` of pydantic-ai-harness or `HandleDeferredToolCalls`.
    A tool without `EFFECT` metadata counts as `"execute"`,
    see `tool_effect` and `DeclaredEffects` for how to declare the effects of third-party tools.

    Args:
        policy: The policy of a run by its context, that of the exar run (`exar.agents.decide`) by default,
            which raises outside one, or such as `lambda ctx: ctx.deps.policy` for a policy of the deps.

    Examples:
        >>> from pydantic_ai import Agent
        >>> from exar.policy import EffectPolicy
        >>> agent = Agent("test", capabilities=[EffectGate()])
        >>> agent = Agent("test", capabilities=[EffectGate(lambda _: EffectPolicy(execute="deny"))])
    """

    policy: Callable[[RunContext[D]], Policy] = _run_policy

    async def prepare_tools(
        self, ctx: RunContext[D], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        policy = self.policy(ctx)

        return [
            replace(
                tool_def,
                metadata={
                    **(tool_def.metadata or {}),
                    EFFECT: tool_effect(tool_def.metadata),
                },
            )
            for tool_def in tool_defs
            if policy(definition_action(tool_def)) != "deny"
        ]

    async def before_tool_execute(
        self,
        ctx: RunContext[D],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        match self.policy(ctx)(definition_action(tool_def, args)):
            case "deny":
                raise SkipToolExecution(
                    ToolDenied(f"The policy denied the call of {tool_def.name!r}.")
                )
            case "ask" if not ctx.tool_call_approved:
                raise ApprovalRequired
            case _:
                return args


@dataclass
class DeclaredEffects[D](AbstractCapability[D]):
    """Declares the effects of tools it does not build, such as those a capability of pydantic-ai-harness adds.

    It sets their `EFFECT` metadata in its `prepare_tools` hook, which wraps the hooks that offer tools by it,
    since `SetToolMetadata` applies its metadata only after every `prepare_tools` hook ran.

    Args:
        effects: The effect of each tool by its name.

    Examples:
        >>> from pydantic_ai import Agent
        >>> from exar.policy import EffectPolicy
        >>> agent = Agent(
        ...     "test",
        ...     capabilities=[EffectGate(), DeclaredEffects({"read_tool_result": "read"})],
        ... )
    """

    effects: Mapping[str, Effect]

    def get_ordering(self) -> CapabilityOrdering:
        """Wraps `EffectGate`, so it sees the effects."""
        return CapabilityOrdering(wraps=[EffectGate])

    async def prepare_tools(
        self, ctx: RunContext[D], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            replace(
                tool_def,
                metadata={
                    **(tool_def.metadata or {}),
                    EFFECT: self.effects[tool_def.name],
                },
            )
            if tool_def.name in self.effects
            else tool_def
            for tool_def in tool_defs
        ]
