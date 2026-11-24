"""Composable policies that allow, refuse, or ask for approval of actions.

Policies guard the surfaces exar owns: the agent invocations of workflows and processes (`exar.agents.call`),
the tools served over MCP (`exar.mcp`), runs served over A2A (`exar.a2a`), and memory writes (`exar.memory`).
Agent frameworks gate their tools natively, such as pydantic-ai with approvals,
which `exar.pydantic_ai.EffectGate` derives from a policy.

Severity grows from `"allow"` over `"ask"` to `"deny"`, so combining policies keeps the strictest decision.
The configurable policies are the members of `PolicyKind`.

Examples:
    >>> from pydantic import TypeAdapter
    >>> adapter = TypeAdapter(PolicyKind)
    >>> policy = adapter.validate_python({"kind": "strictest", "policies": [{"kind": "effect"}]})
    >>> adapter.dump_python(policy)["policies"]
    [{'kind': 'effect', 'read': 'allow', 'write': 'ask', 'execute': 'ask'}]
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import Field

from .errors import Refused
from .schema import Open, serializable
from .typing import Action, Approve, Decision, Policy

__all__ = [
    "AnyPolicy",
    "EffectPolicy",
    "PolicyKind",
    "Rules",
    "Strictest",
    "allow_all",
    "authorize",
]

SEVERITY: tuple[Decision, ...] = ("allow", "ask", "deny")


async def authorize(
    decision: Decision, action: Action, approve: Approve | None = None, /
) -> None:
    """Lets an action pass if the decision allows it or asks and the approver approves it.

    Raises:
        exar.errors.Refused: If the action does not pass.

    Examples:
        >>> import asyncio
        >>> asyncio.run(authorize("ask", Action(kind="tool", name="rm", effect="write", target="a.txt")))
        Traceback (most recent call last):
        exar.errors.Refused: The policy denied tool 'rm' on 'a.txt'
    """
    if decision == "allow" or (
        decision == "ask" and approve is not None and await approve(action)
    ):
        return

    target = "" if action.target is None else f" on {action.target!r}"

    raise Refused(f"The policy denied {action.kind} {action.name!r}{target}")


def allow_all(action: Action, /) -> Decision:
    """Allows every action.

    Examples:
        >>> allow_all(Action(kind="tool", name="rm", effect="execute"))
        'allow'
    """
    return "allow"


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class EffectPolicy:
    """Decides by what an action changes.

    Args:
        read: Decision for actions that change nothing.
        write: Decision for actions that change a memory.
        execute: Decision for actions that change the environment outside the application.

    Examples:
        >>> policy = EffectPolicy(execute="deny")
        >>> policy(Action(kind="tool", name="search", effect="read")), policy(Action(kind="tool", name="mail", effect="execute"))
        ('allow', 'deny')
    """

    kind: Literal["effect"] = field(default="effect", repr=False)
    read: Decision = "allow"
    write: Decision = "ask"
    execute: Decision = "ask"

    def __call__(self, action: Action, /) -> Decision:
        match action.effect:
            case "read":
                return self.read
            case "write":
                return self.write
            case "execute":
                return self.execute


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class Rules:
    """Decides by the name of the tool or the invoked agent.

    Args:
        by_name: Decisions by name.
        default: Decision for names without a rule.

    Examples:
        >>> policy = Rules(by_name={"delete": "ask"})
        >>> policy(Action(kind="tool", name="delete", effect="write")), policy(Action(kind="tool", name="read", effect="read"))
        ('ask', 'allow')
    """

    kind: Literal["rules"] = field(default="rules", repr=False)
    by_name: Mapping[str, Decision]
    default: Decision = "allow"

    def __call__(self, action: Action, /) -> Decision:
        return self.by_name.get(action.name, self.default)


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class Strictest:
    """Combines policies into the strictest of their decisions, `"allow"` without policies.

    Args:
        policies: Policies to combine.

    Examples:
        >>> policy = Strictest(policies=(EffectPolicy(), Rules(by_name={"rm": "deny"})))
        >>> policy(Action(kind="tool", name="ls", effect="read")), policy(Action(kind="tool", name="cp", effect="write"))
        ('allow', 'ask')
        >>> policy(Action(kind="tool", name="rm", effect="write"))
        'deny'
    """

    kind: Literal["strictest"] = field(default="strictest", repr=False)
    policies: Sequence[AnyPolicy]

    def __call__(self, action: Action, /) -> Decision:
        return SEVERITY[
            max((SEVERITY.index(policy(action)) for policy in self.policies), default=0)
        ]


type PolicyKind = Annotated[
    EffectPolicy | Rules | Strictest, Field(discriminator="kind")
]
"""The configurable policies, `allow_all` and other functions are Python only."""


type AnyPolicy = Open[PolicyKind, Policy]
"""A configurable policy or any other `Policy` of Python code, which serialized forms and JSON Schemas only know as `PolicyKind`."""
