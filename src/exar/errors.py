"""Errors of exar, distinguishing refused requests and conflicting writes from failed agents.

Hosts report `Refused` and `Conflict` to the client, since a request caused them,
and every other exception as a failure.
`exar.cbrkit` translates the errors of cbrkit into these, so hosts catch one hierarchy.
"""

from typing import Any

from .typing import Trace

__all__ = [
    "AgentError",
    "AgentFailed",
    "Conflict",
    "ExarError",
    "MissingExtra",
    "Refused",
    "Unbound",
]


class ExarError(Exception):
    """Base class of the errors that a request causes."""


class Refused(ExarError, ValueError):
    """A policy or limit refused an action, such as spawning an agent beyond the allowed depth."""


class Conflict(ExarError):
    """A memory holds other entries than those a write was planned on, so nothing was written."""


class Unbound(ExarError, LookupError):
    """A function of the host that is referred to by name, but whose name is bound to no function."""


class MissingExtra(ModuleNotFoundError):
    """An optional dependency that is not installed, the message names the extra that installs it.

    It is a failure of the installation rather than of a request, so it is no `ExarError`.
    """


class AgentError(Exception):
    """An agent failed or was stopped before it produced a result."""


class AgentFailed(AgentError):
    """An agent raised while `exar.agents.traced` ran it, chained from the error it raised.

    Args:
        trace: What the invocation recorded until it failed, without an output, whose spans carry the error.
    """

    def __init__(self, trace: Trace[Any, None]) -> None:
        super().__init__(f"{trace.root.name} failed")
        self.trace = trace
