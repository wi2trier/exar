"""The agent that runs a BPMN process as a pydantic-graph graph."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cbrkit.model.graph import BpmnGraph, bpmn_document, render_bpmn
from opentelemetry import trace
from pydantic_graph import Graph

from ..typing import Agent, Channel
from ..workflows import Workflow
from ._graph import Deps, State
from ._validate import checked

__all__ = ["PROCESS", "BpmnProcess"]

PROCESS = "exar.bpmn.process"
"""Span attribute of the process of a run as BPMN XML, on the span of the invocation of a `BpmnProcess`."""


@dataclass(slots=True, frozen=True, kw_only=True)
class BpmnProcess[S](Workflow[S, S]):
    """An agent that runs a BPMN process over a typed state, see the module for the supported elements.

    The task is the initial state, a frozen pydantic model or dataclass `S`, and the output the final state.
    Every task invokes the agent named by its `agent` attribute or else its name with the current state,
    through `exar.agents.call`, and its result updates the state by one rule:
    an `S` replaces the state, a mapping updates the fields it names,
    and any other value updates the field that `outputs` names for the task, which fails without one.
    Updates are validated against `S`, so unknown fields and invalid values raise.
    Parallel branches share the state, each update applies to the state at the time it finishes,
    so branches should update the fields they own with mappings rather than replace the state.

    Conditions of sequence flows are expressions of simpleeval over the state as it left the source of the flow,
    with the fields of the state as names, the default functions of simpleeval and `functions`,
    and attribute access to the fields of nested models and dataclasses and to safe attributes of builtins.
    Exclusive gateways take the first flow whose condition holds,
    parallel gateways and activities without conditional flows take all,
    other gateways and activities every flow whose condition holds, the default flow if none does.
    Joining gateways are pydantic-graph joins that wait for every token of the closest fork they follow,
    so a process has to be structured, with every join dominated by a fork that it is part of every cycle with.

    Send tasks and message throw events send the state over the channel to the node's `agent`,
    or else to the pool, agent, or name at the other end of its message flow,
    or else to whoever runs the process.
    Receive tasks and message catch events wait for a message of the channel and update the state with its content
    by the same rule.
    After an event-based gateway, the first message goes to the catch event whose peer is its sender,
    or else to the first one, unless a timer expires first.
    Error boundary events catch every exception of their activity and store its message in the field `error_field`,
    which a process with error boundary events has to name.
    Timer boundary events interrupt their activity after their `timeDuration`.
    Embedded subprocesses run as graphs of their own over the same state.
    A run ends once no token is left or at a terminate end event, which cancels all other tokens.
    Loops are bounded by the limits of the run, see `exar.agents.Limits`.

    The process is validated and compiled into a pydantic-graph graph once, when it is created.
    A run records the process as BPMN XML under `PROCESS` on the span of its invocation,
    so it has to be invoked through `exar.agents.call` or `exar.agents.run`,
    and pydantic-graph instruments the graph and its nodes for Logfire.

    Args:
        graph: The process, as constructed by `cbrkit.model.graph.from_bpmn`.
            All its processes run, so other participants should be pools without a process.
        agents: The agents the tasks invoke by name.
        functions: The functions of conditions besides the default ones of simpleeval.
        channel: Exchanges the messages of message events and send and receive tasks.
        error_field: The field of the state that error boundary events store the message of an error in.
        outputs: The fields of the state that plain results update, by the id of the task or catch event.

    Returns:
        The final state.

    Raises:
        ValueError: On creation if the graph was not read from BPMN or the process has problems, such as unsupported elements, unknown agents,
            unreachable elements, loops without an activity, a join that no fork dominates, error boundary events without
            an `error_field`, or `outputs` of unknown tasks,
            on a run if the state lacks the fields of `error_field` or `outputs`,
            no outgoing flow of an activity or inclusive gateway applies, a task without `outputs` returns a plain value,
            or an update does not fit the state.
        RuntimeError: If no outgoing flow of an exclusive gateway applies.
        TypeError: If the state is neither a pydantic model nor a dataclass.
        exar.errors.AgentError: If an error end event is reached outside of a boundary event.
    """

    graph: BpmnGraph
    agents: Mapping[str, Agent[S, Any]] = field(
        default_factory=dict[str, Agent[S, Any]]
    )
    functions: Mapping[str, Callable[..., object]] = field(
        default_factory=dict[str, Callable[..., object]]
    )
    channel: Channel[S] | None = None
    error_field: str | None = None
    outputs: Mapping[str, str] = field(default_factory=dict[str, str])
    _flow: Graph[State[S], Deps[S], None, None] = field(
        init=False, repr=False, compare=False
    )
    _bpmn: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_bpmn", render_bpmn(bpmn_document(self.graph)))
        problems, flow = checked(
            self.graph, self.agents, self.functions, self.outputs, self.error_field
        )

        if flow is None:
            raise ValueError("Invalid process:\n" + "\n".join(problems))

        object.__setattr__(self, "_flow", flow)

    def render(self) -> str:
        """The graph that runs the process as a Mermaid state diagram."""
        return self._flow.render()

    async def __call__(self, task: S, /) -> S:
        span = trace.get_current_span()

        if span.is_recording():
            span.set_attribute(PROCESS, self._bpmn)

        state = State.of(task)
        named = {*self.outputs.values()} | (
            set() if self.error_field is None else {self.error_field}
        )

        if unknown := sorted(named.difference(state.fields)):
            raise ValueError(f"The state has no fields {', '.join(unknown)}")

        await self._flow.run(
            state=state, deps=Deps[S](agents=self.agents, channel=self.channel)
        )

        return state.value
