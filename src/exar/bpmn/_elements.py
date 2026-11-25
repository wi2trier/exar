"""The supported BPMN elements, the agents that tasks call, and the index of a process graph."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta

from cbrkit.model.graph import (
    BpmnEdge,
    BpmnGraph,
    BpmnNode,
    bpmn,
    bpmn_class,
    bpmn_document,
    bpmn_event_class,
    from_bpmn,
)
from pydantic import TypeAdapter

__all__ = [
    "EVENTS",
    "GATEWAYS",
    "IGNORED",
    "TASKS",
    "agent_of",
    "called_agents",
    "substitute_agents",
]

type Kind = type[bpmn.TBaseElement]

TASKS: tuple[Kind, ...] = (bpmn.TTask, bpmn.TCallActivity)
"""Activities that call the agent `agent_of` names, unless they send or receive messages."""
MESSAGES: tuple[Kind, ...] = (bpmn.TSendTask, bpmn.TReceiveTask)
GATEWAYS: tuple[Kind, ...] = (
    bpmn.TExclusiveGateway,
    bpmn.TInclusiveGateway,
    bpmn.TParallelGateway,
    bpmn.TEventBasedGateway,
)
EVENTS: Mapping[Kind, tuple[Kind | None, ...]] = {
    bpmn.TStartEvent: (None,),
    bpmn.TEndEvent: (
        None,
        bpmn.TMessageEventDefinition,
        bpmn.TErrorEventDefinition,
        bpmn.TTerminateEventDefinition,
    ),
    bpmn.TIntermediateThrowEvent: (None, bpmn.TMessageEventDefinition),
    bpmn.TIntermediateCatchEvent: (
        bpmn.TMessageEventDefinition,
        bpmn.TTimerEventDefinition,
    ),
    bpmn.TBoundaryEvent: (bpmn.TErrorEventDefinition, bpmn.TTimerEventDefinition),
}
"""The supported event definitions of every event class, None for plain events."""
IGNORED: tuple[Kind, ...] = (
    bpmn.TDataObjectReference,
    bpmn.TDataStoreReference,
    bpmn.TParticipant,
)


def kind_of(value: Mapping[str, str | None], /) -> Kind:
    """The generated class of the element a node or edge value records."""
    return bpmn_class(value["type"])


def defines(node: BpmnNode, definition: Kind, /) -> bool:
    """Whether an event has an event definition of a class."""
    event = bpmn_event_class(node.value)

    return event is not None and issubclass(event, definition)


def calls_agent(node: BpmnNode, /) -> bool:
    """Whether a node is a task that calls an agent."""
    kind = kind_of(node.value)

    return issubclass(kind, TASKS) and not issubclass(kind, MESSAGES)


def agent_of(node: BpmnNode, /) -> str | None:
    """The agent a task calls, its `agent` attribute or else its name."""
    return node.value.get("agent") or node.value.get("name")


def called_agents(graph: BpmnGraph, /) -> frozenset[str]:
    """Names of the agents the tasks of a process call, see `agent_of`.

    Examples:
        >>> from cbrkit.model.graph import from_bpmn, parse_bpmn
        >>> graph = from_bpmn(parse_bpmn('''
        ...     <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
        ...       <process id="p"><task id="t" name="Write" agent="writer"/><task id="r" name="reviewer"/></process>
        ...     </definitions>
        ... '''))
        >>> sorted(called_agents(graph)), sorted(called_agents(substitute_agents(graph, {"writer": "author"})))
        (['reviewer', 'writer'], ['author', 'reviewer'])
    """
    return frozenset(
        agent
        for node in graph.nodes.values()
        if issubclass(kind_of(node.value), TASKS)
        and (agent := agent_of(node)) is not None
    )


def substitute_agents(graph: BpmnGraph, agents: Mapping[str, str], /) -> BpmnGraph:
    """The process with its tasks calling other agents, by the names of the agents they replace.

    The tasks get the new agent as their `agent` attribute, see `agent_of`,
    and everything else of the BPMN document of the process stays as it is.

    Raises:
        ValueError: If the process was not read from BPMN, see `cbrkit.model.graph.bpmn_document`.
    """
    renamed = {
        key: agents[agent]
        for key, node in graph.nodes.items()
        if issubclass(kind_of(node.value), TASKS)
        and (agent := agent_of(node)) is not None
        and agent in agents
    }

    def substitute[T](element: T) -> T:
        if isinstance(element, bpmn.TProcess | bpmn.TSubProcess):
            element = replace(
                element, flow_element=tuple(map(substitute, element.flow_element))
            )

        if isinstance(element, bpmn.TFlowNode) and element.id in renamed:
            return replace(
                element,
                other_attributes={
                    **element.other_attributes,
                    "agent": renamed[element.id],
                },
            )

        return element

    document = bpmn_document(graph)

    return from_bpmn(replace(document, choice=tuple(map(substitute, document.choice))))


_DURATION = TypeAdapter(timedelta)


def seconds_of(node: BpmnNode) -> float:
    """The ISO 8601 `timeDuration` of a timer event in seconds."""
    return _DURATION.validate_python(node.value.get("timeDuration")).total_seconds()


@dataclass(slots=True, frozen=True)
class Index:
    """The sequence flows, message flows, boundary events, and start events by node."""

    outgoing: Mapping[str, list[BpmnEdge]]
    incoming: Mapping[str, list[BpmnEdge]]
    messages: Mapping[str, list[BpmnEdge]]
    boundaries: Mapping[str, list[BpmnNode]]
    starts: Mapping[str | None, list[BpmnNode]]

    @classmethod
    def build(cls, graph: BpmnGraph) -> Index:
        outgoing: dict[str, list[BpmnEdge]] = {key: [] for key in graph.nodes}
        incoming: dict[str, list[BpmnEdge]] = {key: [] for key in graph.nodes}
        messages: dict[str, list[BpmnEdge]] = {key: [] for key in graph.nodes}
        boundaries: dict[str, list[BpmnNode]] = {key: [] for key in graph.nodes}
        starts: dict[str | None, list[BpmnNode]] = {}

        for edge in graph.edges.values():
            kind = kind_of(edge.value)

            if issubclass(kind, bpmn.TSequenceFlow):
                outgoing[edge.source.key].append(edge)
                incoming[edge.target.key].append(edge)
            elif issubclass(kind, bpmn.TMessageFlow):
                messages[edge.source.key].append(edge)
                messages[edge.target.key].append(edge)

        for node in graph.nodes.values():
            kind = kind_of(node.value)

            if issubclass(kind, bpmn.TBoundaryEvent):
                attached = node.value.get("attachedToRef") or ""
                boundaries.setdefault(attached, []).append(node)
            elif issubclass(kind, bpmn.TStartEvent):
                starts.setdefault(node.value.get("parent"), []).append(node)

        return cls(outgoing, incoming, messages, boundaries, starts)
