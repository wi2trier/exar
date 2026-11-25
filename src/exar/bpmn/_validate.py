"""Validation of BPMN processes, by static checks and by compiling them."""

from collections.abc import Callable, Collection, Mapping

from cbrkit.model.graph import BpmnGraph, bpmn, bpmn_problems
from pydantic_graph import Graph

from ._conditions import condition
from ._elements import (
    EVENTS,
    GATEWAYS,
    IGNORED,
    TASKS,
    agent_of,
    calls_agent,
    defines,
    kind_of,
    seconds_of,
)
from ._graph import Deps, State, compile_process

# The concrete `SubProcess` leaves out its subclasses, transactions and ad hoc subprocesses.
SUPPORTED = (*EVENTS, *TASKS, *GATEWAYS, *IGNORED, bpmn.SubProcess)
RESULTS = (*TASKS, bpmn.TIntermediateCatchEvent)
"""The nodes whose results update the state."""


def validate(
    graph: BpmnGraph,
    agents: Collection[str] | None,
    functions: Mapping[str, Callable[..., object]],
    /,
    *,
    outputs: Mapping[str, str] = {},
    error_field: str | None = None,
) -> list[str]:
    """Reports the problems that keep a process from running, see `cbrkit.model.graph.bpmn_problems`.

    Its elements, agents, timers, and conditions are checked first,
    and only a process without such problems is compiled to find structural ones,
    such as elements no start event reaches, loops without an activity, or joins that no fork dominates.

    Args:
        graph: The process.
        agents: Names of the available agents, None to skip this check.
        functions: The functions of the conditions besides the default ones of simpleeval.
        outputs: The fields that plain results update by node id, see `exar.bpmn.BpmnProcess`.
        error_field: The field that error boundary events store errors in, see `exar.bpmn.BpmnProcess`.

    Returns:
        A description of every problem, empty if there is none.

    Examples:
        >>> from cbrkit.model.graph import from_bpmn, parse_bpmn
        >>> graph = from_bpmn(parse_bpmn('''
        ...     <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
        ...       <process id="p">
        ...         <startEvent id="s"/>
        ...         <task id="t" name="writer"/>
        ...         <complexGateway id="g"/>
        ...         <sequenceFlow id="f" sourceRef="s" targetRef="t"><conditionExpression>x &gt;</conditionExpression></sequenceFlow>
        ...       </process>
        ...     </definitions>
        ... '''))
        >>> for problem in validate(graph, ["reviewer"], {}):
        ...     print(problem)
        Unsupported element complexGateway 'g'
        Task 't' calls the unknown agent 'writer'
        Condition of 'f': Invalid condition 'x >'
    """
    return checked(graph, agents, functions, outputs, error_field)[0]


def checked[S](
    graph: BpmnGraph,
    agents: Collection[str] | None,
    functions: Mapping[str, Callable[..., object]],
    outputs: Mapping[str, str],
    error_field: str | None,
    /,
) -> tuple[list[str], Graph[State[S], Deps[S], None, None] | None]:
    """The problems of a process, see `validate`, and its graph if there is none."""
    if problems := _problems(graph, agents, functions, outputs, error_field):
        return problems, None

    try:
        return [], compile_process(graph, functions, outputs, error_field)
    except ValueError as error:
        return [str(error)], None


def _problems(
    graph: BpmnGraph,
    agents: Collection[str] | None,
    functions: Mapping[str, Callable[..., object]],
    outputs: Mapping[str, str],
    error_field: str | None,
    /,
) -> list[str]:
    """The problems of the elements, agents, timers, conditions, and outputs of a process."""
    problems = bpmn_problems(graph, supported=SUPPORTED, events=EVENTS)
    problems.extend(
        f"Outputs name {key!r}, which is no task or catch event"
        for key in outputs
        if key not in graph.nodes
        or not issubclass(kind_of(graph.nodes[key].value), RESULTS)
    )

    for key, node in graph.nodes.items():
        kind = kind_of(node.value)

        if calls_agent(node):
            agent = agent_of(node)

            if agent is None:
                problems.append(f"Task {key!r} names no agent")
            elif agents is not None and agent not in agents:
                problems.append(f"Task {key!r} calls the unknown agent {agent!r}")

        if defines(node, bpmn.TTimerEventDefinition):
            try:
                seconds_of(node)
            except ValueError:
                problems.append(f"Timer {key!r} needs an ISO 8601 timeDuration")

        if not issubclass(kind, bpmn.TBoundaryEvent):
            continue

        if node.value.get("cancelActivity") == "false":
            problems.append(f"Non-interrupting boundary event {key!r}")
        elif error_field is None and defines(node, bpmn.TErrorEventDefinition):
            problems.append(f"Error boundary event {key!r} needs an error_field")

    for key, edge in graph.edges.items():
        if (text := edge.value.get("condition")) is not None:
            try:
                _ = condition(text, functions)
            except ValueError as error:
                problems.append(f"Condition of {key!r}: {error}")

    return problems
