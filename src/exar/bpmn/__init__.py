"""Rule-based orchestration by BPMN processes, run as pydantic-graph graphs, needs the `bpmn` extra.

A `BpmnProcess` is an agent that executes a BPMN graph read by `cbrkit.model.graph.parse_bpmn` and `cbrkit.model.graph.from_bpmn`,
which `validate` checks beforehand and `substitute_agents` adapts to other agents.
It compiles the process into a graph of pydantic-graph once, which runs its parallel branches concurrently.
Its tasks invoke agents by name, the `agent` attribute of a task or else its name (`agent_of`),
gateways route by conditions that simpleeval evaluates over the state,
and message events talk to other participants over a `Channel`.

The state is a frozen pydantic model or dataclass `S` shared by all tokens of a run,
like the process variables of BPMN engines, so a `BpmnProcess[S]` is an `exar.typing.Agent[S, S]`.
Tasks get the state as their task, and every result updates it by the rule of `BpmnProcess`,
validated against `S`, while parallel branches apply their updates as they finish.

Supported are start and end events (plain, message, error, terminate), all task types,
intermediate message and timer events, exclusive, inclusive, parallel, and event-based gateways,
interrupting error and timer boundary events, and embedded subprocesses,
in structured processes whose joins pydantic-graph can pair with a fork.
Every run records its process as BPMN XML on the span of its invocation under `PROCESS`,
so the trace of a run keeps the process that experience reuses, see `exar.experience.Process`.

Modules:
- `_conditions`: The conditions of sequence flows, evaluated by simpleeval.
- `_elements`: The supported elements, the agents that tasks call, and the index of a process graph.
- `_graph`: The compilation of a process into a pydantic-graph graph, its state, and its dependencies.
- `_run`: `BpmnProcess` and `PROCESS`.
- `_validate`: The check of a process before it runs, on top of `cbrkit.model.graph.bpmn_problems` and its compilation.
"""

from ._elements import agent_of, called_agents, substitute_agents
from ._run import PROCESS, BpmnProcess
from ._validate import validate

__all__ = [
    "PROCESS",
    "BpmnProcess",
    "agent_of",
    "called_agents",
    "substitute_agents",
    "validate",
]
