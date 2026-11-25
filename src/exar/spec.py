"""An application described as plain data in a JSON file and built into objects by name.

A `Spec` covers the parts exar owns: a policy, memories, tools, limits, BPMN processes by file with their outputs and error field,
and workflows over agents, processes, and other workflows, which they name.
A `case_base` tool (`exar.tools.case_base.CaseBase`) references the sidecar of a cbrkit case base, the specification of cbrkit,
and both files follow the same format: JSON with a `version`, an optional `$schema`, and classes named by their `kind`.
Agents are configured natively in their frameworks and passed to `build` by name, never named by the file.
`Spec.model_json_schema()` is the JSON Schema of the file, and `load` reads it.
The tools are a list of `ToolKind`, and importing this module needs no extra but `pydantic-ai`.
A tool that needs another extra, `cbrkit`, `http`, `sql`, or `web`, raises `exar.errors.MissingExtra` naming it when it is built,
as do processes without the `bpmn` extra, so a spec only needs the extras of what it lists.
The engines, clients, and sessions that tools name are the host's and `load` binds them by name.
Plain Python construction of the objects stays the primary way to compose an application, see the README.

Examples:
    >>> from exar.agents import FunctionAgent, run
    >>> spec = Spec.model_validate(
    ...     {
    ...         "policy": {"kind": "effect", "execute": "allow"},
    ...         "workflows": {"pipeline": {"kind": "chain", "agents": ["add", "double"]}},
    ...         "limits": {"max_depth": 2},
    ...     }
    ... )
    >>> app = build(spec, {"add": FunctionAgent(func=lambda x: x + 1), "double": FunctionAgent(func=lambda x: 2 * x)})
    >>> run(app.workflows["pipeline"], 1, app.config).output
    4
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field

from ._lazy import extra_hint
from .agents import Limits, RunConfig
from .memory import MemoryKind
from .policy import EffectPolicy, PolicyKind
from .schema import Model
from .tools.case_base import CaseBase
from .tools.files import Grep, Jq, Ls, Read
from .tools.http import OpenApi
from .tools.media import View
from .tools.sql import Engine, HostEngine, Sql
from .tools.table import Table
from .tools.web import Client, HostClient, HostSession, WebFetch, WebSearch, WebSession
from .typing import Agent, Memory, Tool
from .workflows import Chain, EvaluateOptimize, Parallel, Rag, Route

__all__ = [
    "App",
    "ChainSpec",
    "EvaluateOptimizeSpec",
    "ParallelSpec",
    "ProcessSpec",
    "RagSpec",
    "RouteSpec",
    "Spec",
    "ToolKind",
    "WorkflowSpec",
    "build",
    "load",
]

type ToolKind = Annotated[
    Read
    | Ls
    | Grep
    | Jq
    | View
    | Table
    | WebFetch
    | WebSearch
    | Sql
    | OpenApi
    | CaseBase,
    Field(discriminator="kind"),
]
"""The configurable tools, whose host-bound names `load` binds."""


class _AgentsSpec(Model):
    """A workflow over a list of agents."""

    agents: Sequence[str]

    def refs(self) -> Sequence[str]:
        """The names it refers to."""
        return self.agents


class ChainSpec(_AgentsSpec):
    """A `exar.workflows.Chain`.

    Args:
        agents: Names of the agents in order.
    """

    kind: Literal["chain"] = "chain"

    def build(self, named: Mapping[str, Agent[Any, Any]]) -> Agent[Any, Any]:
        """The workflow over the named agents."""
        return Chain(agents=[named[name] for name in self.agents])


class ParallelSpec(_AgentsSpec):
    """A `exar.workflows.Parallel`.

    Args:
        agents: Names of the agents to run.
    """

    kind: Literal["parallel"] = "parallel"

    def build(self, named: Mapping[str, Agent[Any, Any]]) -> Agent[Any, Any]:
        """The workflow over the named agents."""
        return Parallel(agents=[named[name] for name in self.agents])


class RouteSpec(Model):
    """A `exar.workflows.Route`.

    Args:
        classifier: Name of the agent that returns a label.
        routes: Names of the handling agents by label.
    """

    kind: Literal["route"] = "route"
    classifier: str
    routes: Mapping[str, str]

    def refs(self) -> Sequence[str]:
        """The names it refers to."""
        return [self.classifier, *self.routes.values()]

    def build(self, named: Mapping[str, Agent[Any, Any]]) -> Agent[Any, Any]:
        """The workflow over the named agents."""
        return Route(
            classifier=named[self.classifier],
            routes={label: named[name] for label, name in self.routes.items()},
        )


class EvaluateOptimizeSpec(Model):
    """A `exar.workflows.EvaluateOptimize`.

    Args:
        generator: Name of the generating agent.
        evaluator: Name of the evaluating agent.
        max_rounds: Most generations.
    """

    kind: Literal["evaluate_optimize"] = "evaluate_optimize"
    generator: str
    evaluator: str
    max_rounds: int = 3

    def refs(self) -> Sequence[str]:
        """The names it refers to."""
        return [self.generator, self.evaluator]

    def build(self, named: Mapping[str, Agent[Any, Any]]) -> Agent[Any, Any]:
        """The workflow over the named agents."""
        return EvaluateOptimize(
            generator=named[self.generator],
            evaluator=named[self.evaluator],
            **self.model_dump(include={"max_rounds"}, exclude_unset=True),
        )


class RagSpec(Model):
    """A `exar.workflows.Rag`.

    Args:
        retrieve: Name of the agent that retrieves context.
        generate: Name of the agent that answers the grounded query.
    """

    kind: Literal["rag"] = "rag"
    retrieve: str
    generate: str

    def refs(self) -> Sequence[str]:
        """The names it refers to."""
        return [self.retrieve, self.generate]

    def build(self, named: Mapping[str, Agent[Any, Any]]) -> Agent[Any, Any]:
        """The workflow over the named agents."""
        return Rag(retrieve=named[self.retrieve], generate=named[self.generate])


class ProcessSpec(Model):
    """A BPMN process of a spec, see `exar.bpmn.BpmnProcess`, whose defaults apply to the fields left out.

    Args:
        file: The BPMN file, a relative path resolves against the working directory.
        error_field: The field of the state that error boundary events store errors in.
        outputs: The fields of the state that plain results update, by the id of the task or catch event.
    """

    file: Path
    error_field: str | None = None
    outputs: Mapping[str, str] = {}


type WorkflowSpec = Annotated[
    ChainSpec | ParallelSpec | RouteSpec | EvaluateOptimizeSpec | RagSpec,
    Field(discriminator="kind"),
]
"""The workflows of a spec, which `exar.experience.CbrCycle` and `ProcessReuse` are not part of since they hold an experience."""


class Spec(Model):
    """An application described as data.

    Args:
        version: Version of the spec format, which exar reads it by.
        schema_ref: URI of the JSON Schema that editors validate the spec with,
            stored as `$schema` and kept as given.
        policy: Checks the agent invocations of a run.
        memories: Memories by name, which the host hands to its agents, such as by `exar.memory.memory_tools`.
        tools: The tools, which `build` turns into functions.
        limits: Bounds of a run.
        processes: BPMN processes by name, whose tasks invoke the agents of the host by name.
        workflows: Workflows by name, which refer to agents, processes, and other workflows by name.
    """

    model_config = ConfigDict(serialize_by_alias=True)

    version: Literal[1] = 1
    schema_ref: str | None = Field(
        default=None, alias="$schema", exclude_if=lambda value: value is None
    )
    policy: PolicyKind = EffectPolicy()
    memories: Mapping[str, MemoryKind] = {}
    tools: Sequence[ToolKind] = ()
    limits: Limits = Limits()
    processes: Mapping[str, ProcessSpec] = {}
    workflows: Mapping[str, WorkflowSpec] = {}


@dataclass(frozen=True, slots=True, kw_only=True)
class App:
    """The objects of a built `Spec`.

    Args:
        config: The policy and limits of a run, which the host also passes to the surfaces it serves,
            and completes with its approver, such as by `dataclasses.replace`.
        memories: Memories by name.
        tools: The functions of the tools (their `tools()`), in order.
        processes: BPMN processes by name, `exar.bpmn.BpmnProcess` agents.
        workflows: Workflows by name.
    """

    config: RunConfig
    memories: Mapping[str, Memory[Any, Any]]
    tools: tuple[Tool, ...]
    processes: Mapping[str, Agent[Any, Any]]
    workflows: Mapping[str, Agent[Any, Any]]


def load(
    path: Path,
    /,
    *,
    engines: Mapping[str, Callable[[], Engine]] = {},
    clients: Mapping[str, Callable[[], Client]] = {},
    sessions: Mapping[str, Callable[[], WebSession]] = {},
) -> Spec:
    """Reads a spec from a JSON file.

    The tools name what the host owns, which this binds to the functions of the host by name,
    a name without a function fails when its tool is used with `exar.errors.Unbound`.

    Args:
        path: The file.
        engines: Database engines for the `engine` of `Sql` tools.
        clients: HTTP clients for the `client` of `OpenApi`, `WebFetch`, and `WebSearch` tools.
        sessions: Web sessions for the `session` of `WebFetch` and `WebSearch` tools.
    """
    context = {HostEngine: engines, HostClient: clients, HostSession: sessions}

    return Spec.model_validate_json(path.read_bytes(), context=context)


def _processes(
    specs: Mapping[str, ProcessSpec], agents: Mapping[str, Agent[Any, Any]]
) -> dict[str, Agent[Any, Any]]:
    """The BPMN processes of files over the agents, which need the `bpmn` extra."""
    with extra_hint("The processes of a spec", "bpmn"):
        from cbrkit.model.graph import from_bpmn, parse_bpmn

        from .bpmn import BpmnProcess

    return {
        name: BpmnProcess(
            graph=from_bpmn(parse_bpmn(spec.file.read_bytes())),
            agents=agents,
            **spec.model_dump(exclude={"file"}, exclude_unset=True),
        )
        for name, spec in specs.items()
    }


def build(spec: Spec, agents: Mapping[str, Agent[Any, Any]], /) -> App:
    """Builds the objects of a spec over the agents of the host, in two phases.

    Processes are built over the agents first,
    then workflows in dependency order over the agents, the processes, and the workflows built before them.

    Raises:
        ValueError: If a name is used more than once, a workflow refers to an unknown name or forms a cycle,
            or a process has problems, see `exar.bpmn.validate`.
        exar.errors.MissingExtra: If the spec lists processes without the `bpmn` extra.
    """
    names = [*agents, *spec.processes, *spec.workflows]

    if duplicates := sorted({name for name in names if names.count(name) > 1}):
        raise ValueError(f"Names used more than once: {', '.join(duplicates)}")

    processes = _processes(spec.processes, agents) if spec.processes else {}
    named: dict[str, Agent[Any, Any]] = {**agents, **processes}
    refs = {name: workflow.refs() for name, workflow in spec.workflows.items()}

    if unknown := sorted(
        {ref for names in refs.values() for ref in names} - named.keys() - refs.keys()
    ):
        raise ValueError(f"Workflows refer to unknown names: {', '.join(unknown)}")

    try:
        order = tuple(TopologicalSorter(refs).static_order())
    except CycleError as error:
        raise ValueError(
            f"Workflows form a cycle: {' -> '.join(error.args[1])}"
        ) from error

    for name in order:
        if name in spec.workflows:
            named[name] = spec.workflows[name].build(named)

    return App(
        config=RunConfig(policy=spec.policy, limits=spec.limits),
        memories=spec.memories,
        tools=tuple(function for tool in spec.tools for function in tool.tools()),
        processes=processes,
        workflows={name: named[name] for name in spec.workflows},
    )
