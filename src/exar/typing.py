"""Protocols of the components of an agentic application and the records they exchange, every protocol replaceable by a custom implementation.

The module depends on the standard library only, so the contracts bind no implementation.
The records are frozen dataclasses, which pydantic validates and serializes at the boundaries.

Records:

- `Span`, `Trace`: One run as a tree of OpenTelemetry spans, following their stable data model,
  whose GenAI payload `exar.otel` reads, such as `exar.otel.usage`.
- `Action`, `Effect`, `Decision`: What an agent is about to do, what it changes, and whether it may.
- `Verdict`, `Evaluation`: The assessment of a run and the results of its evaluators.
- `Hit`, `Changes`, `Add`, `Update`, `Remove`: What a search of a memory finds and the writes to it with their reasons.
- `Question`: A question about a text, which a `Decide` answers with a typed decision and a `Judge` with the probability of yes.

Protocols:

- `Agent`: Language model agents, deterministic functions, workflows, and BPMN processes alike.
- `Decide`, `Judge`: The agents that answer a `Question` with a typed decision or the probability of yes, such as decision models.
- `Policy`: Allows, refuses, or asks for approval of tool calls and agent invocations.
- `Assessed`: A run with its verdict, such as `exar.eval.Assessment` from pydantic-evals evaluators.
- `Memory`: Searches, reads, and writes knowledge, short-term and long-term alike.
- `Experience`: Retrieves, retains, and forgets assessed runs, the case base of the CBR cycle of `exar.experience.CbrCycle`.
- `ListableMemory`: A memory that also lists its keys, such as the index of a mount of `exar.memory.FileMemory`.
- `Approve`: Asks for approval of an action that a policy asks about.
- `Tool`: A tool as a plain typed async function.
- `Redact`: Rewrites a span before it joins a trace.
- `Measure`: Similarity of a text to a query, which memories without an index score their entries by.
- `Solution`, `Adaptation`, `Hints`: What runs reuse, how it is adapted, and how it is rendered, see `exar.experience`.
- `Channel`: The messages of processes, see `exar.bpmn`.

The protocols name no library, so any backend implements them,
such as cbrkit (`exar.cbrkit`) or a case-based reasoning service behind a REST API.
"""

from collections.abc import (
    Awaitable,
    Callable,
    Collection,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Generic,
    Literal,
    Protocol,
    Self,
    SupportsFloat,
    TypeVar,
    runtime_checkable,
)

__all__ = [
    "Action",
    "Adaptation",
    "Add",
    "Agent",
    "Approve",
    "Assessed",
    "Changes",
    "Channel",
    "Decide",
    "Decision",
    "Effect",
    "Evaluation",
    "Experience",
    "Hints",
    "Hit",
    "JsonEntry",
    "Judge",
    "ListableMemory",
    "Measure",
    "Memory",
    "Policy",
    "Question",
    "Redact",
    "Remove",
    "Solution",
    "Span",
    "Tool",
    "Trace",
    "Update",
    "Verdict",
]

# Declared variance, since type checkers infer frozen records as invariant because of `__replace__`.
I_co = TypeVar("I_co", covariant=True)
O_co = TypeVar("O_co", covariant=True)

type JsonEntry = (
    Mapping[str, JsonEntry] | Sequence[JsonEntry] | str | int | float | bool | None
)
"""A JSON value: an object, an array, a string, a number, a boolean, or null."""

type Effect = Literal["read", "write", "execute"]
"""What an action changes: nothing, a memory, or the environment outside the application."""

type Decision = Literal["allow", "ask", "deny"]
"""Whether an action runs, needs approval first, or is refused."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Span:
    """A finished OpenTelemetry span with the spans below it.

    Args:
        name: Name of the span.
        scope: Name of the instrumentation scope that created it, such as `exar` or `pydantic-ai`.
        span_id: Id of the span as hex.
        parent_id: Id of the parent span as hex, None for a root.
        start: When it started.
        end: When it ended.
        status: Its status code.
        error: The description of an error status.
        attributes: Its attributes.
        children: The spans below it in order of their start.
    """

    name: str
    scope: str = ""
    span_id: str
    parent_id: str | None = None
    start: datetime
    end: datetime
    status: Literal["unset", "ok", "error"] = "unset"
    error: str | None = None
    attributes: Mapping[str, JsonEntry] = field(default_factory=dict[str, JsonEntry])
    children: tuple[Span, ...] = ()

    def walk(self) -> Iterator[Span]:
        """The span and every span below it, depth first in order."""
        yield self

        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True, slots=True, kw_only=True)
class Trace(Generic[I_co, O_co]):  # noqa: UP046
    """The record of one run of an agent on a task.

    Args:
        task: The task.
        output: The output of the agent.
        root: The span of the invocation of the agent with every span below it.
        id: Id of the OpenTelemetry trace as hex.
    """

    task: I_co
    output: O_co
    root: Span
    id: str

    def walk(self) -> Iterator[Span]:
        """Every span of the run, depth first in order."""
        return self.root.walk()


@dataclass(frozen=True, slots=True, kw_only=True)
class Action:
    """Something an agent is about to do, which a policy allows, refuses, or sends for approval.

    Args:
        kind: Calling a tool or invoking an agent.
        name: Name of the tool or the agent.
        effect: What the action changes, `"execute"` where it is unknown, see `exar.tools.effect_of`.
        target: Key or path the action writes to, if any.
        depth: Nesting depth of the agent invocation the action happens in, 0 for the root agent.
        details: What an approver needs to judge the action as JSON,
            such as the arguments of a tool call or the task of an agent.
        metadata: What the tool declares about itself, such as its effect, for policies that decide by more.
    """

    kind: Literal["tool", "agent"]
    name: str
    effect: Effect
    target: str | None = None
    depth: int = 0
    details: JsonEntry = None
    metadata: Mapping[str, JsonEntry] = field(default_factory=dict[str, JsonEntry])


@dataclass(frozen=True, slots=True, kw_only=True)
class Verdict:
    """The summary of an assessment of a run.

    Args:
        score: Quality from 0 for a failure to 1 for a perfect result.
        passed: Whether the run is good enough to be kept as experience.
        reason: Why, for people and for language models that learn from it.
    """

    score: float
    passed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class Evaluation:
    """The result of one evaluator of a run, such as of a pydantic-evals evaluator.

    Args:
        name: Name of the evaluator or of its result.
        value: A score, an assertion, or a label.
        reason: Why, if the evaluator says.
    """

    name: str
    value: bool | int | float | str
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Hit[K, V]:
    """An entry that a search of a memory found.

    Args:
        key: Key of the entry.
        value: The entry.
        score: Relevance of the entry to the search, higher is better, comparable within one search only.
    """

    key: K
    value: V
    score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Add[K, V]:
    """Adds an entry.

    Args:
        key: Key of the entry, allocated by the memory if None, which memories keyed by name such as files need.
        value: The new entry.
        reason: Why the entry is added.
    """

    op: Literal["add"] = "add"
    key: K | None = None
    value: V
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Update[K, V]:
    """Replaces an entry.

    Args:
        key: Key of the entry.
        value: The new value of the entry.
        reason: Why the entry changes.
    """

    op: Literal["update"] = "update"
    key: K
    value: V
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Remove[K]:
    """Removes an entry.

    Args:
        key: Key of the entry.
        reason: Why the entry is removed.
    """

    op: Literal["remove"] = "remove"
    key: K
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Changes[K, V]:
    """Writes to a memory, each with its reason, applied as one mutation.

    Args:
        items: Additions, updates, and removals.
        applied: Whether the changes were made, with the keys of the added entries.
    """

    items: Sequence[Add[K, V] | Update[K, V] | Remove[K]] = ()
    applied: bool = False


class Agent[I, O](Protocol):
    """Handles a task of type `I` with an output of type `O`, by a language model, a function, a workflow, or a process.

    Agents of frameworks are configured natively and wrapped thinly, such as by `exar.pydantic_ai.PydanticAiAgent`.

    `exar.agents.run` runs an agent on a task, and an agent runs other agents with `exar.agents.call`,
    so every nested invocation is recorded into the trace of the run.
    An agent returns its output only, what it consumed is recorded by the GenAI spans of its model calls,
    see `exar.otel`, so an agent that calls models without an instrumented framework should emit them itself.
    """

    async def __call__(self, task: I, /) -> O: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Question:
    """A question about a text.

    Args:
        instructions: The question, such as "Does the second passage refer back to the first one?".
        text: What the question is about.
    """

    instructions: str
    text: str


type Decide[T] = Agent[Question, T]
"""Answers a question with a decision of type `T`, such as a label to pick, a score on a rubric, or labels that apply.

A decision model fills such decisions fast and calibrated and a language model as structured output,
such as through `exar.pydantic_ai.decide`, and any agent framework or function of the same signature is one too.
"""

type Judge = Decide[float]
"""Answers a yes/no question with the probability of yes, from 0 to 1, such as through `exar.pydantic_ai.judge`."""


class Policy(Protocol):
    """Decides whether an action runs, needs approval first, or is refused."""

    def __call__(self, action: Action, /) -> Decision: ...


class Assessed[I, O](Protocol):
    """A run with its verdict, which experience learns from."""

    @property
    def trace(self) -> Trace[I, O]:
        """The run."""
        ...

    @property
    def verdict(self) -> Verdict:
        """Its assessment."""
        ...

    @property
    def evaluations(self) -> Sequence[Evaluation]:
        """The results of the evaluators the verdict summarizes."""
        ...


class Memory[K, V](Protocol):
    """Knowledge that agents search, read, and write, such as files, a case base, or a vector database.

    Short-term and long-term memory are two instances, writes carry their reasons as `Changes`.
    """

    async def search(self, *queries: str, limit: int = 10) -> Sequence[Hit[K, V]]:
        """Finds the entries most relevant to the queries, the most relevant first.

        The rankings of several queries are fused by reciprocal rank fusion before the limit applies,
        so an entry that several of them find ranks higher.
        """
        ...

    async def read(self, keys: Collection[K], /) -> Mapping[K, V]:
        """Reads the entries of the keys, leaving out missing ones."""
        ...

    async def write(
        self, changes: Changes[K, V], /, *, expected: Mapping[K, V] | None = None
    ) -> Changes[K, V]:
        """Applies changes as one mutation and returns them applied, with the keys of added entries.

        Args:
            changes: The changes.
            expected: The entries that the changes were planned on, which the memory must still hold,
                None if the changes depend on no entry.

        Raises:
            exar.errors.Conflict: If the memory holds other entries than the expected ones.
            exar.errors.Refused: If a policy refuses a change.
        """
        ...


@runtime_checkable
class ListableMemory[K, V](Memory[K, V], Protocol):
    """A memory that also lists its keys, so whoever keeps it in sync can find stale entries."""

    async def keys(self, /) -> Collection[K]:
        """Keys of all entries."""
        ...


class Experience[K, I, O](Protocol):
    """Assessed runs that agents retrieve for similar tasks, retain, and forget, such as `exar.experience.StorageExperience`.

    It is the case base of case-based reasoning over runs (Meta-CBR), see `exar.experience.CbrCycle`.
    A run that failed with an error has no output, so the outputs of retained runs are None for them.
    """

    async def retrieve(
        self, task: I, /, *, limit: int = 3, passed: bool = False
    ) -> Sequence[Hit[K, Assessed[I, O | None]]]:
        """The assessed runs of the tasks most similar to a task, the most similar first, only those that passed if `passed`."""
        ...

    async def retain(
        self, assessments: Sequence[Assessed[I, O | None]], /
    ) -> Changes[K, Assessed[I, O | None]]:
        """Stores assessed runs and returns the applied changes of the cases that store them, new or merged into."""
        ...

    async def forget(
        self, select: Callable[[Assessed[I, O | None]], bool], /, *, reason: str
    ) -> Changes[K, Assessed[I, O | None]]:
        """Removes the runs a predicate selects, such as the runs of a trace id or of a user, and returns the applied removals."""
        ...


type Approve = Callable[[Action], Awaitable[bool]]
"""Asks for approval of an action, such as a person reviewing a write, and returns whether it was given."""

type Tool = Callable[..., Awaitable[object]]
"""A tool: a plain typed async function, named by `exar.helpers.get_name`, described by its docstring,
whose signature is the input schema and whose `effect` tells what it changes,
or a `exar.tools.SchemaTool`, whose input schema is data."""


type Redact = Callable[[Span], Span]
"""Rewrites a span record before it joins a trace, such as to drop or mask message content."""

type Measure = Callable[[str, str], SupportsFloat]
"""Similarity of a text to a query from 0 to 1, such as `exar.memory.word_coverage` or a pair measure of cbrkit,
whose detailed similarities convert with `float`."""


class Solution(Protocol):
    """A reusable solution that refers to agents and tools by name, which adaptations substitute."""

    @property
    def references(self) -> Collection[str]:
        """Names of the agents and tools it refers to."""
        ...

    def rename(self, names: Mapping[str, str], /) -> Self:
        """The solution with the references renamed by their old names."""
        ...


class Adaptation[I, S: Solution](Protocol):
    """Adapts the solution of a past run to a new task of type `I`, such as `exar.experience.Substitute` or a cbrkit adaptation in an `exar.cbrkit.Adapt`.

    It returns a solution of the type `S` it was given, so an adaptation by a remote service rebuilds it, such as by `type(solution)`.
    """

    async def __call__(self, solution: S, past: I, task: I, /) -> S:
        """The solution of the run of the past task adapted to the new task."""
        ...


class Hints[I, O](Protocol):
    """Renders retrieved runs for a new task as text for a language model, empty without anything to say.

    The full traces never go into the prompt, only what a hint renders from them,
    such as the outputs as few-shot examples (`exar.experience.Examples`) or the trajectories (`exar.experience.Trajectories`).
    """

    async def __call__(self, runs: Sequence[Assessed[I, O | None]], task: I, /) -> str:
        """The hints of the runs for the task."""
        ...


class Channel[S](Protocol):
    """Exchanges the messages of a process with the participants outside it, such as a queue of the host."""

    async def send(self, to: str | None, content: S, /) -> None:
        """Sends content to a participant, None for whoever runs the process."""
        ...

    async def receive(self, /, *, timeout: float | None = None) -> tuple[str, object]:
        """Waits for the next message and returns its sender and content, which updates the state.

        Raises:
            TimeoutError: If no message arrives in time.
        """
        ...
