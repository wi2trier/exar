"""Workflow recipes as agents that hold other agents, so they nest and mix with agents of any framework.

The names follow the patterns of Anthropic's "Building effective agents":
prompt chaining (`Chain`), routing (`Route`), parallelization (`Parallel`),
evaluator-optimizer (`EvaluateOptimize`), and the augmented language model (`Rag`).
The case-based reasoning cycle over runs, `exar.experience.CbrCycle`, `exar.experience.ProcessReuse`,
and `exar.bpmn.BpmnProcess` are workflows as well.
The orchestrator-workers pattern is a language model agent that delegates to subagents,
configured natively in its framework, such as with the `SubAgents` of pydantic-ai-harness.
Every workflow derives from `Workflow`, invokes its agents with `exar.agents.call`, so each invocation is checked and traced,
and is traced as an `invoke_workflow` span itself.
The effect of a workflow is what it changes itself, while each agent it invokes is checked on its own:
`"read"` for a workflow that only invokes agents, `"write"` for one that retains experience, such as `CbrCycle`.

Examples:
    >>> from exar.agents import FunctionAgent, RunConfig, run
    >>> from exar.policy import allow_all
    >>> add, double = FunctionAgent(func=lambda x: x + 1), FunctionAgent(func=lambda x: 2 * x)
    >>> run(Chain(agents=[add, double]), 1, RunConfig(policy=allow_all)).output
    4
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypedDict

from .agents import call
from .otel import Operation
from .typing import Agent, Effect, Verdict

__all__ = [
    "Chain",
    "EvaluateOptimize",
    "Parallel",
    "Rag",
    "Route",
    "Workflow",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Workflow[I, O](ABC):
    """An agent that invokes other agents, traced as an `invoke_workflow` span.

    Its effect is `"read"` unless it changes something itself, see the module.
    """

    operation: ClassVar[Operation] = "invoke_workflow"
    effect: ClassVar[Effect] = "read"

    @abstractmethod
    async def __call__(self, task: I, /) -> O:
        """Solves the task by invoking agents with `exar.agents.call`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Chain[I, O](Workflow[I, O]):
    """Prompt chaining: runs agents in sequence, each on the output of the one before.

    The types of the agents in between are not checked, only the task of the first and the output of the last.

    Args:
        agents: The agents in order.
    """

    agents: Sequence[Agent[Any, Any]]

    async def __call__(self, task: I, /) -> O:
        output: Any = task

        for agent in self.agents:
            output = await call(agent, output)

        return output


@dataclass(frozen=True, slots=True, kw_only=True)
class Route[I, O](Workflow[I, O]):
    """Routing: a classifier agent picks the label of the agent that handles the task.

    Args:
        classifier: The agent that returns a label for the task.
        routes: The handling agents by label.
    """

    classifier: Agent[I, str]
    routes: Mapping[str, Agent[I, O]]

    async def __call__(self, task: I, /) -> O:
        label = await call(self.classifier, task)

        if (agent := self.routes.get(label)) is None:
            raise ValueError(f"No route for label {label!r}")

        return await call(agent, task)


@dataclass(frozen=True, slots=True, kw_only=True)
class Parallel[I, O](Workflow[I, list[O]]):
    """Parallelization: runs agents concurrently on the task, by sections or for votes.

    Chain it with an agent that merges the list of outputs.

    Args:
        agents: The agents to run.
    """

    agents: Sequence[Agent[I, O]]

    async def __call__(self, task: I, /) -> list[O]:
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(call(agent, task)) for agent in self.agents]

        # The first failure cancels the siblings and surfaces as itself.
        except ExceptionGroup as failures:
            raise failures.exceptions[0] from failures

        return [t.result() for t in tasks]


@dataclass(frozen=True, slots=True, kw_only=True)
class Attempt[I, O]:
    """An output for a task, with its verdict once evaluated.

    Args:
        task: The task.
        output: Output of the generator.
        verdict: Assessment of the output by the evaluator.
    """

    task: I
    output: O
    verdict: Verdict | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluateOptimize[I, O](Workflow[I, O]):
    """Evaluator-optimizer: a generator revises its output until an evaluator passes it.

    The generator gets the task first and the evaluated `Attempt` afterwards,
    the evaluator gets the `Attempt` without verdict and returns an `exar.typing.Verdict`.

    Args:
        generator: The generating agent.
        evaluator: The evaluating agent.
        max_rounds: Most generations, after which the best scored output is returned.
    """

    generator: Agent[I | Attempt[I, O], O]
    evaluator: Agent[Attempt[I, O], Verdict]
    max_rounds: int = 3

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {self.max_rounds}")

    async def __call__(self, task: I, /) -> O:
        attempts: list[Attempt[I, O]] = []
        draft: I | Attempt[I, O] = task

        for _ in range(self.max_rounds):
            output = await call(self.generator, draft)
            verdict = await call(self.evaluator, Attempt(task=task, output=output))
            draft = Attempt(task=task, output=output, verdict=verdict)
            attempts.append(draft)

            if verdict.passed:
                break

        best = max(
            attempts,
            key=lambda attempt: attempt.verdict.score if attempt.verdict else 0.0,
        )

        return best.output


class Grounded[Q, C](TypedDict):
    """The task of the generator of `Rag`: the query and the context retrieved for it."""

    query: Q
    context: C


@dataclass(frozen=True, slots=True, kw_only=True)
class Rag[Q, C, O](Workflow[Q, O]):
    """Retrieval-augmented generation: grounds a generator in what a retriever finds.

    Args:
        retrieve: The agent that retrieves context for the query.
        generate: The agent that answers the `Grounded` query.
    """

    retrieve: Agent[Q, C]
    generate: Agent[Grounded[Q, C], O]

    async def __call__(self, task: Q, /) -> O:
        context = await call(self.retrieve, task)

        return await call(self.generate, {"query": task, "context": context})
