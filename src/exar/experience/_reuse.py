"""Reusing retrieved runs: hints for language model agents and the CBR cycle over runs, see `ProcessReuse` for processes."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from ..agents import call, traced
from ..errors import AgentFailed
from ..text import render_example, tagged
from ..typing import (
    Adaptation,
    Agent,
    Assessed,
    Effect,
    Experience,
    Hints,
    Trace,
    Verdict,
)
from ..workflows import Workflow
from ._retained import Retained
from ._solutions import FileChanges, Trajectory

__all__ = ["CbrCycle", "Examples", "Files", "Trajectories", "recall"]


async def recall[I, O](
    experience: Experience[Any, I, O], hints: Hints[I, O], task: I, /, *, limit: int = 3
) -> str:
    """The hints of the runs of the tasks most similar to a task, empty without anything to say.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.sim.strings import WordCoverage
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.experience import StorageExperience
        >>> from exar.policy import allow_all
        >>> cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
        >>> trace = run(FunctionAgent(func=lambda task: "4"), "2 + 2", RunConfig(policy=allow_all))
        >>> _ = asyncio.run(cases.retain([Retained(trace=trace, verdict=Verdict(score=1.0, passed=True))]))
        >>> print(asyncio.run(recall(cases, Examples(), "2 + 3")).splitlines()[0])
        Past runs of similar tasks that passed, follow their approach:
    """
    hits = await experience.retrieve(task, limit=limit)

    return await hints([hit.value for hit in hits], task)


def _sections(
    runs: Sequence[Assessed[Any, Any]], rendered: Sequence[str], what: str
) -> str:
    """Rendered runs under a heading for those that passed and one for the failed ones."""
    titles = {
        True: f"Past runs of similar tasks that passed, follow their {what}:",
        False: "Past runs of similar tasks that failed, avoid their mistakes:",
    }
    sections = {passed: [title] for passed, title in titles.items()}

    for run, text in zip(runs, rendered, strict=True):
        sections[run.verdict.passed].append(text)

    return "\n\n".join(
        "\n\n".join(texts) for texts in sections.values() if len(texts) > 1
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Examples:
    """Renders the tasks and outputs of retrieved runs as few-shot examples with their verdicts.

    Examples:
        >>> import asyncio
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> from exar.experience import Retained
        >>> from exar.typing import Verdict
        >>> trace = run(FunctionAgent(func=lambda task: "5"), "2 + 2", RunConfig(policy=allow_all))
        >>> failed = Retained(trace=trace, verdict=Verdict(score=0.0, passed=False, reason="Wrong sum"))
        >>> print(asyncio.run(Examples()([failed], "2 + 3")))
        Past runs of similar tasks that failed, avoid their mistakes:
        <BLANKLINE>
        <example>
        <task>
        2 + 2
        </task>
        <output>
        5
        </output>
        <verdict score="0" passed="false">
        Wrong sum
        </verdict>
        </example>
    """

    async def __call__(
        self, runs: Sequence[Assessed[Any, Any]], task: object, /
    ) -> str:
        return _sections(
            runs,
            [render_example(r.trace.task, r.trace.output, r.verdict) for r in runs],
            "approach",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Trajectories:
    """Renders the trajectories of retrieved runs: which agents and tools in what order succeeded or failed.

    The trajectories of passed runs are adapted to the new task first.

    Args:
        adaptation: Adapts a trajectory, such as `Substitute` or a cbrkit adaptation in an `exar.cbrkit.Adapt`.

    Examples:
        >>> import asyncio
        >>> from exar.agents import FunctionAgent, RunConfig, call, run
        >>> from exar.policy import allow_all
        >>> from exar.experience import Retained
        >>> from exar.typing import Verdict
        >>> async def lead(task: str) -> str:
        ...     return await call(FunctionAgent(func=str.upper), task, name="writer")
        >>> passed = Retained(trace=run(lead, "hi", RunConfig(policy=allow_all)), verdict=Verdict(score=1.0, passed=True))
        >>> print(asyncio.run(Trajectories()([passed], "hello")))
        Past runs of similar tasks that passed, follow their trajectories:
        <BLANKLINE>
        <run>
        <task>
        hi
        </task>
        <trajectory>
        1. agent writer
        </trajectory>
        </run>
    """

    adaptation: Adaptation[Any, Trajectory] | None = None

    async def __call__(
        self, runs: Sequence[Assessed[Any, Any]], task: object, /
    ) -> str:
        async def render(run: Assessed[Any, Any]) -> str:
            trajectory = Trajectory.of(run.trace)

            if run.verdict.passed and self.adaptation is not None:
                trajectory = await self.adaptation(trajectory, run.trace.task, task)

            return "\n".join(
                (
                    "<run>",
                    tagged("task", run.trace.task),
                    tagged("trajectory", trajectory.render()),
                    "</run>",
                )
            )

        return _sections(runs, await asyncio.gather(*map(render, runs)), "trajectories")


@dataclass(frozen=True, slots=True, kw_only=True)
class Files:
    """Renders the files that retrieved runs changed in their workspace, such as the scripts they wrote and ran.

    The files are those `exar.pydantic_ai.RecordWorkspace` recorded (`FileChanges`),
    so a passed run offers its scripts for reuse and a failed one shows what to avoid.
    """

    async def __call__(
        self, runs: Sequence[Assessed[Any, Any]], task: object, /
    ) -> str:
        def render(run: Assessed[Any, Any]) -> str:
            changes = FileChanges.of(run.trace)
            files = () if changes is None or not changes.files else (changes.render(),)

            return "\n".join(
                ("<run>", tagged("task", run.trace.task), *files, "</run>")
            )

        return _sections(runs, [render(run) for run in runs], "files")


@dataclass(frozen=True, slots=True, kw_only=True)
class CbrCycle[I, O](Workflow[I, O]):
    """The case-based reasoning cycle over the runs of an agent (Meta-CBR), declarative and serializable.

    Retrieve: the runs of the most similar tasks are retrieved from the experience and rendered as hints.
    Reuse: the reuse agent solves the task with the hints, which `augment` adds to the task.
    Revise: the revise agent assesses the trace of the run, such as `exar.eval.Evaluate`.
    Retain: the assessed run is retained in the experience, as a negative case if it failed.
    A run whose reuse agent raised is retained as a negative case too, without its output and without being revised,
    and the error then propagates.
    Agents of frameworks may take the hints natively instead,
    such as by the capability `exar.pydantic_ai.RetrieveExperience`, so `augment` is left out.
    With a `ProcessReuse` as reuse agent, it reuses processes: retrieve, adapt, validate, run, revise, and retain.
    It writes the experience, so its effect is `"write"`, while the agents it invokes are checked on their own.

    Args:
        experience: The experience, such as an `exar.experience.StorageExperience`.
        reuse: The agent that solves the task.
        revise: The agent that assesses the trace of the run.
        hints: Renders the retrieved runs.
        augment: Adds the hints to the task, such as `lambda task, hints: f"{hints} {task}"` for text tasks, nothing is retrieved without it.
        limit: Most retrieved runs.

    Examples:
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.sim.strings import WordCoverage
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> from exar.experience import Retained, StorageExperience
        >>> from exar.typing import Verdict
        >>> async def check(trace):
        ...     return Retained(trace=trace, verdict=Verdict(score=1.0, passed=trace.output == "4"))
        >>> cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
        >>> cycle = CbrCycle[str, str](
        ...     experience=cases, reuse=FunctionAgent(func=lambda task: "4"), revise=check,
        ...     augment=lambda task, hints: f"{hints} {task}"
        ... )
        >>> from exar.otel import agents
        >>> first, second = (run(cycle, "Add 2 and 2", RunConfig(policy=allow_all)) for _ in range(2))
        >>> first.output, agents(second.root)
        ('4', ('cbr_cycle', '<lambda>', 'check'))
    """

    effect: ClassVar[Effect] = "write"
    name: ClassVar[str] = "cbr_cycle"
    experience: Experience[Any, I, O]
    reuse: Agent[I, O]
    revise: Agent[Trace[I, O], Assessed[I, O]]
    hints: Hints[I, O] = field(default_factory=Examples)
    augment: Callable[[I, str], I] | None = None
    limit: int = 3

    async def __call__(self, task: I, /) -> O:
        content = task

        if self.augment is not None and (
            text := await recall(self.experience, self.hints, task, limit=self.limit)
        ):
            content = self.augment(task, text)

        try:
            trace = replace(await traced(self.reuse, content), task=task)
        except AgentFailed as failed:
            # A failed run has no output, which experiences retain as None.
            failure = Retained(
                trace=replace(failed.trace, task=task),
                verdict=Verdict(score=0.0, passed=False),
            )
            await self.experience.retain([failure])

            raise

        await self.experience.retain([await call(self.revise, trace)])

        return trace.output
