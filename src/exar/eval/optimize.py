"""Reflective prompt optimization: a language model improves instructions from the failures of runs.

`optimize` follows GEPA (Agrawal et al. 2025, arXiv 2507.19457):
it runs each candidate instruction on every case of a pydantic-evals dataset, assessed by its evaluators and those of the case,
picks a parent from the Pareto front of the candidates that are best on at least one task,
shows a proposer model of pydantic-ai failed runs of the parent with the reasons of their verdicts,
and scores the improved instructions it proposes as a new candidate.
Instructions are plain strings, storing the best one is up to the caller.
`render_examples` renders runs for a prompt, also as few-shot examples.
"""

import random
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from pydantic_ai import Agent
from pydantic_evals import Dataset

from ..helpers import bounded_gather
from ..text import render_example, tagged
from ..typing import Assessed, Trace
from .assessment import Assessment, _assess
from .evaluators import JudgeModel

__all__ = ["REFLECTION", "Candidate", "optimize", "render_examples"]

REFLECTION = """Improve the instructions of an agent from runs where it fell short.
Find the causes of the failures in the verdicts, keep what works, and generalize beyond the examples.
Respond with the complete improved instructions only."""


def render_examples(assessments: Iterable[Assessed[Any, Any]], /) -> str:
    """Renders assessed runs as examples for a prompt.

    Examples:
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> from exar.experience import Retained
        >>> from exar.typing import Verdict
        >>> trace = run(FunctionAgent(func=lambda task: 5), "2 + 2", RunConfig(policy=allow_all))
        >>> verdict = Verdict(score=0.0, passed=False, reason="Wrong sum")
        >>> print(render_examples([Retained(trace=trace, verdict=verdict)]))
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
    return "\n\n".join(
        render_example(a.trace.task, a.trace.output, a.verdict) for a in assessments
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate[I, O]:
    """Instructions with their assessed runs on the cases of the dataset.

    Args:
        prompt: The instructions.
        assessments: The assessed run of each case of the dataset, in order.
        parent: The instructions this candidate improves, if any.
    """

    prompt: str
    assessments: Sequence[Assessment[I, O]]
    parent: str | None = None

    @property
    def scores(self) -> list[float]:
        """Score of each case."""
        return [a.verdict.score for a in self.assessments]

    @property
    def score(self) -> float:
        """Mean score over the tasks."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


def _select[C: Candidate[Any, Any]](candidates: Sequence[C], rng: random.Random) -> C:
    """Samples a candidate of the Pareto front, weighted by the number of tasks it is best on."""
    wins: Counter[int] = Counter()

    for scores in zip(*(c.scores for c in candidates), strict=True):
        best = max(scores)
        wins.update(i for i, value in enumerate(scores) if value == best)

    if not wins:
        return candidates[0]

    [index] = rng.choices(list(wins), weights=list(wins.values()))

    return candidates[index]


async def optimize[I, O, M](
    prompt: str,
    dataset: Dataset[I, O, M],
    run: Callable[[str, I], Awaitable[Trace[I, O]]],
    propose: JudgeModel,
    /,
    *,
    budget: int = 10,
    examples: int = 3,
    instructions: str = REFLECTION,
    seed: int | None = None,
    concurrency: int = 8,
    threshold: float = 0.5,
) -> list[Candidate[I, O]]:
    """Searches for instructions that score best on a dataset, starting from a prompt.

    The search stops early once a parent passes every case.

    Args:
        prompt: The initial instructions.
        dataset: The cases to score the instructions on, such as a training split,
            assessed by the evaluators of the dataset and their own.
        run: Solves the inputs of a case with instructions.
        propose: The pydantic-ai model that proposes improved instructions.
        budget: Most proposals, each scored on every case.
        examples: Most failed runs shown per proposal.
        instructions: How to improve, followed by the parent instructions and the failed runs.
        seed: Seed of the sampling of parents and failed runs.
        concurrency: Most cases solved at once.
        threshold: The least mean score that passes when no evaluator asserts, see `evaluate`.

    Returns:
        The scored candidates, the best first.
    """
    rng = random.Random(seed)
    proposer = Agent(propose, output_type=str)

    async def score(text: str, parent: str | None) -> Candidate[I, O]:
        assessments = await bounded_gather(
            dataset.cases,
            lambda case: _assess(
                case, partial(run, text), dataset.evaluators, threshold
            ),
            limit=concurrency,
        )

        return Candidate(prompt=text, assessments=assessments, parent=parent)

    candidates = [await score(prompt, None)]

    for _ in range(budget):
        parent = _select(candidates, rng)
        failed = [a for a in parent.assessments if not a.verdict.passed]

        if not failed:
            break

        shown = rng.sample(failed, min(examples, len(failed)))
        reflection = "\n\n".join(
            (
                instructions,
                tagged("instructions", parent.prompt),
                render_examples(shown),
            )
        )
        proposal = (await proposer.run(reflection)).output
        candidates.append(await score(proposal.strip(), parent.prompt))

    return sorted(candidates, key=lambda c: c.score, reverse=True)
