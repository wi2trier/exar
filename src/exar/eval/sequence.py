"""Learning curves: an application solves a sequence of test cases in order, each with what it learned so far.

Each test case is solved by the application as it is at that point, then evaluated against its gold data,
so every score measures how well the experience of the earlier test cases carries over to an unseen one.
What the application learns is up to the application, such as a `exar.experience.CbrCycle`,
which retrieves, reuses, revises, and retains its runs on its own,
while the gold data only ever reaches the evaluation.
Comparing the curve with that of a baseline, such as the same application without experience,
shows what the experience contributes, and repeating it with shuffled sequences separates learning from order.

Examples:
    >>> import asyncio
    >>> from pydantic_evals import Case, Dataset
    >>> from pydantic_evals.evaluators import EqualsExpected
    >>> from exar.agents import FunctionAgent, RunConfig, run_async
    >>> from exar.policy import allow_all
    >>> async def run(task: int) -> Trace[int, int]:
    ...     return await run_async(FunctionAgent(func=lambda x: x % 3), task, RunConfig(policy=allow_all))
    >>> dataset = Dataset[int, int, None](name="mod3", cases=[Case(inputs=x, expected_output=0) for x in range(4)], evaluators=[EqualsExpected()])
    >>> curve = asyncio.run(evaluate_sequence(dataset, run))
    >>> curve.pass_rates
    [1.0, 0.5, 0.3333333333333333, 0.5]
"""

import itertools
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from pydantic_evals import Dataset

from ..otel import usage
from ..typing import Trace
from .assessment import Assessment, _assess

__all__ = ["LearningCurve", "evaluate_sequence"]


def _running_mean(values: Iterable[float]) -> list[float]:
    return [
        total / count for count, total in enumerate(itertools.accumulate(values), 1)
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class LearningCurve[I, O]:
    """The assessed runs of a sequence of test cases with the cumulative measures after each one.

    Args:
        assessments: The assessed runs in the order of the sequence.
    """

    assessments: Sequence[Assessment[I, O]]

    @property
    def scores(self) -> list[float]:
        """Mean score of the runs so far."""
        return _running_mean(a.verdict.score for a in self.assessments)

    @property
    def pass_rates(self) -> list[float]:
        """Share of passing runs so far."""
        return _running_mean(float(a.verdict.passed) for a in self.assessments)

    @property
    def costs(self) -> list[Decimal | None]:
        """Total `cost` usage of the priced model calls of the runs so far, `None` until one is priced."""
        return [
            usage.cost
            for usage in itertools.accumulate(usage(a.trace.root) for a in self.assessments)
        ]


async def evaluate_sequence[I, O, M](
    dataset: Dataset[I, O, M],
    run: Callable[[I], Awaitable[Trace[I, O]]],
    /,
    *,
    done: Sequence[Assessment[I, O]] = (),
    threshold: float = 0.5,
) -> LearningCurve[I, O]:
    """Solves and evaluates the test cases of a dataset one after another.

    Args:
        dataset: The test cases in order, assessed by the evaluators of the dataset and their own with their gold data,
            see `exar.eval.evaluate`.
        run: Solves the inputs of a test case with what the application learned so far,
            such as with `exar.agents.run_async`.
        done: Assessed runs of the leading test cases from an earlier, interrupted evaluation,
            which are kept instead of solving these test cases again.
        threshold: The least mean score that passes when no evaluator asserts, see `evaluate`.

    Raises:
        ValueError: If the tasks of `done` are not the inputs of the leading test cases.
    """
    assessments = list(done)
    remaining = iter(dataset.cases)

    for index, assessment in enumerate(done):
        case = next(remaining, None)

        if case is None or assessment.trace.task != case.inputs:
            raise ValueError(
                f"The run {index} done before is not of the test case at its position"
            )

    for case in remaining:
        assessments.append(await _assess(case, run, dataset.evaluators, threshold))

    return LearningCurve(assessments=assessments)
