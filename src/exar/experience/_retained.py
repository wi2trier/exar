"""Assessed runs, the solutions of the cases of experience."""

from dataclasses import dataclass

from ..typing import Assessed, Evaluation, Trace, Verdict

__all__ = ["Retained"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Retained[I, O]:
    """An assessed run, the solution of a case of experience, and the simplest `exar.typing.Assessed`.

    It keeps the full trace, so measures and reuse extract the features they need from it,
    such as the output, the trajectory (`Trajectory.of`), or the process (`Process.of`).
    It is JSON through pydantic, `TypeAdapter(Retained[I, O])`, if the tasks and outputs are,
    so it is what an experience stores and what a remote one sends.

    Args:
        trace: The run.
        verdict: Its assessment.
        evaluations: The results of the evaluators the verdict summarizes.
    """

    trace: Trace[I, O]
    verdict: Verdict
    evaluations: tuple[Evaluation, ...] = ()

    @classmethod
    def of(cls, assessment: Assessed[I, O], /) -> Retained[I, O]:
        """The assessed run of any assessment, such as an `exar.eval.Assessment`."""
        return cls(
            trace=assessment.trace,
            verdict=assessment.verdict,
            evaluations=tuple(assessment.evaluations),
        )
