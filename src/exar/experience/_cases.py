"""Assessed runs as cases in a cbrkit storage, the casebase that retrieves, retains, and forgets them."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import cbrkit.model
from cbrkit.cycle import run_query_async
from cbrkit.model import Case, replan
from cbrkit.retain import Retainer
from cbrkit.retrieve import Retriever
from cbrkit.typing import AnyCbrFunc, AnySimFunc, Float, Storage

from ..cbrkit._convert import from_changes
from ..memory._common import top_hits
from ..typing import Assessed, Changes, Hit
from ._retained import Retained

__all__ = ["StorageExperience"]

type _Case[I, O] = Case[I, Retained[I, O | None]]
"""A case of experience, whose solution is a run without an output if it failed with an error."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StorageExperience[K, I, O]:
    """Assessed runs as split cases in a cbrkit storage, the first-party `exar.typing.Experience`.

    A case is a `cbrkit.model.Case` whose problem is the task of a run and whose solution is the `Retained` run,
    whose output is None if the run failed with an error.
    The retriever retrieves the runs of the most similar tasks, a `cbrkit.retrieve.Retriever` of the measure by default,
    failed runs are negative cases whose similarity is reduced by `penalty`,
    and a `cbrkit.retain.Retainer` retains runs as one mutation of the storage.
    Describe tasks for the measure with `cbrkit.sim.Transpose`, such as the draft of a state,
    and maintain the cases with `cbrkit.maintain`, such as `replan(Maintainer(measure=..., duplicate=0.95), cases.storage)`.

    Args:
        storage: The cases, such as a `cbrkit.indexable.InMemory` or a `cbrkit.io.TypedStorage` over a `cbrkit.io.FileStorage`.
        measure: Similarity of the task of a case to a new task, any measure of `cbrkit.sim`.
        retriever: Builds the steps of a retrieval for its limit, such as a storage retriever of `cbrkit.indexable`,
            a `cbrkit.retrieve.Retriever` of the measure over all cases if None.
            The penalty and `passed` apply to the cases the steps keep, so steps that keep more than the limit leave room for them.
        retainer: Retains the runs, whose storage and missing measure the casebase fills in,
            such as with a `duplicate` threshold and a `merge`.
        penalty: Factor of the similarity of failed runs.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.sim.strings import WordCoverage
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.experience import Retained
        >>> from exar.typing import Verdict
        >>> from exar.policy import allow_all
        >>> cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
        >>> for task, output in [("Add 2 and 2", "4"), ("Add 2 and 3", "6")]:
        ...     trace = run(FunctionAgent(func=lambda _: output), task, RunConfig(policy=allow_all))
        ...     verdict = Verdict(score=float(output == "4"), passed=output == "4")
        ...     changes = asyncio.run(cases.retain([Retained(trace=trace, verdict=verdict)]))
        >>> [(change.op, change.key) for change in changes.items]
        [('add', 1)]
        >>> [(hit.value.trace.output, hit.score) for hit in asyncio.run(cases.retrieve("Add 2 and 3"))]
        [('4', 0.75), ('6', 0.5)]
    """

    storage: Storage[K, _Case[I, O]]
    measure: AnySimFunc[I, Float]
    retriever: Callable[[int], Sequence[AnyCbrFunc[K, _Case[I, O], Float]]] | None = (
        None
    )
    retainer: Retainer[K, _Case[I, O]] = field(default_factory=Retainer)
    penalty: float = 0.5

    async def retrieve(
        self, task: I, /, *, limit: int = 3, passed: bool = False
    ) -> list[Hit[K, Assessed[I, O | None]]]:
        """The `limit` runs of the tasks most similar to a task, failed ones with their similarity reduced by `penalty`.

        Raises:
            ValueError: If the limit is negative.
        """
        if limit < 0:
            raise ValueError(f"limit must not be negative, got {limit}")

        steps = (
            [Retriever(measure=self.measure)]
            if self.retriever is None
            else self.retriever(limit)
        )
        result = await run_query_async(self.storage, Case(problem=task), *steps)
        runs = {key: _assessed(case) for key, case in result.casebase.items()}
        scores = {
            key: float(value) * (1.0 if runs[key].verdict.passed else self.penalty)
            for key, value in result.similarities.items()
            if not passed or runs[key].verdict.passed
        }
        return top_hits(scores, runs, limit)

    async def retain(
        self, assessments: Sequence[Assessed[I, O | None]], /
    ) -> Changes[K, Assessed[I, O | None]]:
        """Stores assessed runs, as positive cases if they passed and as negative cases otherwise.

        Returns:
            The applied changes of the cases that store the runs, new or merged into.
        """
        retainer = replace(
            self.retainer,
            storage=self.storage,
            measure=self.retainer.measure or self.measure,
        )
        results = await retainer(
            [
                (
                    {
                        0: Case(
                            problem=assessment.trace.task,
                            solution=Retained.of(assessment),
                        )
                    },
                    Case(problem=assessment.trace.task),
                )
                for assessment in assessments
            ]
        )
        # Runs merged into one case share its change.
        changes = {
            key: similarities[key].change(key, case)
            for cases, similarities in results
            for key, case in cases.items()
        }

        return _runs(cbrkit.model.Changes(items=list(changes.values()), applied=True))

    async def forget(
        self, select: Callable[[Assessed[I, O | None]], bool], /, *, reason: str
    ) -> Changes[K, Assessed[I, O | None]]:
        """Removes the runs a predicate selects, such as the runs of a trace id or of a user.

        Examples:
            >>> import asyncio
            >>> from cbrkit.indexable import InMemory
            >>> from cbrkit.sim.strings import WordCoverage
            >>> from exar.agents import FunctionAgent, RunConfig, run
            >>> from exar.experience import Retained
            >>> from exar.typing import Verdict
            >>> from exar.policy import allow_all
            >>> trace = run(FunctionAgent(func=str.upper), "hi", RunConfig(policy=allow_all))
            >>> cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
            >>> _ = asyncio.run(cases.retain([Retained(trace=trace, verdict=Verdict(score=1, passed=True))]))
            >>> changes = asyncio.run(cases.forget(lambda run: run.trace.id == trace.id, reason="Requested"))
            >>> [(change.op, change.key) for change in changes.items], dict(asyncio.run(cases.storage.get_index()))
            ([('remove', 0)], {})
        """

        return _runs(await replan(_Forget[K, I, O](select, reason), self.storage))


@dataclass(frozen=True, slots=True)
class _Forget[K, I, O]:
    """Plans the removal of the cases whose runs a predicate selects, a `cbrkit.model.Planner`."""

    select: Callable[[Assessed[I, O | None]], bool]
    reason: str

    @property
    def destructive(self) -> bool:
        return True

    def plan(
        self, stored: Mapping[K, _Case[I, O]], /
    ) -> cbrkit.model.Changes[K, _Case[I, O]]:
        return cbrkit.model.Changes(
            items=[
                cbrkit.model.Remove[K](key=key, reason=self.reason)
                for key, case in stored.items()
                if self.select(_assessed(case))
            ],
            expected=stored,
        )


def _assessed[I, O](case: _Case[I, O]) -> Retained[I, O | None]:
    if case.solution is None:
        raise ValueError("A case of experience has no run")

    return case.solution


def _runs[K, I, O](
    changes: cbrkit.model.Changes[K, _Case[I, O]], /
) -> Changes[K, Assessed[I, O | None]]:
    """The changes of cases as the changes of the runs they store."""
    return from_changes(changes, _assessed)
