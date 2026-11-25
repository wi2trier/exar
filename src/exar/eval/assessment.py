"""Assessments of exar traces by pydantic-evals evaluators, and conversions between traces and span trees.

`evaluate` runs pydantic-evals evaluators on a trace, with the gold data of a `pydantic_evals.Case`,
through the public `pydantic_evals.online.run_evaluators` for evaluating stored runs.
The evaluators see the spans of the trace as `ctx.span_tree`, see `span_tree`,
so span-based evaluators such as `HasMatchingSpan` work on traces of every framework.
The `Assessment` keeps the trace with the evaluation results, which experience and learning curves summarize as a verdict.

Examples:
    >>> import asyncio
    >>> from pydantic_evals.evaluators import Contains, HasMatchingSpan
    >>> from exar.agents import FunctionAgent, RunConfig, run
    >>> from exar.policy import allow_all
    >>> trace = run(FunctionAgent(func=lambda question: "Paris"), "Capital of France?", RunConfig(policy=allow_all))
    >>> evaluators = [Contains(value="Paris"), HasMatchingSpan(query={"name_contains": "invoke_agent"})]
    >>> assessment = asyncio.run(evaluate(trace, evaluators))
    >>> assessment.verdict
    Verdict(score=1.0, passed=True, reason='')
"""

import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python
from pydantic_evals import Case
from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import run_evaluators
from pydantic_evals.otel import SpanNode, SpanTree

from ..otel import usage
from ..typing import Effect, Evaluation, Span, Trace, Verdict

_ATTRIBUTES = TypeAdapter(dict[str, JsonValue])

__all__ = ["Assessment", "Evaluate", "context", "evaluate", "span_of", "span_tree"]


def _nodes(span: Span, trace_id: int) -> list[SpanNode]:
    node = SpanNode(
        name=span.name,
        trace_id=trace_id,
        span_id=int(span.span_id, 16),
        parent_span_id=None if span.parent_id is None else int(span.parent_id, 16),
        start_timestamp=span.start,
        end_timestamp=span.end,
        attributes=dict(span.attributes),
        status=span.status,
    )

    return [node, *(n for child in span.children for n in _nodes(child, trace_id))]


def span_tree(trace: Trace[Any, Any], /) -> SpanTree:
    """The spans of a trace as a pydantic-evals span tree.

    Examples:
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> tree = span_tree(run(FunctionAgent(func=len), "abc", RunConfig(policy=allow_all)))
        >>> [node.name for node in tree]
        ['invoke_agent len']
    """
    tree = SpanTree()
    tree.add_spans(_nodes(trace.root, int(trace.id, 16)))

    return tree


def span_of(node: SpanNode, /) -> Span:
    """The record of a node of a pydantic-evals span tree with the nodes below it, the inverse of `span_tree`.

    Examples:
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> trace = run(FunctionAgent(func=len), "abc", RunConfig(policy=allow_all))
        >>> from dataclasses import replace
        >>> span_of(span_tree(trace).roots[0]) == replace(trace.root, scope="", error=None)
        True
    """
    return Span(
        name=node.name,
        span_id=f"{node.span_id:016x}",
        parent_id=None
        if node.parent_span_id is None
        else f"{node.parent_span_id:016x}",
        start=node.start_timestamp,
        end=node.end_timestamp,
        status=node.status,
        attributes=_ATTRIBUTES.validate_python(to_jsonable_python(node.attributes)),
        children=tuple(span_of(child) for child in node.children),
    )


def context[I, O, M](
    trace: Trace[I, O], /, case: Case[I, O, M] | None = None
) -> EvaluatorContext[I, O, M]:
    """The context of pydantic-evals evaluators for a trace, with the gold data of a case."""
    return EvaluatorContext[I, O, M](
        name=None if case is None else case.name,
        inputs=trace.task,
        metadata=None if case is None else case.metadata,
        expected_output=None if case is None else case.expected_output,
        output=trace.output,
        duration=(trace.root.end - trace.root.start).total_seconds(),
        _span_tree=span_tree(trace),
        attributes={},
        metrics={}
        if (cost := usage(trace.root).cost) is None
        else {"cost": float(cost)},
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Assessment[I, O]:
    """A trace with the results of pydantic-evals evaluators, an `exar.typing.Assessed`.

    Args:
        trace: The run.
        results: The results of the evaluators, scores, assertions, and labels alike.
        failures: The evaluators that raised.
        threshold: The least mean score that passes when no evaluator asserts.
    """

    trace: Trace[I, O]
    results: Sequence[EvaluationResult] = ()
    failures: Sequence[EvaluatorFailure] = field(
        default_factory=tuple[EvaluatorFailure, ...]
    )
    threshold: float = 0.5

    @property
    def scores(self) -> dict[str, float]:
        """The numeric results by name."""
        return {
            result.name: float(result.value)
            for result in self.results
            if isinstance(result.value, int | float)
            and not isinstance(result.value, bool)
        }

    @property
    def evaluations(self) -> tuple[Evaluation, ...]:
        """The results as the evaluations of `exar.typing.Assessed`."""
        return tuple(
            Evaluation(name=result.name, value=result.value, reason=result.reason)
            for result in self.results
        )

    @property
    def assertions(self) -> dict[str, bool]:
        """The boolean results by name."""
        return {
            result.name: result.value
            for result in self.results
            if isinstance(result.value, bool)
        }

    @property
    def verdict(self) -> Verdict:
        """The summary of the results.

        The score is the mean of the scores, or else the share of passed assertions, 1 without either.
        It passes if no evaluator failed and every assertion holds,
        or without assertions if the score reaches the `threshold`,
        and the reason lists the reasons of the results and the failures.
        """
        scores, assertions = self.scores, self.assertions
        values = list(scores.values()) or [
            float(value) for value in assertions.values()
        ]
        reasons = [
            *(f"{r.name}: {r.reason}" for r in self.results if r.reason),
            *(f"{f.name} failed: {f.error_message}" for f in self.failures),
        ]

        score = statistics.fmean(values) if values else 1.0
        holds = all(assertions.values()) if assertions else score >= self.threshold

        return Verdict(
            score=score,
            passed=holds and not self.failures,
            reason="\n".join(reasons),
        )


async def evaluate[I, O, M](
    trace: Trace[I, O],
    evaluators: Sequence[Evaluator[I, O, M]] = (),
    /,
    *,
    case: Case[I, O, M] | None = None,
    threshold: float = 0.5,
) -> Assessment[I, O]:
    """Assesses a trace with pydantic-evals evaluators and those of a case.

    Args:
        trace: The run.
        evaluators: The evaluators, such as those of a dataset.
        case: The case of the task, whose name, metadata, expected output, and evaluators apply.
        threshold: The least mean score that passes when no evaluator asserts.
    """
    results, failures = await run_evaluators(
        [*evaluators, *(() if case is None else case.evaluators)], context(trace, case)
    )

    return Assessment(
        trace=trace, results=results, failures=failures, threshold=threshold
    )


async def _assess[I, O, M](
    case: Case[I, O, M],
    run: Callable[[I], Awaitable[Trace[I, O]]],
    evaluators: Sequence[Evaluator[I, O, M]],
    threshold: float,
) -> Assessment[I, O]:
    """Solves the inputs of a case of a dataset and assesses the run with the evaluators of the dataset and the case."""
    return await evaluate(
        await run(case.inputs), evaluators, case=case, threshold=threshold
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Evaluate[I, O, M]:
    """An agent that assesses the trace of a run with pydantic-evals evaluators, see `evaluate`.

    It is the revise agent of an `exar.experience.CbrCycle`, which retains the assessed run.

    Args:
        evaluators: The evaluators.
        threshold: The least mean score that passes when no evaluator asserts.

    Examples:
        >>> from pydantic_evals.evaluators import Contains
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> config = RunConfig(policy=allow_all)
        >>> trace = run(FunctionAgent(func=lambda question: "Paris"), "Capital of France?", config)
        >>> run(Evaluate(evaluators=[Contains(value="Paris")]), trace, config).output.verdict.passed
        True
    """

    effect: ClassVar[Effect] = "read"
    evaluators: Sequence[Evaluator[I, O, M]]
    threshold: float = 0.5

    async def __call__(self, trace: Trace[I, O], /) -> Assessment[I, O]:
        return await evaluate(trace, self.evaluators, threshold=self.threshold)
