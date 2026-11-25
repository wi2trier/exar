"""The ranking metrics of cbrkit as a pydantic-evals evaluator."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cbrkit.eval import Metric
from cbrkit.typing import EvalMetricFunc
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ..eval.evaluators import scored

__all__ = ["RetrievalMetric"]


def _output_scores(ctx: EvaluatorContext[Any, Any, Any]) -> Mapping[str, float]:
    return ctx.output


@dataclass(repr=False, kw_only=True)
class RetrievalMetric(Evaluator[object, object, object]):
    """Scores the items a run retrieved by a cbrkit retrieval metric against relevance judgments.

    Args:
        qrels: Relevance of the items for the task, higher is more relevant.
        retrieved: Reads the scores of the retrieved items by key from the context,
            the output by default, or such as the results of a search tool from `ctx.span_tree`.
        metric: A metric of `cbrkit.eval` with values from 0 to 1, such as `Metric(name="recall", k=10)`.
        threshold: The least passing value of the metric.
        evaluation_name: Name of the score in reports, the class name by default.

    Examples:
        >>> import asyncio
        >>> from pydantic_evals import Case, Dataset
        >>> metric = RetrievalMetric(qrels={"a": 2, "b": 1}, metric=Metric(name="recall"))
        >>> dataset = Dataset[str, dict[str, float], None](name="search", cases=[Case(inputs="cbr")], evaluators=[metric])
        >>> report = asyncio.run(dataset.evaluate(lambda q: asyncio.sleep(0, {"a": 0.9, "c": 0.4}), progress=False))
        >>> report.cases[0].scores["RetrievalMetric"].value
        0.5
    """

    qrels: Mapping[str, int]
    retrieved: Callable[[EvaluatorContext[Any, Any, Any]], Mapping[str, float]] = (
        _output_scores
    )
    metric: EvalMetricFunc = field(default_factory=lambda: Metric(name="ndcg"))
    threshold: float = 0.5
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or self.get_serialization_name()

    def evaluate(
        self, ctx: EvaluatorContext[object, object, object]
    ) -> dict[str, EvaluationReason]:
        value = self.metric(self.qrels, self.retrieved(ctx))
        name = self.get_default_evaluation_name()

        if value is None:
            return scored(name, 0.0, False, f"{self.metric} is undefined")

        return scored(
            name,
            min(max(value, 0.0), 1.0),
            value >= self.threshold,
            f"{self.metric} is {value:.3g}",
        )
