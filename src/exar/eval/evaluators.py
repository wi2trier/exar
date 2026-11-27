"""Native pydantic-evals evaluators for agentic applications: key facts, answerability, and judged questions.

They run in pydantic-evals datasets like its own evaluators, and on exar traces through `exar.eval.evaluate`.
The judges ask a pydantic-ai model, a `Model` or a model name such as `"openai:gpt-6.1-sol"`.
`Judged` asks an `exar.typing.Judge` a yes/no question about the output, such as a decision model, and scores its probability.
`KeyFacts` and `Judged` report a score under their name and whether it passed under `<name>_passed`, see `scored`,
and `exar.cbrkit.RetrievalMetric` adds the ranking metrics of cbrkit.

Examples:
    >>> import asyncio
    >>> from pydantic_evals import Case, Dataset
    >>> facts = KeyFacts(facts=[KeyFact(text="paris"), KeyFact(text="since 508", importance="nice")])
    >>> dataset = Dataset[str, str, None](name="capitals", cases=[Case(inputs="Capital of France?")], evaluators=[facts])
    >>> report = asyncio.run(dataset.evaluate(lambda question: asyncio.sleep(0, "Paris"), progress=False))
    >>> report.cases[0].scores["KeyFacts"].value, report.cases[0].assertions["KeyFacts_passed"].value
    (0.5, True)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ..agents import call
from ..text import as_text, tagged
from ..typing import Judge, Question

__all__ = [
    "Answerability",
    "AnswerabilityJudge",
    "JudgeModel",
    "Judged",
    "KeyFact",
    "KeyFacts",
    "scored",
]

KEY_FACTS = """Decide which key facts the answer states, in any wording.
Respond with the numbers of the stated key facts."""

ANSWERABILITY = """Classify how the answer treats the question:
answerable if it answers the question fully, partial if it answers only a part,
and unanswerable if it declines or states that the information is missing."""

type Answerability = Literal["answerable", "partial", "unanswerable"]
"""How an answer treats its question, after MTRAG (Katsis et al. 2025, arXiv 2501.03468)."""

type JudgeModel = Model | KnownModelName | str
"""A pydantic-ai model or model name."""


def scored(
    name: str, score: float, passed: bool, reason: str
) -> dict[str, EvaluationReason]:
    """A score under a name and whether it passed under `<name>_passed`, as the evaluators of exar report them."""
    return {
        name: EvaluationReason(value=score, reason=reason),
        f"{name}_passed": EvaluationReason(value=passed, reason=reason),
    }


async def _judge[T](model: JudgeModel, output_type: type[T], prompt: str) -> T:
    return (await Agent(model, output_type=output_type).run(prompt)).output


@dataclass(frozen=True, slots=True, kw_only=True)
class KeyFact:
    """A fact that a good answer states.

    Args:
        text: The fact, such as an excerpt of the gold answer.
        importance: Whether a good answer has to state it or merely should.
    """

    text: str
    importance: Literal["must", "nice"] = "must"


class _Stated(BaseModel):
    facts: list[int] = Field(description="Numbers of the key facts the answer states")


class _Classified(BaseModel):
    answerability: Answerability


@dataclass(repr=False, kw_only=True)
class KeyFacts(Evaluator[Any, Any, Any]):
    """Scores the share of key facts the output states, passing if it states every must fact.

    Without a model, the output states a fact that it contains ignoring case,
    a model also recognizes paraphrases.

    Args:
        facts: The key facts.
        model: The model that decides which facts the output states.
        evaluation_name: Name of the score in reports, the class name by default.
    """

    facts: Sequence[KeyFact]
    model: JudgeModel | None = None
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or self.get_serialization_name()

    async def evaluate(
        self, ctx: EvaluatorContext[Any, Any, Any]
    ) -> dict[str, EvaluationReason]:
        name = self.get_default_evaluation_name()

        if not self.facts:
            return scored(name, 1.0, True, "No key facts")

        output = as_text(ctx.output)

        if self.model is None:
            stated = {
                index
                for index, fact in enumerate(self.facts, 1)
                if fact.text.casefold() in output.casefold()
            }
        else:
            facts = "\n".join(
                f"{index}. {fact.text}" for index, fact in enumerate(self.facts, 1)
            )
            prompt = "\n\n".join(
                (KEY_FACTS, tagged("facts", facts), tagged("answer", output))
            )
            stated = set((await _judge(self.model, _Stated, prompt)).facts)

        missing = [
            fact for index, fact in enumerate(self.facts, 1) if index not in stated
        ]

        return scored(
            name,
            1 - len(missing) / len(self.facts),
            all(fact.importance != "must" for fact in missing),
            "Missing key facts: " + ", ".join(fact.text for fact in missing)
            if missing
            else "States every key fact",
        )


@dataclass(repr=False, kw_only=True)
class AnswerabilityJudge(Evaluator[Any, Any, Any]):
    """Asks a model whether the output treats the question in the inputs as answerable as expected.

    Args:
        expected: The gold answerability.
        model: The model that classifies the output.
        evaluation_name: Name of the assertion in reports, the class name by default.
    """

    expected: Answerability
    model: JudgeModel
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or self.get_serialization_name()

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> EvaluationReason:
        prompt = "\n\n".join(
            (
                ANSWERABILITY,
                tagged("question", ctx.inputs),
                tagged("answer", ctx.output),
            )
        )
        actual = (await _judge(self.model, _Classified, prompt)).answerability

        return EvaluationReason(
            value=actual == self.expected,
            reason=f"Treated as {actual}, expected {self.expected}",
        )


@dataclass(repr=False, kw_only=True)
class Judged(Evaluator[Any, Any, Any]):
    """Scores the probability of yes that a judge gives to a question about the inputs and the output.

    A decision model answers with a calibrated probability, so the score is graded rather than pass or fail.
    The judge sees the inputs and the output framed by tags, and the question names them as such.

    Args:
        question: The yes/no question, such as "Is the answer grounded in the question?".
        judge: Answers the question, such as `exar.pydantic_ai.judge` of a model.
        threshold: Probability from which the output passes.
        evaluation_name: Name of the score in reports, the class name by default.

    Examples:
        >>> import asyncio
        >>> from pydantic_evals import Case, Dataset
        >>> async def polite(question: Question) -> float:
        ...     return 0.9 if "please" in question.text else 0.1
        >>> judged = Judged(question="Is the answer polite?", judge=polite)
        >>> dataset = Dataset[str, str, None](name="manners", cases=[Case(inputs="Pass the salt")], evaluators=[judged])
        >>> report = asyncio.run(dataset.evaluate(lambda task: asyncio.sleep(0, "Here, please."), progress=False))
        >>> report.cases[0].scores["Judged"].value, report.cases[0].assertions["Judged_passed"].value
        (0.9, True)
    """

    question: str
    judge: Judge
    threshold: float = 0.5
    evaluation_name: str | None = field(default=None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or self.get_serialization_name()

    async def evaluate(
        self, ctx: EvaluatorContext[Any, Any, Any]
    ) -> dict[str, EvaluationReason]:
        text = f"{tagged('inputs', ctx.inputs)}\n{tagged('output', ctx.output)}"
        probability = await call(
            self.judge, Question(instructions=self.question, text=text), name="judge"
        )

        return scored(
            self.get_default_evaluation_name(),
            probability,
            probability >= self.threshold,
            f"Probability {probability:.2f} of yes to: {self.question}",
        )
