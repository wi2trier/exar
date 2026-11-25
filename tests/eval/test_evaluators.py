from dataclasses import dataclass

from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, HasMatchingSpan

from exar.agents import FunctionAgent, RunConfig, run_async
from exar.eval import AnswerabilityJudge, KeyFact, KeyFacts, evaluate
from exar.eval.assessment import span_of, span_tree
from exar.policy import allow_all
from exar.typing import Trace, Verdict


def judge(response: dict[str, object], prompts: list[str]) -> FunctionModel:
    """A model that answers with a fixed structured output and records its prompts."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.append(str(messages[-1].parts[-1]))

        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, response)])

    return FunctionModel(respond)


async def answer(output: str, question: str = "Capital of France?") -> Trace[str, str]:
    def solve(task: str) -> str:
        return output

    return await run_async(
        FunctionAgent(func=solve), question, RunConfig(policy=allow_all)
    )


FACTS = [KeyFact(text="Paris"), KeyFact(text="since 508", importance="nice")]


async def test_key_facts() -> None:
    prompts: list[str] = []
    assessment = await evaluate(await answer("It is paris."), [KeyFacts(facts=FACTS)])

    assert (assessment.scores, assessment.assertions) == (
        {"KeyFacts": 0.5},
        {"KeyFacts_passed": True},
    )

    model = judge({"facts": [2]}, prompts)
    assessment = await evaluate(
        await answer("Capital since 508."), [KeyFacts(facts=FACTS, model=model)]
    )

    assert (assessment.verdict.score, assessment.verdict.passed) == (0.5, False)
    assert "1. Paris" in prompts[0]


async def test_answerability_and_spans_with_case() -> None:
    case = Case[str, str, None](
        inputs="Capital of France?",
        evaluators=(HasMatchingSpan(query={"name_equals": "invoke_agent solve"}),),
    )
    model = judge({"answerability": "answerable"}, [])
    trace = await answer("Paris")
    assessment = await evaluate(
        trace, [AnswerabilityJudge(expected="unanswerable", model=model)], case=case
    )

    assert assessment.assertions == {
        "AnswerabilityJudge": False,
        "HasMatchingSpan": True,
    }
    assert assessment.verdict == Verdict(
        score=0.5,
        passed=False,
        reason="AnswerabilityJudge: Treated as answerable, expected unanswerable",
    )
    assert span_of(span_tree(trace).roots[0]).attributes == trace.root.attributes


async def test_score_only_verdict_passes_at_the_threshold() -> None:
    @dataclass
    class Score(Evaluator[str, str, None]):
        value: float

        def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> float:
            return self.value

    trace = await answer("Paris")

    assert not (await evaluate(trace, [Score(0.0)])).verdict.passed
    assert (await evaluate(trace, [Score(0.7)])).verdict.passed
    assert not (await evaluate(trace, [Score(0.7)], threshold=0.8)).verdict.passed
