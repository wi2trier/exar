from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai.models.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
)

from exar.pydantic_ai import decide, judge
from exar.typing import Question


@dataclass(init=False)
class Recorder(DecisionModel[None]):
    """A decision model that picks the first option and answers yes with 0.75, recording its requests."""

    requests: list[DecisionRequest] = field(default_factory=list[DecisionRequest])

    def __init__(self) -> None:
        super().__init__()
        self.requests = []

    @property
    def model_name(self) -> str:
        return "recorder"

    @property
    def system(self) -> str:
        return "recorder"

    @property
    def base_url(self) -> None:
        return None

    async def decide(
        self, request: DecisionRequest, model_settings: DecisionModelSettings
    ) -> DecisionResponse:
        self.requests.append(request)

        def answer(question: DecisionQuestion) -> DecisionAnswer:
            if isinstance(question, ChoiceQuestion):
                first = next(iter(question.criteria))
                return ChoiceAnswer(
                    choice=first, confidence=1.0, probabilities={first: 1.0}
                )

            return NoulAnswer(noul=0.75)

        return DecisionResponse(
            answers={
                name: answer(question) for name, question in request.questions.items()
            },
            model_name="recorder",
        )


class Triage(BaseModel):
    """How to handle a ticket."""

    label: Literal["billing", "bug"] = Field(description="Which label fits the ticket?")
    urgent: bool = Field(description="Is the ticket urgent?")


async def test_decision_models_answer_typed_decisions_and_probabilities() -> None:
    model = Recorder()
    question = Question(
        instructions="Triage the ticket.", text="Checkout fails since 9am."
    )

    triage = await decide(model, Triage)(question)
    probability = await judge(model)(
        Question(instructions="Is it urgent?", text="Asap!")
    )

    assert triage == Triage(label="billing", urgent=True)
    assert probability == 0.75
    assert [request.state for request in model.requests] == [
        "Checkout fails since 9am.",
        "Asap!",
    ]
    # A decision model is asked the question alone, a language model would also be told to answer a probability.
    assert model.requests[1].questions["response"].instructions == "Is it urgent?"
