"""Decisions on pydantic-ai models about a text, typed by an output type, a probability of yes for a judge.

A decision model, such as `system-one:<model>` for one served by vLLM, answers fast and calibrated,
and a language model answers the same output type as structured output.
"""

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.models.decision import DecisionModel
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings

from exar.typing import Decide, Judge, Question

from .agent import CountUsage, PydanticAiAgent

__all__ = ["decide", "judge"]

_PROBABILITY = "Answer with the probability from 0 to 1 that the answer is yes."
"""What a language model is told beside a yes/no question, which a decision model answers with a probability by itself."""


def _reading_agent[T](
    output_type: OutputSpec[T],
    name: str,
    capabilities: Sequence[AbstractCapability[None]] = (),
) -> Agent[None, T]:
    """A pydantic-ai agent of an output type without deps, counting its usage toward the exar run."""
    return Agent[None, T](
        output_type=output_type,
        deps_type=type(None),
        name=name,
        capabilities=[CountUsage[None](), *capabilities],
    )


def _decide[T](
    model: Model | KnownModelName | str,
    output_type: OutputSpec[T],
    *,
    settings: ModelSettings | None,
    name: str,
    capabilities: Sequence[AbstractCapability[None]],
    framing: str = "",
) -> PydanticAiAgent[Question, T]:
    resolved = infer_model(model)
    suffix = (
        f"\n\n{framing}" if framing and not isinstance(resolved, DecisionModel) else ""
    )
    agent = _reading_agent(output_type, name, capabilities)

    async def run(question: Question) -> AgentRunResult[T]:
        return await agent.run(
            question.text,
            model=resolved,
            instructions=question.instructions + suffix,
            model_settings=settings,
        )

    return PydanticAiAgent(run=run, name=name, effect="read")


def decide[T](
    model: Model | KnownModelName | str,
    output_type: OutputSpec[T],
    /,
    *,
    settings: ModelSettings | None = None,
    name: str = "decide",
    capabilities: Sequence[AbstractCapability[None]] = (),
) -> Decide[T]:
    """A `exar.typing.Decide` on a pydantic-ai model, which reads only, deciding the output type about the text of a question.

    A decision model fills the decisions of the output type, a `bool`, a `Literal` or `Enum` to pick from,
    a `float` bounded to 0 and 1 for a probability, a rubric of whole numbers, a `list` of options,
    or a model of these, whose field descriptions are its questions,
    while a language model fills any output type as structured output.
    The capabilities, such as retries of transient errors, apply to every run beside `CountUsage`.

    Examples:
        >>> import asyncio
        >>> from typing import Literal
        >>> from pydantic_ai.models.test import TestModel
        >>> label = decide(TestModel(custom_output_args="bug"), Literal["billing", "bug"])
        >>> asyncio.run(label(Question(instructions="Which label fits the ticket?", text="Checkout fails since 9am.")))
        'bug'
    """
    return _decide(
        model, output_type, settings=settings, name=name, capabilities=capabilities
    )


def judge(
    model: Model | KnownModelName | str,
    /,
    *,
    settings: ModelSettings | None = None,
    name: str = "judge",
    capabilities: Sequence[AbstractCapability[None]] = (),
) -> Judge:
    """A `exar.typing.Judge` on a pydantic-ai model, the `decide` of the probability of yes.

    A decision model answers with its calibrated probability, while a language model is asked for its estimate.

    Examples:
        >>> import asyncio
        >>> from pydantic_ai.models.test import TestModel
        >>> asyncio.run(judge(TestModel(custom_output_args=0.8))(Question(instructions="Is it a greeting?", text="Hi")))
        0.8
    """
    # Without a description, since a decision model would take that as the question.
    probability = Annotated[float, Field(ge=0, le=1)]

    return _decide(
        model,
        probability,
        settings=settings,
        name=name,
        capabilities=capabilities,
        framing=_PROBABILITY,
    )
