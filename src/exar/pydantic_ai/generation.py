"""Structured generation on a pydantic-ai model for every prompt, such as for the language models of cbrkit."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.settings import ModelSettings

from exar.agents import call
from exar.helpers import bounded_gather

from .agent import PydanticAiAgent
from .decision import _reading_agent

__all__ = ["Generation"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Generation:
    """Generates a response of a type for every prompt, each a traced invocation of its own, a bounded number at once.

    It has the signature of a `cbrkit.typing.GenerationFunc`, so a host binds it to the names of the language models
    of cbrkit, such as for `cbrkit.sim.llm.Critique`, `cbrkit.sim.llm.Fit`, or `cbrkit.adapt.llm.Rewrite`.
    A decision model fills a response type of decisions only, such as the probability of `cbrkit.sim.llm.Fit`.

    Args:
        model: The model, a decision model as well as a language model.
        settings: The settings of every request.
        concurrency: Prompts generated at once.
        name: Name of the invocations in traces.
        capabilities: Capabilities of every run beside `CountUsage`, such as retries of transient errors.

    Examples:
        >>> import asyncio
        >>> from pydantic import BaseModel
        >>> from pydantic_ai.models.test import TestModel
        >>> class Fitness(BaseModel):
        ...     fits: float
        >>> generate = Generation(model=TestModel(custom_output_args={"fits": 0.5}))
        >>> asyncio.run(generate(["a", "b"], Fitness))
        [Fitness(fits=0.5), Fitness(fits=0.5)]
    """

    model: Model | KnownModelName | str
    settings: ModelSettings | None = None
    concurrency: int = 4
    name: str = "generation"
    capabilities: Sequence[AbstractCapability[None]] = ()

    async def __call__[T](
        self, prompts: Sequence[str], response_type: type[T], /
    ) -> Sequence[T]:
        agent = _reading_agent(response_type, self.name, self.capabilities)
        # Resolved once, since a model name would otherwise build a client per prompt.
        model = infer_model(self.model)
        generate = PydanticAiAgent[str, T](
            run=lambda prompt: agent.run(
                prompt, model=model, model_settings=self.settings
            ),
            name=self.name,
            effect="read",
        )

        return await bounded_gather(
            prompts, lambda prompt: call(generate, prompt), limit=self.concurrency
        )
