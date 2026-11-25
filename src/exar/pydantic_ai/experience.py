"""Experience of past runs for pydantic-ai agents, injected natively by a capability.

Prompts render as text wherever exar renders a task, see `.content`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Self

from opentelemetry.util.genai.types import TextPart
from pydantic_ai import FunctionToolset, RunContext, UserContent
from pydantic_ai.capabilities import AbstractCapability

from exar.experience import Examples, recall
from exar.text import genai_parts
from exar.tools import effect
from exar.typing import Experience, Hints

from .tools import toolset

__all__ = ["RetrieveExperience", "prompt_text"]


def prompt_text(prompt: str | Sequence[UserContent], /) -> str | None:
    """The text of a prompt, its text parts joined by blank lines, None without any.

    Examples:
        >>> from pydantic_ai import ImageUrl
        >>> prompt_text("Describe it")
        'Describe it'
        >>> prompt_text(["Describe", ImageUrl("https://example.com/cat.png"), "the cat"])
        'Describe\\n\\nthe cat'
        >>> prompt_text([ImageUrl("https://example.com/cat.png")]) is None
        True
    """
    texts = [
        part.content for part in genai_parts(prompt) or () if isinstance(part, TextPart)
    ]

    return "\n\n".join(texts) if texts else None


@dataclass(kw_only=True)
class RetrieveExperience[D, I, O](AbstractCapability[D]):
    """Gives a pydantic-ai agent the hints of past runs of similar tasks, see `exar.experience`.

    `task` projects a prompt onto a task of the experience, such as the text of a multimodal prompt (`prompt_text`).
    With `mode="instructions"`, the runs most similar to the task of the prompt of a run are retrieved once when it starts
    and their hints become instructions of the run, nothing is added for prompts without a task or without hints.
    With `mode="tool"`, the model retrieves them on demand with the tool `retrieve_experience` for a task it describes as text.
    The full traces never reach the model, only what `hints` renders, such as `exar.experience.Trajectories`.
    Revising and retaining the runs is up to the host, such as an `exar.experience.CbrCycle` around the wrapped agent.

    Args:
        experience: The experience of runs of tasks.
        task: Projects a prompt, or the text task of the tool, onto a task of the experience, None to skip retrieval.
        hints: Renders the retrieved runs.
        mode: Whether the hints become instructions of every run or the model retrieves them with a tool.
        limit: Most retrieved runs.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.sim.strings import WordCoverage
        >>> from pydantic_ai import Agent, ImageUrl
        >>> from pydantic_ai.models.test import TestModel
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> from exar.experience import Retained, StorageExperience
        >>> from exar.typing import Verdict
        >>> cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
        >>> trace = run(FunctionAgent(func=lambda task: "4"), "Add 2 and 2", RunConfig(policy=allow_all))
        >>> _ = asyncio.run(cases.retain([Retained(trace=trace, verdict=Verdict(score=1, passed=True))]))
        >>> agent = Agent(TestModel(), capabilities=[RetrieveExperience(experience=cases, task=prompt_text)])
        >>> print(asyncio.run(agent.run(["Add 2 and 3", ImageUrl("https://example.com/sum.png")])).all_messages()[0].instructions)
        Past runs of similar tasks that passed, follow their approach:
        ...
    """

    experience: Experience[Any, I, O]
    task: Callable[[str | Sequence[UserContent]], I | None]
    hints: Hints[I, O] = field(default_factory=Examples)
    mode: Literal["instructions", "tool"] = "instructions"
    limit: int = 3
    text: str = field(default="", init=False)

    async def _hints(self, prompt: str | Sequence[UserContent]) -> str:
        task = self.task(prompt)

        if task is None:
            return ""

        return await recall(self.experience, self.hints, task, limit=self.limit)

    async def for_run(self, ctx: RunContext[D]) -> Self:
        """Retrieves the hints for the prompt of the run in the instructions mode."""
        if self.mode != "instructions" or ctx.prompt is None:
            return self

        capability = replace(self)
        capability.text = await self._hints(ctx.prompt)

        return capability

    def get_instructions(self) -> str | None:
        """The hints of the run, none while there are none."""
        return self.text or None

    def get_toolset(self) -> FunctionToolset[D] | None:
        """The tool `retrieve_experience` in the tool mode."""
        if self.mode != "tool":
            return None

        @effect("read")
        async def retrieve_experience(task: str) -> str:
            """Retrieves how past runs of tasks similar to a task went, to follow what worked and avoid what failed.

            Args:
                task: The task to retrieve similar past runs of.
            """
            return await self._hints(task) or "No similar past runs."

        return toolset([retrieve_experience])
