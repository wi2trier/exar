"""Adapting reused solutions to a new task: substituting unavailable agents and tools, and rewriting by a language model."""

from collections.abc import Mapping
from dataclasses import dataclass
from operator import itemgetter

from ..memory import word_coverage
from ..typing import Measure, Solution

__all__ = ["REWRITE", "Substitute"]

REWRITE = """Adapt the solution of the case, a similar past task, to the problem of the query, the new task.
Keep what fits, change what the new task needs, and refer only to the available agents and tools by their names.
Respond with the adapted solution in the format of the given one."""
"""Instructions of a `cbrkit.adapt.llm.Rewrite` of solutions in an `exar.cbrkit.Adapt`, given the available agents and tools as context."""


@dataclass(slots=True, frozen=True, kw_only=True)
class Substitute:
    """Replaces references to agents and tools the deployment lacks by the most similar available ones.

    It is a deterministic adaptation that needs no language model:
    the measure scores the descriptions of the available agents and tools for the name of a missing one,
    and the best one above `min_similarity` replaces it.
    It adapts any `Solution`, such as a `Trajectory` or a `Process`.

    Args:
        available: Descriptions of the available agents and tools by name, a name describes itself if empty.
        measure: Similarity of a description to the name of a missing reference.
        min_similarity: A reference is only replaced by an available one of a higher similarity,
            others stay and fail validation.

    Examples:
        >>> import asyncio
        >>> from exar.experience import Step, Trajectory
        >>> substitute = Substitute(available={"author": "writer of drafts", "critic": "reviewer"})
        >>> trajectory = Trajectory(steps=(Step(kind="agent", name="writer"), Step(kind="agent", name="critic")))
        >>> [step.name for step in asyncio.run(substitute(trajectory, "task", "task")).steps]
        ['author', 'critic']
    """

    available: Mapping[str, str]
    measure: Measure = word_coverage
    min_similarity: float = 0.0

    async def __call__[S: Solution](
        self, solution: S, past: object, task: object, /
    ) -> S:
        missing = [name for name in solution.references if name not in self.available]

        if not missing or not self.available:
            return solution

        descriptions = {name: text or name for name, text in self.available.items()}
        renames: dict[str, str] = {}

        for name in missing:
            best, similarity = max(
                (
                    (key, float(self.measure(description, name)))
                    for key, description in descriptions.items()
                ),
                key=itemgetter(1),
            )

            if similarity > self.min_similarity:
                renames[name] = best

        return solution.rename(renames) if renames else solution
