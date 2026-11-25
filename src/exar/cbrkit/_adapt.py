"""cbrkit adaptations of split cases as adaptations of reused solutions."""

from dataclasses import dataclass

from cbrkit.helpers import to_thread, unbatchify
from cbrkit.model import Case
from cbrkit.typing import AnyAdaptationFunc

from ..typing import Solution

__all__ = ["Adapt"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Adapt[I, S: Solution]:
    """A cbrkit adaptation of split cases of tasks `I` and solutions `S` as an `exar.typing.Adaptation`.

    The case is the past task with its solution and the query is the new task,
    such as for a `cbrkit.adapt.llm.Rewrite` with the instructions `exar.experience.REWRITE`.
    The adaptation runs in a worker thread, from which it awaits language models on the event loop of the caller.

    Args:
        func: Adapts the solution of a case to a query, several chained with `cbrkit.adapt.generic.Pipe`.
    """

    func: AnyAdaptationFunc[Case[I, S]]

    async def __call__(self, solution: S, past: I, task: I, /) -> S:
        """The solution adapted to the task.

        Raises:
            ValueError: If the adaptation returns a case without a solution.
        """
        adapted = await to_thread(
            unbatchify(self.func),
            Case(problem=past, solution=solution),
            Case(problem=task),
        )

        if adapted.solution is None:
            raise ValueError("The adaptation returned a case without a solution")

        return adapted.solution
