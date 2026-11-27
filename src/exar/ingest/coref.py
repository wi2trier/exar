"""Merging chunks whose meaning depends on the chunk before them, as a `Judge` decides.

A chunk that starts with a reference back, such as "It" or "This method", reads poorly alone.
`CorefMerge` asks a judge about every two neighboring chunks at once and joins the runs it finds dependent,
as far as they fit, so each chunk is understood on its own.
Every question is an invocation of the judge (`exar.agents.call`), traced like any other.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from ..agents import call
from ..helpers import bounded_gather
from ..text import tagged
from ..typing import Judge, Question
from .chunkers import annotate
from .host import RUNTIME, AsyncChunker
from .model import Chunk

__all__ = ["COREF_INSTRUCTIONS", "CorefMerge"]

COREF_INSTRUCTIONS = (
    "Does the current passage refer back to something only the previous passage names, "
    "such as a pronoun or a phrase like 'this method', so it cannot be understood without it?"
)
"""The question a judge is asked about two neighboring chunks."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CorefMerge:
    """Chunks a document with a chunker, then joins each chunk to the one before it where the judge finds it dependent.

    Every two neighboring chunks that fit in `size` tokens together are judged at once.
    A dependent chunk joins the one before it into one that covers the span of both,
    with the prefix of the first, such as its headings or the header of its table, and the suffix of the last, such as its overlap.
    The tokens of a joined chunk are the sum of its parts.

    Args:
        chunker: Splits the document, best into chunks well below `size`, so there is room to join them.
        judge: Answers whether a chunk depends on the one before it with a probability.
        size: Most tokens of a joined chunk.
        threshold: Probability from which a chunk counts as dependent.
        concurrency: Questions asked at once.
        instructions: The question asked about two neighboring chunks.

    Examples:
        >>> import asyncio
        >>> from exar.ingest import Offloaded
        >>> from exar.typing import Question
        >>> async def pronoun(question: Question) -> float:
        ...     return float("<current>\\nIt " in question.text)
        >>> text = "Paris is large.\\n\\nIt is old.\\n\\nRome is old."
        >>> paragraphs = lambda text: annotate(text, [Chunk(text=p, start=text.index(p), end=text.index(p) + len(p), tokens=len(p.split())) for p in text.split("\\n\\n")])
        >>> merge = CorefMerge(chunker=Offloaded(paragraphs), judge=pronoun, size=64)
        >>> [chunk.text for chunk in asyncio.run(merge(text))]
        ['Paris is large.\\n\\nIt is old.', 'Rome is old.']
    """

    chunker: AsyncChunker
    judge: Judge
    size: int
    threshold: float = 0.5
    concurrency: Annotated[int, RUNTIME] = 8
    instructions: str = COREF_INSTRUCTIONS

    async def _dependent(self, pair: tuple[Chunk, Chunk]) -> bool:
        previous, current = pair
        question = Question(
            instructions=self.instructions,
            text=f"{tagged('previous', previous.text)}\n{tagged('current', current.text)}",
        )

        return await call(self.judge, question, name="judge") >= self.threshold

    async def __call__(self, text: str, /) -> list[Chunk]:
        chunks: Sequence[Chunk] = await self.chunker(text)
        candidates = [
            index
            for index in range(1, len(chunks))
            if chunks[index - 1].tokens + chunks[index].tokens <= self.size
        ]
        verdicts = await bounded_gather(
            [(chunks[index - 1], chunks[index]) for index in candidates],
            self._dependent,
            limit=self.concurrency,
        )
        dependent = {
            index
            for index, verdict in zip(candidates, verdicts, strict=True)
            if verdict
        }
        merged: list[Chunk] = []

        for index, chunk in enumerate(chunks):
            last = merged[-1] if merged else None

            if (
                last is not None
                and index in dependent
                and last.tokens + chunk.tokens <= self.size
            ):
                merged[-1] = Chunk(
                    text=last.prefix + text[last.start : chunk.end] + chunk.suffix,
                    start=last.start,
                    end=chunk.end,
                    tokens=last.tokens + chunk.tokens,
                    hierarchy=last.hierarchy,
                    prefix=last.prefix,
                    suffix=chunk.suffix,
                )
            else:
                merged.append(chunk)

        return annotate(text, merged)
