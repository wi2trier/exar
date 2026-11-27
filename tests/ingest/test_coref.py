from exar.ingest import Chunk, CorefMerge, Offloaded, annotate
from exar.typing import Question

TEXT = "Paris is a city.\n\nIt is old.\n\nIt is large.\n\nRome is old.\n\nIt is huge."


def _paragraphs(text: str) -> list[Chunk]:
    """Chunks of the paragraphs of *text*, each starting with a heading that is not in its span."""
    chunks: list[Chunk] = []
    start = 0

    for paragraph in text.split("\n\n"):
        start = text.index(paragraph, start)
        chunks.append(
            Chunk(
                text=f"# A\n\n{paragraph}",
                start=start,
                end=start + len(paragraph),
                tokens=len(paragraph.split()),
                prefix="# A\n\n",
            )
        )
        start += len(paragraph)

    return annotate(text, chunks)


async def _pronoun(question: Question) -> float:
    return float("<current>\n# A\n\nIt " in question.text)


async def test_coref_joins_dependent_runs_up_to_the_size_keeping_the_prefix() -> None:
    merged = await CorefMerge(chunker=Offloaded(_paragraphs), judge=_pronoun, size=8)(
        TEXT
    )

    assert [chunk.text for chunk in merged] == [
        "# A\n\nParis is a city.\n\nIt is old.",
        "# A\n\nIt is large.",
        "# A\n\nRome is old.\n\nIt is huge.",
    ]
    assert [(chunk.start_line, chunk.end_line, chunk.tokens) for chunk in merged] == [
        (1, 3, 7),
        (5, 5, 3),
        (7, 9, 6),
    ]
