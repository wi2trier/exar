"""The chonkie backends of `exar.ingest.chunkers`, one cached chonkie chunker per configuration and process."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from functools import cache

import chonkie
from chonkie.tokenizer import AutoTokenizer
from chonkie.types import Chunk as ChonkieChunk

from ..helpers import frozendict
from .model import Chunk


def _convert(chunks: list[ChonkieChunk]) -> list[Chunk]:
    return [
        Chunk(
            text=chunk.text,
            start=chunk.start_index,
            end=chunk.end_index,
            tokens=chunk.token_count,
        )
        for chunk in chunks
    ]


@cache
def _chunker(
    cls: Callable[..., chonkie.BaseChunker],
    args: tuple[object, ...],
    options: frozendict[str, object],
) -> chonkie.BaseChunker:
    return cls(*args, **options)


def chunk[**P](
    text: str,
    cls: Callable[P, chonkie.BaseChunker],
    /,
    *args: P.args,
    **options: P.kwargs,
) -> list[Chunk]:
    """Splits a document with the chonkie chunker of a class and its options."""
    return _convert(_chunker(cls, args, frozendict(options)).chunk(text))


@cache
def _refinery(tokenizer: str, size: int) -> chonkie.OverlapRefinery:
    return chonkie.OverlapRefinery(
        tokenizer=tokenizer, context_size=size, method="suffix"
    )


def overlap(
    text: str, chunks: Sequence[Chunk], *, size: int, tokenizer: str
) -> list[Chunk]:
    """Appends to every chunk the first tokens of the span of the next one, keeping its own span.

    The context comes from the span of the next chunk, never its text,
    so a table chunk is followed by the rows of the next one without its repeated header.
    """
    if size == 0:
        return list(chunks)

    spans = [
        ChonkieChunk(
            text=text[item.start : item.end],
            start_index=item.start,
            end_index=item.end,
            token_count=item.tokens,
        )
        for item in chunks
    ]
    refined = _refinery(tokenizer, size).refine(spans)

    return [
        replace(
            item,
            text=item.text + span.text[item.end - item.start :],
            tokens=span.token_count,
        )
        for item, span in zip(chunks, refined, strict=True)
    ]


@cache
def _chef(tokenizer: str) -> tuple[chonkie.MarkdownChef, AutoTokenizer]:
    return chonkie.MarkdownChef(tokenizer=tokenizer), AutoTokenizer(tokenizer)


def _blocks(text: str, chef: chonkie.MarkdownChef) -> Iterator[tuple[int, int, bool]]:
    """The spans of the tables, code blocks, and passages between them that tile *text*, and whether each is a table.

    A block nested in an earlier one, such as a table in a code block, belongs to the earlier one.
    """
    document = chef.parse(text)
    position = 0

    for start, end, table in sorted(
        [
            *((table.start_index, table.end_index, True) for table in document.tables),
            *((code.start_index, code.end_index, False) for code in document.code),
        ]
    ):
        if start < position:
            continue

        yield position, start, False
        yield start, end, table
        position = end

    yield position, len(text), False


def markdown(
    text: str, *, tokenizer: str, size: int, rows: int, min_characters: int
) -> list[Chunk]:
    """Splits Markdown into its text passages, tables, and code blocks, in order of position.

    A passage is all text between tables and code blocks, images included,
    so the chunks tile the document but for the whitespace between them.
    A block of more than *size* tokens is split again,
    a table into chunks of *rows* rows that repeat its header, anything else recursively.
    """
    chef, counter = _chef(tokenizer)
    chunks: list[Chunk] = []

    for start, end, table in _blocks(text, chef):
        block = text[start:end]

        if not block.strip():
            continue

        if (tokens := counter.count_tokens(block)) > size and (
            parts := (
                chunk(block, chonkie.TableChunker, tokenizer="row", chunk_size=rows)
                if table
                else chunk(
                    block,
                    chonkie.RecursiveChunker,
                    tokenizer=tokenizer,
                    chunk_size=size,
                    min_characters_per_chunk=min_characters,
                )
            )
        ):
            chunks.extend(
                Chunk(
                    text=part.text,
                    start=start + part.start,
                    end=start + part.end,
                    tokens=counter.count_tokens(part.text),
                )
                for part in parts
            )
        else:
            # Within the size, or a table too malformed to split by rows.
            chunks.append(Chunk(text=block, start=start, end=end, tokens=tokens))

    return chunks
