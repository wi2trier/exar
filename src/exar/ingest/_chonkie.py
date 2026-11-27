"""The chonkie backends of `exar.ingest.chunkers`, one cached chonkie chunker per configuration and process."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache

import chonkie
from chonkie.tokenizer import AutoTokenizer
from chonkie.types import Chunk as ChonkieChunk

from ..helpers import frozendict
from ._markdown import pieces
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
def _counter(tokenizer: str) -> AutoTokenizer:
    return AutoTokenizer(tokenizer)


def markdown(
    text: str, *, tokenizer: str, size: int, rows: int, min_characters: int
) -> list[Chunk]:
    """Splits Markdown into chunks of whole sections, or of their blocks, in order of position, each with the headings above it as its hierarchy.

    The front matter is one chunk of its own, and a block of more than *size* tokens is split,
    a table into chunks of *rows* rows that repeat its header, anything else recursively.
    """
    counter = _counter(tokenizer)

    def split(block: str, limit: int, table: bool) -> list[Chunk]:
        if table:
            # Counted as tokens, not as the rows that split them.
            parts = [
                replace(part, tokens=counter.count_tokens(part.text))
                for part in chunk(block, chonkie.TableChunker, tokenizer="row", chunk_size=rows)
            ]
        else:
            parts = chunk(
                block,
                chonkie.RecursiveChunker,
                tokenizer=tokenizer,
                chunk_size=limit,
                min_characters_per_chunk=min_characters,
            )

        # A table too malformed to split by rows stays whole.
        return parts or [Chunk(text=block, start=0, end=len(block), tokens=counter.count_tokens(block))]

    return [
        Chunk(
            text=text[piece.start : piece.end] if piece.text is None else piece.text,
            start=piece.start,
            end=piece.end,
            tokens=piece.tokens,
            hierarchy=piece.hierarchy,
        )
        for piece in pieces(text, size=size, count=counter.count_tokens, split=split)
    ]
