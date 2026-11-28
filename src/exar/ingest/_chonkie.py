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


def _convert(text: str, chunks: list[ChonkieChunk]) -> list[Chunk]:
    """The chunks of chonkie, the text of a table chunk the header of its table before its span of rows."""
    return [
        Chunk(
            text=chunk.text,
            start=chunk.start_index,
            end=chunk.end_index,
            tokens=chunk.token_count,
            prefix=chunk.text.removesuffix(text[chunk.start_index : chunk.end_index]),
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
    return _convert(text, _chunker(cls, args, frozendict(options)).chunk(text))


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
            text=item.text + (suffix := span.text[item.end - item.start :]),
            suffix=item.suffix + suffix,
            tokens=span.token_count,
        )
        for item, span in zip(chunks, refined, strict=True)
    ]


@cache
def load_tokenizer(name: str) -> AutoTokenizer:
    """The tokenizer of chonkie *name* names, cached per process, which loads one of a model hub on the first call."""
    return AutoTokenizer(name)


def load_embeddings(model: str) -> None:
    """Loads the embedding model of chonkie *model* names, as its semantic chunker does."""
    _ = chonkie.AutoEmbeddings.get_embeddings(model)


def _context(
    lines: tuple[str, ...], count: Callable[[str], int], limit: int
) -> tuple[str, int]:
    """The heading *lines*, as many innermost ones as fit in *limit* tokens, and their tokens."""
    for skip in range(len(lines)):
        prefix = "\n".join(lines[skip:]) + "\n\n"

        if (tokens := count(prefix)) <= limit:
            return prefix, tokens

    return "", 0


def markdown(
    text: str,
    *,
    tokenizer: str,
    size: int,
    rows: int | None,
    min_characters: int,
    context: bool,
) -> list[Chunk]:
    """Splits Markdown into chunks of whole sections, or of their blocks, in order of position, each with the headings above it as its hierarchy.

    The front matter is one chunk of its own, and a block of more than *size* tokens is split,
    a table into chunks of *rows* rows, or of the rows that fit if None, that repeat its header, anything else recursively.
    With *context*, an eighth of *size* is left for the headings every chunk starts with.
    """
    counter = load_tokenizer(tokenizer)
    reserve = size // 8 if context else 0

    def split(block: str, limit: int, table: bool) -> list[Chunk]:
        if table:
            # Counted as tokens, not as the rows that split them.
            parts = [
                replace(part, tokens=counter.count_tokens(part.text))
                for part in chunk(
                    block,
                    chonkie.TableChunker,
                    tokenizer="row" if rows is not None else tokenizer,
                    chunk_size=rows if rows is not None else limit,
                )
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
        return parts or [
            Chunk(
                text=block, start=0, end=len(block), tokens=counter.count_tokens(block)
            )
        ]

    # The headings a piece lacks, shared by the pieces of a section, so each chain is counted once.
    contexts: dict[tuple[str, ...], tuple[str, int]] = {}
    result: list[Chunk] = []

    for piece in pieces(
        text, size=size - reserve, count=counter.count_tokens, split=split
    ):
        lines = piece.context if context else ()

        if lines not in contexts:
            contexts[lines] = _context(lines, counter.count_tokens, reserve)

        heading, tokens = contexts[lines]
        span = text[piece.start : piece.end]
        # A part of a table is its header before its rows, which are its span.
        prefix = heading + ("" if piece.text is None else piece.text.removesuffix(span))
        result.append(
            Chunk(
                text=prefix + span,
                start=piece.start,
                end=piece.end,
                tokens=piece.tokens + tokens,
                hierarchy=piece.hierarchy,
                prefix=prefix,
            )
        )

    return result
