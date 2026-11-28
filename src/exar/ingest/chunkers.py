"""Chunkers that split a document into the passages a retriever indexes.

A chunker is a callable from text to `Chunk`s that know their character span and line range,
so a retrieved passage can be traced back to its place in the document.
Chunkers are synchronous and picklable, so the caller decides where they run.
The chonkie backends import chonkie on the first call and cache one instance per configuration and process.
"""

import bisect
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate, repeat
from operator import add
from typing import Annotated, Protocol

from pydantic import Field

from .._lazy import extra_hint
from .model import Chunk

__all__ = [
    "Chunker",
    "MarkdownChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SentenceChunker",
    "TableChunker",
    "TokenChunker",
    "Whole",
    "annotate",
    "overlap",
]


class Chunker(Protocol):
    """Splits a document into chunks, in order of their position."""

    def __call__(self, text: str, /) -> Sequence[Chunk]: ...

    def prepare(self) -> None:
        """Loads every model the first call would download, none unless a chunker overrides it."""
        # A body beyond the docstring, since a protocol method without one is abstract.
        return


def annotate(text: str, chunks: Iterable[Chunk]) -> list[Chunk]:
    """Sets the line range of chunks of a document from their character span.

    Examples:
        >>> chunk = Chunk(text="b\\nc", start=2, end=5, tokens=2)
        >>> [(c.start_line, c.end_line) for c in annotate("a\\nb\\nc", [chunk])]
        [(2, 3)]
    """
    # Each line starts one past the end of the one before, summed without a Python loop.
    starts = list(
        accumulate(map(add, map(len, text.split("\n")), repeat(1)), initial=0)
    )

    return [
        replace(
            chunk,
            start_line=bisect.bisect_right(starts, chunk.start),
            end_line=bisect.bisect_right(starts, max(chunk.end - 1, chunk.start)),
        )
        for chunk in chunks
    ]


def overlap(
    text: str, chunks: Sequence[Chunk], *, size: int, tokenizer: str = "character"
) -> list[Chunk]:
    """Appends to every chunk of a document the first `size` tokens of the next one, keeping its span.

    Chunkers without overlap of their own, such as one of a language model, gain it this way.
    It is applied by chonkie's `OverlapRefinery`, so it needs the `chonkie` extra.
    """
    with extra_hint("overlap", "chonkie"):
        from ._chonkie import overlap

    return overlap(text, chunks, size=size, tokenizer=tokenizer)


# Validated where a host builds a component from options, with `exar.ingest.configure`.
type _Size = Annotated[int, Field(ge=64, le=32768)]
type _Rows = Annotated[int, Field(ge=1, le=32768)]
type _Overlap = Annotated[int, Field(ge=0)]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Overlapping(Chunker, ABC):
    """Splits a document with a chonkie chunker and appends the overlap to every chunk.

    The overlap means the same for every chunker: the first `overlap` tokens of the next chunk by `tokenizer`,
    appended to a chunk that keeps its own span, see `overlap`.
    """

    overlap: _Overlap = 0
    tokenizer: str = "character"

    @abstractmethod
    def _split(self, text: str, /) -> list[Chunk]:
        """The chunks of chonkie without overlap."""

    def __call__(self, text: str, /) -> list[Chunk]:
        chunks = self._split(text)

        return annotate(
            text, overlap(text, chunks, size=self.overlap, tokenizer=self.tokenizer)
        )

    def prepare(self) -> None:
        """Loads the tokenizer, which downloads one named after a model of a hub."""
        with extra_hint(type(self).__name__, "chonkie"):
            from ._chonkie import load_tokenizer

        _ = load_tokenizer(self.tokenizer)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Sized(_Overlapping, ABC):
    """A chonkie chunker of chunks of at most `size` tokens."""

    size: _Size = 2048


@dataclass(frozen=True, slots=True, kw_only=True)
class Whole(Chunker):
    """Keeps a document as one chunk, such as a generated image description.

    Its token count is the number of words.

    Examples:
        >>> Whole()("one two\\nthree")
        [Chunk(text='one two\\nthree', start=0, end=13, tokens=3, start_line=1, end_line=2, hierarchy=(), prefix='', suffix='')]
    """

    def __call__(self, text: str, /) -> list[Chunk]:
        if not text:
            return []

        return annotate(
            text, [Chunk(text=text, start=0, end=len(text), tokens=len(text.split()))]
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenChunker(_Sized):
    """Splits a document into chunks of a fixed number of tokens, such as tables in CSV.

    Args:
        size: Tokens of a chunk.
        overlap: Tokens of the next chunk appended to every chunk, which keeps its own span.
        tokenizer: The chonkie tokenizer, such as `"character"`, `"gpt2"`, or `"cl100k_base"`.
    """

    def _split(self, text: str, /) -> list[Chunk]:
        with extra_hint("TokenChunker", "chonkie"):
            import chonkie

            from ._chonkie import chunk

        return chunk(
            text, chonkie.TokenChunker, tokenizer=self.tokenizer, chunk_size=self.size
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RecursiveChunker(_Sized):
    """Splits a document by paragraphs, then sentences, then words, until chunks fit, such as Markdown.

    Args:
        size: Most tokens of a chunk.
        overlap: Tokens of the next chunk appended to every chunk, which keeps its own span.
        tokenizer: The chonkie tokenizer.
        min_characters: Fewest characters of a chunk.
    """

    min_characters: int = 24

    def _split(self, text: str, /) -> list[Chunk]:
        with extra_hint("RecursiveChunker", "chonkie"):
            import chonkie

            from ._chonkie import chunk

        return chunk(
            text,
            chonkie.RecursiveChunker,
            tokenizer=self.tokenizer,
            chunk_size=self.size,
            min_characters_per_chunk=self.min_characters,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SentenceChunker(_Sized):
    """Splits a document into chunks of whole sentences, such as prose.

    Args:
        size: Most tokens of a chunk.
        overlap: Tokens of the next chunk appended to every chunk, which keeps its own span.
        tokenizer: The chonkie tokenizer.
        min_sentences: Fewest sentences of a chunk.
        min_characters: Fewest characters of a sentence.
    """

    min_sentences: int = 1
    min_characters: int = 12

    def _split(self, text: str, /) -> list[Chunk]:
        with extra_hint("SentenceChunker", "chonkie"):
            import chonkie

            from ._chonkie import chunk

        return chunk(
            text,
            chonkie.SentenceChunker,
            tokenizer=self.tokenizer,
            chunk_size=self.size,
            min_sentences_per_chunk=self.min_sentences,
            min_characters_per_sentence=self.min_characters,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticChunker(_Sized):
    """Splits a document where the similarity of consecutive sentences drops, keeping topics together.

    Args:
        size: Most tokens of a chunk, counted by the tokenizer of the embedding model.
        overlap: Tokens of the next chunk appended to every chunk, which keeps its own span.
        tokenizer: The chonkie tokenizer of the overlap.
        model: The embedding model of chonkie, a Model2Vec model by default.
        threshold: Similarity below which a new chunk starts.
        window: Sentences compared at once.
        min_sentences: Fewest sentences of a chunk.
        min_characters: Fewest characters of a sentence.
    """

    model: str = "minishlab/potion-base-32M"
    threshold: float = 0.8
    window: int = 3
    min_sentences: int = 1
    min_characters: int = 24

    def _split(self, text: str, /) -> list[Chunk]:
        with extra_hint("SemanticChunker", "semantic"):
            import chonkie

            from ._chonkie import chunk

        return chunk(
            text,
            chonkie.SemanticChunker,
            embedding_model=self.model,
            chunk_size=self.size,
            threshold=self.threshold,
            similarity_window=self.window,
            min_sentences_per_chunk=self.min_sentences,
            min_characters_per_sentence=self.min_characters,
        )

    def prepare(self) -> None:
        """Loads the tokenizer of the overlap and the embedding model."""
        with extra_hint("SemanticChunker", "semantic"):
            from ._chonkie import load_embeddings, load_tokenizer

        _ = load_tokenizer(self.tokenizer)
        load_embeddings(self.model)


@dataclass(frozen=True, slots=True, kw_only=True)
class TableChunker(_Sized):
    """Splits a Markdown or HTML table into chunks of rows, such as a converted spreadsheet.

    Each chunk repeats the header of the table, so its text is not its span, which covers only its rows.
    A document that is no table stays one chunk.

    Args:
        size: Rows of a chunk, or most tokens of a chunk with another tokenizer than `"row"`.
        overlap: Rows, or tokens, of the next chunk appended to every chunk, which keeps its own span.
        tokenizer: The chonkie tokenizer, `"row"` to count rows.
    """

    size: _Rows = 3
    tokenizer: str = "row"

    def _split(self, text: str, /) -> list[Chunk]:
        with extra_hint("TableChunker", "chonkie"):
            import chonkie

            from ._chonkie import chunk

        return chunk(
            text, chonkie.TableChunker, tokenizer=self.tokenizer, chunk_size=self.size
        )

    def __call__(self, text: str, /) -> list[Chunk]:
        return super().__call__(text) or Whole()(text)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarkdownChunker(_Sized):
    """Splits Markdown into chunks of whole sections, or of their blocks where a section is too long.

    A section that fits in `size` tokens is one chunk, else its blocks, such as paragraphs, list items, code blocks,
    and subsections, are joined while they fit, a heading or list marker kept with the block after it.
    The front matter is one chunk of its own, and each chunk knows the headings of the sections it lies in as its hierarchy.
    A table of more than `size` tokens is split like `TableChunker` into chunks of `rows` rows that repeat its header,
    a longer block like `RecursiveChunker`, its first chunk keeping the heading before it.
    With `context`, every chunk starts with the lines of the headings above it that it lacks, as the document writes them,
    so it reads on its own, while its span stays that of its passage.

    Args:
        size: Most tokens of a chunk, but for a table row of more.
        overlap: Tokens of the next chunk appended to every chunk, which keeps its own span.
        tokenizer: The chonkie tokenizer.
        rows: Rows of a chunk of a table that is split, as many as fit in `size` tokens if None.
        min_characters: Fewest characters of a chunk of a passage that is split.
        context: Whether to start every chunk with the headings of its hierarchy,
            which take at most an eighth of `size`, the outermost dropped first.

    Examples:
        >>> text = "# A\\n\\n## B\\n\\n" + "Words of a long passage. " * 24
        >>> [chunk.text.partition("\\n\\n")[0] for chunk in MarkdownChunker(size=256, context=True)(text)]
        ['# A', '# A\\n## B', '# A\\n## B', '# A\\n## B']
    """

    rows: _Rows | None = 3
    min_characters: int = 24
    context: bool = False

    def _split(self, text: str, /) -> list[Chunk]:
        with extra_hint("MarkdownChunker", "chonkie"):
            from ._chonkie import markdown

        return markdown(
            text,
            tokenizer=self.tokenizer,
            size=self.size,
            rows=self.rows,
            min_characters=self.min_characters,
            context=self.context,
        )
