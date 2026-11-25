"""Ingesting a parsed document into an index of chunks, with its digest and the fingerprint of its chunker stamped last.

Chunking and indexing are separate steps, often separate transactions,
so the digest of the indexed text and the fingerprint of the chunker, see `exar.ingest.fingerprint`,
are recorded only once the chunks are stored.
A document without a stamp, or with the stamp of other text or of a chunker configured otherwise,
is then the one mark of an index that needs work,
whether the text or the chunker changed or an earlier ingestion was torn by a crash.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..memory import digest
from .host import AsyncChunker
from .model import Chunk

__all__ = ["ChunkStore", "Ingested", "index_document"]


class ChunkStore[K](Protocol):
    """Where the chunks of documents are indexed, such as the chunk table of a database."""

    async def replace(self, key: K, chunks: Sequence[Chunk], /) -> None:
        """Replaces the chunks of a document and clears its stamp."""
        ...

    async def stamp(self, key: K, digest: str, chunker: str, /) -> None:
        """Records the digest of the text the chunks of a document were made from and the fingerprint of the chunker that made them."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Ingested:
    """The chunks a document was indexed as, the digest of its text, and the fingerprint of its chunker."""

    chunks: Sequence[Chunk]
    digest: str
    chunker: str


async def index_document[K](
    key: K,
    text: str,
    chunk: AsyncChunker,
    store: ChunkStore[K],
    /,
    *,
    fingerprint: str,
) -> Ingested:
    """Chunks a document, stores its chunks, and stamps its digest and the fingerprint of the chunker last.

    Args:
        key: The document.
        text: Its content, such as the Markdown of a parsed document.
        chunk: Splits the content, such as a language model,
            or a synchronous `exar.ingest.chunkers.Chunker` run off the event loop with `exar.ingest.Offloaded`.
        store: Indexes the chunks.
        fingerprint: What `chunk` was configured as, such as `exar.ingest.fingerprint` of the chunker it runs.

    Returns:
        The chunks, the digest of the text, and the fingerprint of the chunker.
    """
    chunks = await chunk(text)
    await store.replace(key, chunks)
    result = Ingested(chunks=chunks, digest=digest(text), chunker=fingerprint)
    await store.stamp(key, result.digest, result.chunker)

    return result
