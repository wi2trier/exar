"""Memories over cbrkit storages."""

from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from cbrkit.cycle import run_async
from cbrkit.typing import AnyCbrFunc, Casebase, ConversionFunc, Float, Storage

from ..typing import Changes, Hit
from ._convert import from_changes, to_changes, translated

__all__ = ["IndexMemory", "StorageMemory"]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Stored[K, V]:
    """Reads, keys, and writes of a cbrkit storage, which its memories share."""

    storage: Storage[K, V]
    select: Callable[[Collection[K]], Awaitable[Casebase[K, V]]] | None = None

    async def read(self, keys: Collection[K], /) -> dict[K, V]:
        """Reads the cases of the keys, leaving out missing ones and those `select` leaves out."""
        if self.select is not None:
            return dict(await self.select(keys))

        cases = await self.storage.get_index()

        return {key: cases[key] for key in keys if key in cases}

    async def keys(self, /) -> list[K]:
        """Keys of the current cases, leaving out those `select` leaves out."""
        keys = (await self.storage.get_index()).keys()

        return list(keys if self.select is None else await self.select(keys))

    async def write(
        self, changes: Changes[K, V], /, *, expected: Mapping[K, V] | None = None
    ) -> Changes[K, V]:
        """Applies changes as one mutation of the storage, which allocates the keys of additions without one.

        Raises:
            ValueError: If the changes are applied already or change a case twice.
            exar.errors.Conflict: If the storage holds other cases than the expected ones
                or a case under the key of an addition.
        """
        with translated():
            applied = await to_changes(changes, expected).apply(self.storage)

        return from_changes(applied, lambda value: value)


@dataclass(frozen=True, slots=True, kw_only=True)
class StorageMemory[K, V](_Stored[K, V]):
    """A cbrkit storage whose current cases cbrkit retrievers score, such as a case base on disk.

    Searching and reading without `select` load the current cases of the storage,
    so storages that search themselves are an `IndexMemory` instead.

    Args:
        storage: The cases, such as `cbrkit.io.FileStorage` or a storage of `cbrkit.indexable`.
        retriever: Builds the steps of a search for its limit, such as a `cbrkit.retrieve.Retriever`.
            The rankings of several queries are fused before the limit applies, see `cbrkit.model.Result.hits`,
            so steps that keep more cases than the limit deepen the fusion.
        conversion: Turns the text of a search into a query case, such as `lambda text: {"task": text}`.
        select: Reads the cases of keys, such as `get_where` of a `cbrkit.typing.FilterableIndex`.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.retrieve import Retriever
        >>> from cbrkit.sim.strings import Levenshtein
        >>> memory = StorageMemory[int, str](
        ...     storage=InMemory({0: "retrieve", 1: "retain"}),
        ...     retriever=lambda limit: [Retriever(measure=Levenshtein(), limit=limit)],
        ...     conversion=str,
        ... )
        >>> [hit.key for hit in asyncio.run(memory.search("retained", limit=1))]
        [1]
    """

    retriever: Callable[[int], Sequence[AnyCbrFunc[K, V, Float]]]
    conversion: ConversionFunc[str, V]

    async def search(self, *queries: str, limit: int = 10) -> list[Hit[K, V]]:
        """Retrieves the cases most similar to the query cases of the texts."""
        with translated():
            result = await run_async(
                self.storage,
                {i: self.conversion(query) for i, query in enumerate(queries)},
                *self.retriever(limit),
            )

        return [
            Hit(key=hit.key, value=hit.value, score=float(hit.score))
            for hit in result.hits(limit)
        ]


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexMemory[K, V](_Stored[K, V]):
    """A cbrkit storage that searches its own text index, such as a pgvector table of typed rows.

    A search queries the index with the text itself and reads its hits as `read` returns them,
    leaving out hits it can't read, so searches and reads see the same values.

    Args:
        storage: The cases that writes change, such as a `cbrkit.indexable.Pgvector` scoped to a tenant.
        retriever: Builds the steps of a search for its limit, run without a casebase,
            such as `PgvectorRetriever(storage=..., where=..., limit=...)` of more candidates than the limit
            and a reranker of `cbrkit.retrieve.rerank`.
            The queries of a search run as one batch, whose rankings are fused before the limit applies,
            so the candidates the steps keep are the depth of the fusion.
        select: Reads the cases of keys, such as `get_where` of a `cbrkit.typing.FilterableIndex`
            with the filter its retriever searches within, so reads see what searches see.

    Examples:
        >>> import asyncio
        >>> from cbrkit.indexable import InMemory
        >>> from cbrkit.retrieve import Retriever
        >>> from cbrkit.sim.strings import Levenshtein
        >>> texts = {0: "retrieve", 1: "retain"}
        >>> def index(batches):
        ...     return Retriever(measure=Levenshtein())([(texts, query) for _, query in batches])
        >>> memory = IndexMemory[int, str](
        ...     storage=InMemory({0: "RETRIEVE", 1: "RETAIN"}), retriever=lambda limit: [index]
        ... )
        >>> [hit.value for hit in asyncio.run(memory.search("retained", limit=1))]
        ['RETAIN']
    """

    retriever: Callable[[int], Sequence[AnyCbrFunc[K, str, Float]]]

    async def search(self, *queries: str, limit: int = 10) -> list[Hit[K, V]]:
        """Searches the index with the texts and reads the hits at once."""
        with translated():
            result = await run_async(
                None, dict(enumerate(queries)), *self.retriever(limit)
            )

        hits = result.hits(limit)
        values = await self.read([hit.key for hit in hits])

        return [
            Hit(key=hit.key, value=values[hit.key], score=float(hit.score))
            for hit in hits
            if hit.key in values
        ]
