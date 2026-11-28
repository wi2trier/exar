from collections.abc import Sequence

import pytest

from exar.ingest import Chunk, Offloaded, Whole, index_document
from exar.memory import digest

whole = Offloaded(Whole())


class Store:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, int | str, str]] = []

    async def replace(self, key: str, chunks: Sequence[Chunk], /) -> None:
        self.events.append(("replace", key, len(chunks), ""))

    async def stamp(self, key: str, digest: str, chunker: str, /) -> None:
        self.events.append(("stamp", key, digest, chunker))


async def test_index_document_stamps_the_digest_last() -> None:
    store = Store()
    result = await index_document("doc", "one two", whole, store, fingerprint="whole")

    assert (result.digest, result.chunker) == (digest("one two"), "whole")
    assert store.events == [
        ("replace", "doc", 1, ""),
        ("stamp", "doc", digest("one two"), "whole"),
    ]


async def test_index_document_does_not_stamp_a_failed_store() -> None:
    class Broken(Store):
        async def replace(self, key: str, chunks: Sequence[Chunk], /) -> None:
            raise OSError("down")

    store = Broken()

    with pytest.raises(OSError):
        await index_document("doc", "text", whole, store, fingerprint="whole")

    assert store.events == []
