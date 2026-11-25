from collections.abc import Collection

import pytest
from cbrkit.indexable import InMemory
from cbrkit.retrieve import Retriever
from cbrkit.sim.strings import Levenshtein
from cbrkit.typing import Batches, Results

from exar.cbrkit import IndexMemory, StorageMemory
from exar.errors import Conflict
from exar.memory import InMemoryMemory
from exar.typing import Add, Changes, Remove


async def test_storage_memory() -> None:
    memory = StorageMemory[int, str](
        storage=InMemory({0: "retrieve", 1: "reuse"}),
        retriever=lambda limit: [Retriever(measure=Levenshtein(), limit=limit)],
        conversion=str,
        select=lambda keys: InMemory({0: "retrieve"}).get_index(),
    )
    await memory.write(Changes(items=[Add(value="retain", reason="new")]))

    assert [hit.key for hit in await memory.search("retained", limit=1)] == [2]
    assert await memory.read([0, 2]) == {0: "retrieve"}


async def test_index_memory() -> None:
    texts = {0: "retrieve", 1: "reuse", 2: "retain"}

    def index(batches: Batches[int, str]) -> Results[int, str, float]:
        return Retriever(measure=Levenshtein())([(texts, q) for _, q in batches])

    memory = IndexMemory[int, tuple[str, int]](
        storage=InMemory({0: ("retrieve", 0), 1: ("reuse", 1)}),
        retriever=lambda limit: [index],
    )

    assert [hit.value for hit in await memory.search("retained", limit=2)] == [
        ("retrieve", 0)
    ]


async def test_index_memory_fuses_the_queries_in_one_batch_before_the_limit() -> None:
    rankings = {"q1": [0, 1, 2], "q2": [3, 4, 2]}
    calls: list[list[str]] = []

    def index(batches: Batches[int, str]) -> Results[int, str, float]:
        calls.append([q for _, q in batches])

        return [
            (
                {k: str(k) for k in rankings[q]},
                {k: 1 / (i + 1) for i, k in enumerate(rankings[q])},
            )
            for _, q in batches
        ]

    reads: list[list[int]] = []

    async def select(keys: Collection[int]) -> dict[int, str]:
        reads.append(list(keys))

        return {key: str(key) for key in keys}

    memory = IndexMemory[int, str](
        storage=InMemory({}), retriever=lambda limit: [index], select=select
    )

    # 2 ranks third in both rankings, which fusing them before the limit finds.
    assert [hit.key for hit in await memory.search("q1", "q2", limit=1)] == [2]
    assert calls == [["q1", "q2"]] and reads == [[2]]


async def test_storage_memory_inserts_keyed_additions_and_translates_conflicts() -> (
    None
):
    memory = StorageMemory[int, str](
        storage=InMemory({0: "a"}), retriever=lambda limit: [], conversion=str
    )
    changes = await memory.write(Changes(items=[Add(key=5, value="b", reason="r")]))

    assert changes.items[0].key == 5 and await memory.read([5]) == {5: "b"}

    with pytest.raises(Conflict):
        await memory.write(Changes(items=[Add(key=0, value="b", reason="r")]))

    with pytest.raises(Conflict):
        await memory.write(
            Changes(items=[Remove(key=0, reason="r")]), expected={0: "x"}
        )


async def test_in_memory_memory_scores_by_a_cbrkit_measure() -> None:
    memory = InMemoryMemory[int, str](
        entries={0: "retrieve", 1: "retain"}, measure=Levenshtein()
    )

    assert [hit.key for hit in await memory.search("retained", limit=1)] == [1]
