"""Entries kept in process."""

from collections.abc import Callable, Collection, Mapping
from dataclasses import InitVar, dataclass, field, replace
from threading import Lock
from typing import Literal, cast
from uuid import uuid7

from ..errors import Conflict
from ..helpers import frozendict
from ..schema import PythonOnly, serializable
from ..typing import Add, Changes, Hit, Measure, Remove, Update
from ._common import _search, word_coverage

__all__ = ["InMemoryMemory", "next_keys"]


@dataclass(slots=True)
class _State[K, V]:
    """The current snapshot of an `InMemoryMemory` and the lock of its writers."""

    entries: frozendict[K, V]
    lock: Lock = field(default_factory=Lock)


def next_keys[K](taken: Collection[K], count: int, /) -> list[K]:
    """The keys of new entries: the integers after the largest key if every key is an integer, new UUIDv7 strings otherwise.

    An empty memory starts at the integer 0, so a memory of other keys passes its own `new_keys`.
    It is the rule of `cbrkit.helpers.next_keys`, which exar does not import, since cbrkit is an extra.

    Examples:
        >>> next_keys([0, 4, 2], 2), next_keys([], 1)
        ([5, 6], [0])
        >>> [len(key) for key in next_keys(["a"], 1)]
        [36]
    """
    ints = [key for key in taken if isinstance(key, int)]

    if len(ints) == len(taken):
        start = max(ints, default=-1) + 1

        return cast(list[K], list(range(start, start + count)))

    return cast(list[K], [str(uuid7()) for _ in range(count)])


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class InMemoryMemory[K, V]:
    """Entries kept in process, searched by the similarity of their text to the queries.

    Every write replaces the immutable snapshot that searches and reads see,
    and its lock guards it against writers on other threads.

    Args:
        entries: The initial entries, which serialized forms leave out.
        text: Renders an entry for the measure, `str` by default.
        measure: Similarity of the text of an entry to a query, the share of the words of the query by default.
        new_keys: The distinct keys of a number of new entries outside the keys taken so far, `next_keys` by default.

    Examples:
        >>> import asyncio
        >>> memory = InMemoryMemory[int, str](entries={0: "retrieve"})
        >>> changes = asyncio.run(memory.write(Changes(items=[Add(value="retain", reason="new")])))
        >>> changes.items[0].key, [hit.key for hit in asyncio.run(memory.search("retain"))]
        (1, [1, 0])
    """

    kind: Literal["in_memory"] = field(default="in_memory", repr=False)
    entries: InitVar[Mapping[K, V] | None] = None
    text: PythonOnly[Callable[[V], str]] = str
    measure: PythonOnly[Measure] = word_coverage
    new_keys: PythonOnly[Callable[[Collection[K], int], list[K]]] = next_keys
    _state: _State[K, V] = field(init=False, repr=False, compare=False)

    def __post_init__(self, entries: Mapping[K, V] | None) -> None:
        object.__setattr__(self, "_state", _State(frozendict(entries or {})))

    async def search(self, *queries: str, limit: int = 10) -> list[Hit[K, V]]:
        """Finds the entries whose text is most similar to the queries."""
        return await _search(
            self._state.entries, queries, self.text, self.measure, limit
        )

    async def read(self, keys: Collection[K], /) -> dict[K, V]:
        """Reads the entries of the keys, leaving out missing ones."""
        entries = self._state.entries

        return {key: entries[key] for key in keys if key in entries}

    async def keys(self, /) -> list[K]:
        """Keys of all entries."""
        return list(self._state.entries)

    async def write(
        self, changes: Changes[K, V], /, *, expected: Mapping[K, V] | None = None
    ) -> Changes[K, V]:
        """Applies changes as one mutation, new entries without a key get one of `new_keys`.

        Raises:
            ValueError: If the changes are applied already or change an entry twice.
            exar.errors.Conflict: If an addition exists already or the expected entries changed, then nothing is written.
        """
        if changes.applied:
            raise ValueError("The changes are applied already")

        keys = [change.key for change in changes.items if change.key is not None]

        if len(set(keys)) != len(keys):
            raise ValueError("The changes change an entry twice")

        with self._state.lock:
            current = self._state.entries

            if any(
                key not in current or current[key] != value
                for key, value in (expected or {}).items()
            ):
                raise Conflict("The entries changed since the changes were planned")

            if added := [
                change.key
                for change in changes.items
                if isinstance(change, Add) and change.key in current
            ]:
                raise Conflict(f"{added[0]!r} exists already")

            count = sum(change.key is None for change in changes.items)
            fresh = iter(self.new_keys({*current, *keys}, count) if count else ())
            entries = dict(current)
            items: list[Add[K, V] | Update[K, V] | Remove[K]] = []

            for change in changes.items:
                if isinstance(change, Remove):
                    _ = entries.pop(change.key, None)
                    items.append(change)
                    continue

                key = next(fresh) if change.key is None else change.key
                entries[key] = change.value
                items.append(replace(change, key=key))

            self._state.entries = frozendict(entries)

        return replace(changes, items=items, applied=True)
