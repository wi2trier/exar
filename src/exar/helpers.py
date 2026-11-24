"""Helpers of agentic applications: immutable mappings, names, bounded concurrency, and durable writes.

`frozendict` is an immutable and hashable mapping, `get_name` names functions and objects,
`bounded_gather` and the context manager `bounded_as_completed` run a coroutine function on many items with bounded concurrency,
`shield_to_completion` finishes a critical section through the cancellation of its caller,
and `atomic_write` writes a file durably and atomically,
or `stage_write` and `sync_directory` for a caller that renames many files itself.
"""

import asyncio
import logging
import os
import secrets
import stat
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar, Self, get_args, override

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, SchemaSerializer, core_schema

logger = logging.getLogger(__name__)

__all__ = [
    "atomic_write",
    "bounded_as_completed",
    "bounded_gather",
    "frozendict",
    "get_name",
    "new_id",
    "shield_to_completion",
    "stage_write",
    "sync_directory",
]


class frozendict[K, V](Mapping[K, V]):
    """Immutable, hashable mapping, a stand-in for the builtin of PEP 814.

    Equal to any mapping with the same items and hashable if its values are.
    Pydantic serializes it as a dict in JSON mode, also in fields typed `Any`.

    Examples:
        >>> d = frozendict({"a": 1}, b=2)
        >>> d
        frozendict({'a': 1, 'b': 2})
        >>> d == {"a": 1, "b": 2}, hash(d) == hash(frozendict(b=2, a=1))
        (True, True)
    """

    __slots__ = ("_data", "_hash")

    _data: dict[K, V]
    _hash: int | None
    __pydantic_serializer__: ClassVar[SchemaSerializer]

    def __init__(
        self, data: Mapping[K, V] | Iterable[tuple[K, V]] = (), /, **kwargs: V
    ) -> None:
        self._data = dict(data, **kwargs)
        self._hash = None

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._data

    @override
    def __eq__(self, other: object) -> bool:
        return self._data == other

    @override
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))

        return self._hash

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __or__(self, other: Mapping[K, V]) -> Self:
        return type(self)({**self._data, **other})

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[dict[K, V]]]:
        return type(self), (self._data,)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        key, value = get_args(source) or (Any, Any)
        schema = core_schema.dict_schema(
            handler.generate_schema(key), handler.generate_schema(value)
        )

        return core_schema.no_info_after_validator_function(
            cls,
            schema,
            serialization=core_schema.wrap_serializer_function_ser_schema(
                lambda data, serializer: serializer(dict(data)), schema=schema
            ),
        )


# pydantic serializes values of unknown type, such as cases in fields typed `Any`, with this serializer
frozendict.__pydantic_serializer__ = SchemaSerializer(
    core_schema.is_instance_schema(
        frozendict,
        serialization=core_schema.plain_serializer_function_ser_schema(
            dict, info_arg=False, when_used="json"
        ),
    )
)


def get_name(obj: object) -> str:
    """The name of an object, its `__name__` or else the name of its class, a string as it is, and empty for None.

    Examples:
        >>> import functools
        >>> get_name(len), get_name(functools.partial(len)), get_name(None)
        ('len', 'partial', '')
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    name = getattr(obj, "__name__", None)

    return name if isinstance(name, str) else type(obj).__name__


def _bounded[T, R](
    run: Callable[[T], Awaitable[R]], limit: int
) -> Callable[[T], Coroutine[Any, Any, R]]:
    """Wraps a coroutine function so that at most `limit` of its calls are in flight at once.

    Raises:
        ValueError: If `limit` is less than one, which would never start a call.
    """
    if limit < 1:
        raise ValueError(f"The concurrency limit must be at least 1, not {limit}")

    semaphore = asyncio.Semaphore(limit)

    async def guarded(item: T) -> R:
        async with semaphore:
            return await run(item)

    return guarded


async def _unwind(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Cancels the unfinished tasks and awaits all of them, even through the cancellation of the caller."""
    for task in tasks:
        _ = task.cancel()

    _ = await shield_to_completion(asyncio.gather(*tasks, return_exceptions=True))


@asynccontextmanager
async def bounded_as_completed[T, R](
    items: Iterable[T],
    run: Callable[[T], Awaitable[R]],
    /,
    *,
    limit: int,
) -> AsyncGenerator[AsyncIterator[R]]:
    """Iterates `run(item)` for every item as the calls finish, at most `limit` in flight.

    A context manager, so leaving its block for any reason, be it the end, a break, an error, or a cancellation,
    cancels every unfinished call and awaits it to completion before the block is left,
    so a call owning cleanup, such as a rollback, finishes it first.
    `run` should return its failures rather than raise, so one bad item does not abort the batch.

    Raises:
        ValueError: If `limit` is less than one.

    Examples:
        >>> async def double(value):
        ...     return 2 * value
        >>> async def run():
        ...     async with bounded_as_completed([1, 2], double, limit=1) as results:
        ...         return sorted([x async for x in results])
        >>> asyncio.run(run())
        [2, 4]
    """
    guarded = _bounded(run, limit)
    tasks = [asyncio.ensure_future(guarded(item)) for item in items]

    async def results() -> AsyncIterator[R]:
        for future in asyncio.as_completed(tasks):
            yield await future

    try:
        yield results()
    finally:
        await _unwind(tasks)


async def bounded_gather[T, R](
    items: Iterable[T],
    run: Callable[[T], Awaitable[R]],
    /,
    *,
    limit: int,
) -> list[R]:
    """Returns `run(item)` for every item in input order, at most `limit` in flight.

    The ordered sibling of `bounded_as_completed`, for callers that present results side by side.
    The first exception cancels the remaining calls and propagates,
    so `run` should return its expected failures rather than raise them.

    Examples:
        >>> async def double(value):
        ...     return 2 * value
        >>> asyncio.run(bounded_gather([1, 2, 3], double, limit=2))
        [2, 4, 6]
        >>> async def fail(value):
        ...     raise ValueError(value)
        >>> asyncio.run(bounded_gather([1, 2], fail, limit=2))
        Traceback (most recent call last):
        ...
        ValueError: 1
    """
    guarded = _bounded(run, limit)
    tasks = [asyncio.ensure_future(guarded(item)) for item in items]

    try:
        return await asyncio.gather(*tasks)
    finally:
        await _unwind(tasks)


async def shield_to_completion[T](work: Awaitable[T], /) -> T:
    """Runs an awaitable to completion, surviving the cancellation of the calling task.

    `asyncio.shield` protects the wrapped coroutine, but its awaiter still raises `CancelledError` at once,
    leaving the work running detached.
    For a critical section that mutates shared state under a lock, that is a bug,
    since the lock releases before the work finishes and whatever runs next races the orphaned task.
    This instead keeps awaiting the shielded task through the cancellation of the caller,
    and re-raises a deferred cancellation afterwards, which takes precedence even over an error of the work.
    Such an error is logged and chained as the context of the cancellation, so it is never lost.

    Examples:
        >>> asyncio.run(shield_to_completion(asyncio.sleep(0, result="done")))
        'done'
    """
    task = asyncio.ensure_future(work)
    cancelled: asyncio.CancelledError | None = None

    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.done():
                raise

            # Our own cancellation while the work is still in flight, so wait for it.
            cancelled = exc
        except BaseException:
            # The work raised, but a deferred cancellation of the caller takes precedence.
            if cancelled is not None:
                logger.exception(
                    "A shielded task failed after its caller was cancelled"
                )

                raise cancelled

            raise
        else:
            if cancelled is not None:
                raise cancelled

            return result


def sync_directory(path: Path) -> None:
    """Flushes the entries of a folder to disk, so a rename or removal in it survives a power loss.

    Does nothing where a folder cannot be opened, which is anywhere but posix.
    """
    if os.name != "posix":
        return

    folder = os.open(path, os.O_RDONLY)

    try:
        os.fsync(folder)
    finally:
        os.close(folder)


def stage_write(path: Path, data: bytes | str | Iterable[bytes]) -> Path:
    """Writes a file durably to a unique sibling of `path` and returns it, ready to replace `path` by a rename.

    The sibling has a name of constant length unrelated to the target, so every valid target name works,
    named `.~<16 hex digits>.partial`, so a sweep recognizes one an interruption left behind.
    Creates the missing parent folders, writes text as UTF-8, and streams an iterable of chunks,
    keeps the permissions of the file at `path` whatever the umask, gives a new one the mode `open` would,
    and syncs the sibling to disk, which it removes again if the write fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mode: int | None = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None

    partial = path.with_name(f".~{secrets.token_hex(8)}.partial")
    chunks = (
        [data.encode() if isinstance(data, str) else data]
        if isinstance(data, bytes | str)
        else data
    )

    try:
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)

        with os.fdopen(fd, "wb") as file:
            if mode is not None:
                # `open` applies the umask, which the mode of the replaced file already passed.
                os.fchmod(file.fileno(), mode)

            for chunk in chunks:
                _ = file.write(chunk)

            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return partial


def atomic_write(path: Path, data: bytes | str | Iterable[bytes]) -> None:
    """Writes a file durably and atomically through `stage_write` and a rename, so readers never see a partial file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder, "a", "b.txt")
        ...     atomic_write(path, "hello")
        ...     path.read_text()
        'hello'
    """
    partial = stage_write(path, data)

    try:
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)

    sync_directory(path.parent)


def new_id() -> str:
    """A random identifier.

    Examples:
        >>> len(new_id())
        32
    """
    return uuid.uuid4().hex
