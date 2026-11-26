"""Building parsers and chunkers from the options of an operator, and running them off the event loop.

A host, such as a server, builds every component with `configure`
from the facts of its `Host` the component has a field for, then the options of its operator,
which the pydantic annotations of the component validate,
and `fingerprint` names what a component was configured as, such as the chunker that made an index.
`Offloaded` runs a synchronous component in an executor, such as an `exar.ingest.ProcessPool`,
so it serves as an `AsyncParser` or `AsyncChunker`, and `offloaded` does both steps at once.
"""

import asyncio
import inspect
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Literal, Protocol

from pydantic import TypeAdapter

from ..memory import digest
from .model import Chunk, Parsed

__all__ = [
    "AsyncChunker",
    "AsyncParser",
    "Device",
    "Host",
    "Offloaded",
    "SyncComponent",
    "configure",
    "fingerprint",
    "offloaded",
]

type Device = Literal["cpu", "cuda", "mps"]
"""Where the models of a component run."""


class SyncComponent[A, R](Protocol):
    """A synchronous component, such as a `Parser` or a `Chunker`."""

    def __call__(self, arg: A, /) -> R: ...


class AsyncParser(Protocol):
    """Converts a source file to Markdown off the event loop."""

    async def __call__(self, source: Path, /) -> Parsed: ...


class AsyncChunker(Protocol):
    """Chunks a document off the event loop."""

    async def __call__(self, text: str, /) -> Sequence[Chunk]: ...


@dataclass(frozen=True, slots=True)
class Offloaded[A, R]:
    """A synchronous component run off the event loop, in an executor that may pickle it.

    Args:
        func: The component, such as a parser or chunker.
        executor: Where it runs, the default executor of the event loop if None.
    """

    func: Callable[[A], R]
    executor: Executor | None = None

    async def __call__(self, arg: A, /) -> R:
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, self.func, arg
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Host:
    """What a host tells every component, each one taking the facts it has a field for.

    Plain data apart from the executor, so the facts reach worker processes.
    """

    executor: Executor | None = None
    """Where the synchronous components run, see `Offloaded`."""

    threads: int = 1
    batch_size: int = 8
    max_image_pixels: int | None = None
    ocr_languages: tuple[str, ...] = ("eng",)
    skip_ocr_with_text: bool = True
    extract_assets: bool = True
    classify_assets: bool = False
    device: Device = "cpu"


_HOST = Host()


def configure[C](
    cls: type[C],
    options: Mapping[str, object],
    host: Host = _HOST,
    /,
    **defaults: object,
) -> C:
    """Builds a component from the host facts it has a field for, its defaults, then the options of the operator.

    Raises:
        ValueError: For an option the component has no field for, which would otherwise be ignored unseen.
        ValidationError: For an option value the field refuses.

    Examples:
        >>> from exar.ingest import TokenChunker, Whole
        >>> configure(TokenChunker, {"size": "512"})
        TokenChunker(overlap=0, tokenizer='character', size=512)
        >>> configure(Whole, {"size": 1})
        Traceback (most recent call last):
        ValueError: Whole takes no option size
    """
    names = inspect.signature(cls).parameters.keys()

    if unknown := sorted(options.keys() - names):
        raise ValueError(
            f"{getattr(cls, '__name__', cls)} takes no option {', '.join(unknown)}"
        )

    facts = {
        name: getattr(host, name)
        for name in (item.name for item in fields(host))
        if name in names
    }

    return TypeAdapter[C](cls).validate_python(facts | defaults | dict(options))


def offloaded[A, R](
    cls: type[SyncComponent[A, R]], /, **defaults: object
) -> Callable[[Mapping[str, object], Host], Offloaded[A, R]]:
    """Builds a synchronous component with `configure` and runs it on the executor of the host.

    Examples:
        >>> import asyncio
        >>> from exar.ingest import Whole
        >>> chunk = offloaded(Whole)({}, Host())
        >>> [part.text for part in asyncio.run(chunk("some text"))]
        ['some text']
    """

    def build(options: Mapping[str, object], host: Host) -> Offloaded[A, R]:
        return Offloaded(configure(cls, options, host, **defaults), host.executor)

    return build


@cache
def _adapter[C](cls: type[C], /) -> TypeAdapter[C]:
    return TypeAdapter(cls)


def fingerprint(component: object, /) -> str:
    """The digest of a configured component by its class and options, such as the chunker an index was made with.

    It covers the options alone, so a host that changed the code of a component reindexes explicitly.
    An `Offloaded` component is the component it runs.

    Examples:
        >>> from exar.ingest import TokenChunker
        >>> fingerprint(TokenChunker(size=512)) == fingerprint(offloaded(TokenChunker)({"size": "512"}, Host()))
        True
        >>> fingerprint(TokenChunker(size=512)) == fingerprint(TokenChunker(size=256))
        False
    """
    if isinstance(component, Offloaded):
        return fingerprint(component.func)

    cls = type(component)
    options = _adapter(cls).dump_json(component).decode()

    return digest(f"{cls.__module__}.{cls.__qualname__} {options}")
