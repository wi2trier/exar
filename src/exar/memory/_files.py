"""Files of mounted folders as documents with frontmatter metadata."""

import asyncio
import contextlib
import threading
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from ..errors import Conflict
from ..helpers import frozendict, shield_to_completion, stage_write, sync_directory
from ..schema import Open, PythonOnly, serializable
from ..typing import (
    Add,
    Changes,
    Decision,
    Hit,
    JsonEntry,
    ListableMemory,
    Measure,
    Remove,
    Update,
)
from ._common import _authorize, _fuse, _search, digest, word_coverage
from ._document import Document
from ._in_memory import InMemoryMemory
from .mounts import Mount, fold, folds_case, resolve, targets

__all__ = ["FileMemory"]

_WRITING = threading.Lock()
"""Serializes the checks and the writes of every file memory, so concurrent changes cannot overwrite each other."""


@dataclass(frozen=True, slots=True)
class _File:
    """A file with its modification time and size, which change with its content."""

    path: Path
    mtime_ns: int
    size: int

    @classmethod
    def stat(cls, path: Path) -> Self:
        stat = path.stat()

        return cls(path=path, mtime_ns=stat.st_mtime_ns, size=stat.st_size)


@dataclass(frozen=True, slots=True)
class _Indexed:
    """What an index holds of a file, its stat is None if it was read back from the index."""

    file: _File | None
    digest: JsonEntry


@dataclass(frozen=True, slots=True)
class _Write:
    """A planned write of a file, a deletion if the document is None, which must not exist if it is new."""

    key: str
    path: Path
    document: Document | None
    new: bool


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class FileMemory:
    """Files of mounted folders, keyed by their address, the target of their mount joined with their relative path.

    Entries that a mount excludes, such as hidden files and folders by default, are left out and refused as keys,
    as are files that are no UTF-8 text and symlinks.
    Searching fuses the results of every mount by reciprocal rank fusion weighted by trust.
    An index is synced before each search: files whose size or modification time changed are read,
    those whose digest of text and metadata changed are written to the index with the digest in their metadata,
    and the entries of deleted files are removed.
    A new process reads the keys and digests back from the index,
    so an unchanged mount is not indexed again and files deleted meanwhile are removed.
    The same mounts serve the file tools of `exar.tools.files`.

    Args:
        mounts: The folders, such as one with the target `/notes` or `~`.
        measure: Scores the documents of mounts without index, the share of the words of the query by default.
        indexes: Searches the documents of the mounts by target, which keeps them in sync by file key,
            such as an `InMemoryMemory` or any other `ListableMemory` of Python code,
            the measure scores every document of a mount without one.
        trust: Weights of the mounts by target when fusing search results, such as 1 for curated and 0.5 for generated knowledge, 1 if missing.
        name: Name of the memory in the actions of writes to mounts whose write mode is `"ask"`,
            which the approver of the run answers (`exar.agents.approve`).

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> memory = FileMemory(mounts=[Mount(source=Path(tempfile.mkdtemp()), target="/notes", write="allow")])
        >>> change = Update(key="/notes/sky.md", value=Document(text="The sky is blue"), reason="seen")
        >>> _ = asyncio.run(memory.write(Changes(items=[change])))
        >>> [hit.key for hit in asyncio.run(memory.search("sky"))]
        ['/notes/sky.md']
    """

    kind: Literal["file"] = field(default="file", repr=False)
    mounts: Sequence[Mount]
    measure: PythonOnly[Measure] = word_coverage
    indexes: Mapping[
        str, Open[InMemoryMemory[str, Document], ListableMemory[str, Document]]
    ] = field(default_factory=dict)
    trust: Mapping[str, float] = field(default_factory=dict)
    name: str = "memory"
    _synced: dict[str, dict[str, _Indexed]] = field(
        init=False, repr=False, compare=False
    )
    _syncing: dict[str, asyncio.Lock] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_synced", {})
        object.__setattr__(self, "_syncing", {})

    def resolve(self, key: str, /) -> tuple[Mount, Path]:
        """The mount and the file of a key.

        Raises:
            ValueError: If the key lies in no mount, leaves its folder, or names an excluded file or folder.
        """
        resolved = resolve(self.mounts, key)

        if resolved is None:
            raise ValueError(
                f"{key!r} lies in no mount of {', '.join(targets(self.mounts))}"
            )

        mount, local = resolved
        found = mount.canonical(local)

        if found is None:
            raise ValueError(f"{key!r} names no file of the mount {mount.target!r}")

        if mount.excludes(local) or mount.excludes(found[0]):
            raise ValueError(
                f"{key!r} names an excluded file of the mount {mount.target!r}"
            )

        return mount, found[1]

    def _mount(self, target: str, /) -> Mount:
        """The mount of a target.

        Raises:
            ValueError: If no mount has the target.
        """
        for mount in self.mounts:
            if mount.target == target:
                return mount

        raise ValueError(f"{target!r} is no mount of {', '.join(targets(self.mounts))}")

    def _files(self, mount: Mount) -> Iterator[tuple[str, Path]]:
        root = mount.root()

        for local, _status in mount.walk(root):
            yield mount.render(local), root / local

    def _scan(self, prefix: str | None) -> dict[str, Path]:
        return {
            key: path
            for mount in self.mounts
            for key, path in self._files(mount)
            if prefix is None or PurePosixPath(key).is_relative_to(prefix)
        }

    async def keys(self, prefix: str | None = None, /) -> list[str]:
        """Keys of all files, or of those below a prefix."""
        return list(await asyncio.to_thread(self._scan, prefix))

    async def read(self, keys: Collection[str], /) -> dict[str, Document]:
        """Reads the documents of the keys, leaving out missing files.

        Raises:
            ValueError: If a key lies in no mount.
        """
        paths = {key: self.resolve(key)[1] for key in keys}

        return await asyncio.to_thread(_load, paths)

    async def write(
        self,
        changes: Changes[str, Document],
        /,
        *,
        expected: Mapping[str, Document] | None = None,
    ) -> Changes[str, Document]:
        """Writes, replaces, and deletes files after checking every change, serialized within the process.

        Additions need their key, as the key is the path of the file.
        Every new file is written before any is renamed into place, so a failed serialization or write changes nothing.

        Raises:
            ValueError: If the changes are applied already, an addition has no key, or a key lies in no mount.
            exar.errors.Refused: If a mount refuses a change, then nothing is written.
            exar.errors.Conflict: If destinations overlap, an addition exists, or the expected documents changed.
        """
        if changes.applied:
            raise ValueError("The changes are applied already")

        planned: list[_Write] = []
        modes: dict[str | None, Decision] = {}

        for change in changes.items:
            if (key := change.key) is None:
                raise ValueError("Additions to files need a key")

            mount, path = self.resolve(key)
            modes[key] = mount.write
            planned.append(
                _Write(
                    key=key,
                    path=path,
                    document=None if isinstance(change, Remove) else change.value,
                    new=isinstance(change, Add),
                )
            )

        await _authorize(changes, lambda action: modes[action.target], self.name)
        paths = {key: self.resolve(key)[1] for key in expected or {}}
        await shield_to_completion(
            asyncio.to_thread(_commit, planned, paths, dict(expected or {}))
        )

        return replace(changes, applied=True)

    async def sync(self, target: str, /) -> Changes[str, Document]:
        """Writes the changed documents of the mount with a target to its index and returns the changes.

        Raises:
            ValueError: If no mount has the target.
        """
        mount = self._mount(target)

        async with self._syncing.setdefault(target, asyncio.Lock()):
            if (index := self.indexes.get(target)) is None:
                return Changes[str, Document]()

            if (known := self._synced.get(target)) is None:
                stored = await index.read(await index.keys())
                known = self._synced.setdefault(
                    target,
                    {
                        key: _Indexed(file=None, digest=document.meta.get("digest"))
                        for key, document in stored.items()
                    },
                )

            files = await asyncio.to_thread(
                lambda: {key: _File.stat(path) for key, path in self._files(mount)}
            )
            stale = {
                key: file.path
                for key, file in files.items()
                if key not in known or known[key].file != file
            }
            plan = _plan(known, files, stale, await asyncio.to_thread(_load, stale))
            changes = (
                await index.write(Changes(items=plan.items))
                if plan.items
                else Changes[str, Document]()
            )

            for key in plan.gone:
                del known[key]

            known.update(plan.indexed)

            return changes

    async def search(self, *queries: str, limit: int = 10) -> list[Hit[str, Document]]:
        """Searches every mount and fuses their results by reciprocal rank fusion weighted by trust."""

        async def search_mount(mount: Mount) -> Sequence[Hit[str, Document]]:
            if (index := self.indexes.get(mount.target)) is None:
                documents = await asyncio.to_thread(
                    lambda: _load(dict(self._files(mount)))
                )

                return await _search(documents, queries, str, self.measure, limit)

            await self.sync(mount.target)

            return await index.search(*queries, limit=limit)

        rankings = await asyncio.gather(*(search_mount(mount) for mount in self.mounts))

        return _fuse(
            rankings,
            weights=[self.trust.get(mount.target, 1.0) for mount in self.mounts],
            limit=limit,
        )


def _load(paths: Mapping[str, Path]) -> dict[str, Document]:
    """The documents of files by key, leaving out missing files and those that are no UTF-8 text."""
    documents: dict[str, Document] = {}

    for key, path in paths.items():
        with contextlib.suppress(
            FileNotFoundError, IsADirectoryError, UnicodeDecodeError
        ):
            documents[key] = Document.parse(path.read_text())

    return documents


@dataclass(frozen=True, slots=True)
class _SyncPlan:
    """The changes that bring an index up to date with its files and what it then knows of them."""

    items: list[Update[str, Document] | Remove[str]]
    gone: list[str]
    indexed: dict[str, _Indexed]


def _plan(
    known: Mapping[str, _Indexed],
    files: Mapping[str, _File],
    stale: Mapping[str, Path],
    documents: Mapping[str, Document],
) -> _SyncPlan:
    """Plans the updates of the documents whose digest changed and the removal of files that are gone or no text."""
    digests = {key: digest(document.dump()) for key, document in documents.items()}
    gone = [
        key
        for key in known
        if key not in files or (key in stale and key not in documents)
    ]
    items: list[Update[str, Document] | Remove[str]] = [
        Update(
            key=key,
            value=replace(
                document, meta=frozendict({**document.meta, "digest": digests[key]})
            ),
            reason="The file changed",
        )
        for key, document in documents.items()
        if key not in known or known[key].digest != digests[key]
    ]
    items.extend(
        Remove(key=key, reason="The file was deleted or is no text") for key in gone
    )

    return _SyncPlan(
        items=items,
        gone=gone,
        indexed={
            key: _Indexed(file=files[key], digest=digests[key]) for key in documents
        },
    )


def _check_writes(planned: Sequence[_Write]) -> None:
    """Refuse conflicting destinations before making any change."""
    destinations: set[Path] = set()

    for write in planned:
        path = write.path
        identity = Path(fold(path.as_posix())) if folds_case(path.parent) else path

        if identity in destinations:
            raise Conflict(f"{write.key!r} identifies a file being changed twice")

        destinations.add(identity)

        if path.exists(follow_symlinks=False):
            if write.new:
                raise Conflict(f"{write.key!r} exists already")

            if not path.is_file(follow_symlinks=False):
                raise Conflict(f"{write.key!r} must be a regular file")

    for path in destinations:
        if any(parent in destinations for parent in path.parents):
            raise Conflict(f"{path} is below another file being changed")


def _commit(
    planned: Sequence[_Write],
    paths: Mapping[str, Path],
    expected: Mapping[str, Document],
) -> None:
    """Stages every new file before renaming any into place, so a failed serialization or write changes nothing.

    Raises:
        exar.errors.Conflict: If destinations overlap, a new file exists already, or the expected documents changed.
    """
    with _WRITING:
        _check_writes(planned)

        if _load(paths) != expected:
            raise Conflict("The documents changed since the changes were planned")

        staged: list[tuple[Path, Path | None]] = []
        changed: set[Path] = set()

        try:
            for write in planned:
                document = write.document
                partial = (
                    None
                    if document is None
                    else stage_write(write.path, document.dump())
                )
                staged.append((write.path, partial))

            for path, partial in staged:
                if partial is not None:
                    partial.replace(path)
                elif path.exists(follow_symlinks=False):
                    path.unlink()
                else:
                    continue

                changed.add(path.parent)
        finally:
            for _, partial in staged:
                if partial is not None:
                    partial.unlink(missing_ok=True)

        for parent in changed:
            sync_directory(parent)
