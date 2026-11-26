"""`Ls`, which lists the files and folders of a tree."""

import asyncio
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path, PurePosixPath
from stat import S_ISDIR
from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import ToolReturn

from ...memory import Mount
from ...memory.mounts import Mounts, check_pattern, current, narrow, targets
from ...schema import PythonOnly, serializable
from ...text import format_bytes, hint_suffix, nfc
from .. import BaseTool, Retry, retry_on
from ..batch import batch_field, tool_return
from ..common import (
    DEFAULT_EXCLUDE_DIRS,
    IncludeIgnoredArg,
    excluded_dirs,
    is_in_excluded_dir,
    omission_hints,
    refuse_nul,
)
from .grep import MAX_RESULTS

__all__ = [
    "DocumentSummary",
    "Ls",
    "LsMaxDepthArg",
    "LsMaxResultsArg",
    "LsPathArg",
    "LsPatternsArg",
]

LsPathArg = Annotated[
    str | None,
    Field(
        description=(
            "Optional subdirectory to scope the operation within. "
            "Lead with a scope prefix to restrict the operation to that one scope; "
            "without a prefix it spans every scope you can access. "
            "Omit to cover them all."
        ),
    ),
]
LsPatternsArg = Annotated[
    list[str] | None,
    batch_field(
        "Optional glob patterns matched against workspace-relative filenames "
        "at any depth (e.g. `*.md`, `reports/*.csv`), so only files matching "
        "any of them are listed, each once, and `max_depth` does not apply. "
        "To restrict to one workspace, prefix the `path` argument rather than "
        "the patterns."
    ),
]
LsMaxDepthArg = Annotated[
    int | None,
    Field(
        description="Maximum nesting depth relative to the selected directory, null for every depth.",
        ge=1,
    ),
]
LsMaxResultsArg = Annotated[
    int,
    Field(
        description="Maximum number of entries to return.",
        ge=1,
        le=MAX_RESULTS,
    ),
]


@dataclass(slots=True, frozen=True)
class DocumentSummary:
    """Summary of a document or directory."""

    file_path: str
    size: int
    modified_at: datetime
    is_directory: bool = False
    case_base: bool = False
    """Whether it is a case base with a saved similarity configuration."""


def _relative_depth(local: str, subdir: str | None) -> int:
    """The nesting depth of a path below the directory a walk started at, a direct child counting as 1.

    Examples:
        >>> _relative_depth("docs/a/b.md", "docs"), _relative_depth("a.md", None)
        (2, 1)
    """
    relative = local[len(subdir) + 1 :] if subdir else local

    return relative.count("/") + 1


def _normalize_subdir(subdir: str | None) -> str | None:
    """Folds the harmless spellings of a directory argument, `./docs/`, `docs/`, and `docs` naming one directory and `.` the root.

    A `..` segment is left to `exar.memory.mounts.canonical`, which resolves it against the folder it is relative to.

    Examples:
        >>> _normalize_subdir("./docs/"), _normalize_subdir("."), _normalize_subdir(None)
        ('docs', None, None)
    """
    if not subdir:
        return None

    normalized = PurePosixPath(subdir).as_posix()

    return None if normalized == "." else normalized


def _scoped_directory(
    mounts: Sequence[Mount], path: str | None
) -> tuple[Sequence[Mount], tuple[tuple[Mount, Path], ...], str | None]:
    """Resolves a directory argument into the mounts a walk starts from.

    Returns the mounts the prefix of the path narrows to, a pair of mount and folder for each that holds the directory,
    and the directory in the canonical spelling its entries are named under.
    Resolving it once costs one fold and one stat for the whole call and bounds the walk to the subtree.

    Raises:
        Retry: If the path names a directory that no mount holds.
    """
    if path is not None:
        refuse_nul(path)

    scoped, subdir = narrow(mounts, path)
    subdir = _normalize_subdir(subdir)

    if subdir is None:
        return (
            scoped,
            tuple((mount, root) for mount in scoped if (root := mount.root()).is_dir()),
            None,
        )

    folders: list[tuple[Mount, Path]] = []
    spelled = subdir

    for mount in scoped:
        resolved = mount.canonical(subdir)

        if resolved is not None and resolved[1].is_dir():
            spelled, folder = resolved
            folders.append((mount, folder))

    if not folders:
        names = targets(scoped)
        known = f" This tool addresses {', '.join(names)}." if names else ""

        raise Retry(f"Directory '{path or subdir}' does not exist.{known}")

    return scoped, tuple(folders), spelled


def _empty_message(noun: str, mounts: Sequence[Mount], subdir: str | None, /) -> str:
    """An empty result as what the call covered, in the grammar a path argument takes back and never as a host path.

    Examples:
        >>> from pathlib import Path
        >>> _empty_message("no documents", [Mount(source=Path("/a"), target="~")], "docs")
        "(no documents under '~/docs')"
    """
    if len(mounts) == 1:
        label = mounts[0].render(subdir or "")
        location = f" under {label!r}" if label else " in the workspace"
    elif subdir is not None:
        location = f" under {subdir!r} in any accessible workspace"
    else:
        location = " in accessible workspaces"

    return f"({noun}{location})"


def _walk_entries(
    folders: tuple[tuple[Mount, Path], ...], pattern: str | None, *, include_dirs: bool
) -> Iterator[tuple[Mount, str, os.stat_result]]:
    """The entries below folders by mount, path local to the mount, and own stat, see `exar.memory.mounts.walk`."""
    for mount, folder in folders:
        for local, status in mount.walk(folder, pattern, dirs=include_dirs):
            yield mount, local, status


def _never_case_base(mount: Mount, local: str, /) -> bool:
    return False


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Ls(BaseTool):
    """Lists the files and folders below mounts with their sizes and dates, a tool of effect `"read"`.

    Without patterns it lists files and folders down to `max_depth`,
    and with them the files that any pattern matches at any depth, each once.
    What the listing left out is named, so a short result is not read as all.

    Args:
        mounts: The mounts to list, a sequence or a function of the run.
        exclude_dirs: Names of directories that no listing enters unless `include_ignored` is set.
        hidden: What `include_ignored` reveals, as the hint on hidden entries names it.
        notes: What to add to the description after it, such as which other entries are hidden.
        size: Renders a size in bytes.
        case_base: Whether the entry of a mount and local path is a case base, which the listing marks.

    Examples:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "a.md").write_text("one")
        >>> ls = Ls(mounts=[Mount(source=folder, target="/notes")])
        >>> [entry.file_path for entry in asyncio.run(ls(patterns=["*.md"])).metadata]
        ['/notes/a.md']
    """

    kind: Literal["ls"] = field(default="ls", repr=False)
    mounts: Mounts
    name: str = "ls"
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS
    hidden: str = "common build and vendor directories"
    size: PythonOnly[Callable[[int], str]] = format_bytes
    case_base: PythonOnly[Callable[[Mount, str], bool]] = _never_case_base

    def _scan(
        self,
        folders: tuple[tuple[Mount, Path], ...],
        patterns: list[str] | None,
        subdir: str | None,
        max_depth: int | None,
        max_results: int,
        exclude: tuple[str, ...],
    ) -> tuple[list[DocumentSummary], list[str]]:
        """Collects the entries, with hints naming what was left out.

        Each pattern is a path argument matched against canonically named entries, so it is folded to Unicode NFC here,
        since unlike a scoped subdirectory it passes no mount.
        One walk per pattern keeps the semantics of each, and a file that two of them match is listed once.
        """
        if patterns:
            unique = dict.fromkeys(nfc(pattern) for pattern in patterns)

            with retry_on(ValueError):
                for pattern in unique:
                    check_pattern(pattern)

            walk = chain.from_iterable(
                _walk_entries(folders, pattern, include_dirs=False)
                for pattern in unique
            )
            max_depth = None
        else:
            walk = _walk_entries(folders, None, include_dirs=True)

        results: dict[str, DocumentSummary] = {}
        hidden: set[str] = set()
        deeper = 0

        for mount, local, status in walk:
            filename = mount.render(local)
            ignored = bool(exclude) and (
                is_in_excluded_dir(local, exclude) or mount.hides(local)
            )

            if max_depth is not None and _relative_depth(local, subdir) > max_depth:
                deeper += not ignored
                continue

            if ignored:
                hidden.add(filename)
                continue

            is_dir = S_ISDIR(status.st_mode)
            results[filename] = DocumentSummary(
                file_path=filename,
                size=0 if is_dir else status.st_size,
                modified_at=datetime.fromtimestamp(status.st_mtime, tz=UTC),
                is_directory=is_dir,
                case_base=self.case_base(mount, local),
            )

            if len(results) >= max_results:
                break

        return list(results.values()), omission_hints(
            hidden=len(hidden),
            deeper=deeper,
            max_depth=max_depth,
            max_results=max_results,
            shown=len(results),
            what=self.hidden,
        )

    def _list(
        self,
        mounts: Sequence[Mount],
        path: str | None,
        patterns: list[str] | None,
        max_depth: int | None,
        max_results: int,
        include_ignored: bool,
    ) -> ToolReturn[list[DocumentSummary]]:
        """Scans the mounts and renders one line per entry."""
        mounts, folders, subdir = _scoped_directory(mounts, path)
        entries, hints = self._scan(
            folders,
            patterns,
            subdir,
            max_depth,
            max_results,
            excluded_dirs(include_ignored, self.exclude_dirs),
        )
        note = hint_suffix(hints)

        if not entries:
            noun = "no matches" if patterns else "no documents"

            return tool_return(entries, _empty_message(noun, mounts, subdir) + note)

        lines = [
            f"{'d' if entry.is_directory else '-'} {entry.modified_at:%Y-%m-%d %H:%M}Z  "
            f"{self.size(entry.size):>8}  {entry.file_path}"
            f"{'  (case base)' if entry.case_base else ''}"
            for entry in entries
        ]

        return tool_return(entries, "\n".join(lines) + note)

    async def __call__(
        self,
        path: LsPathArg = None,
        patterns: LsPatternsArg = None,
        max_depth: LsMaxDepthArg = 1,
        max_results: LsMaxResultsArg = 200,
        include_ignored: IncludeIgnoredArg = False,
    ) -> ToolReturn[list[DocumentSummary]]:
        """List available documents with sizes and dates.

        Pass ``patterns`` to list only the files whose names match, at any
        depth.  Hidden entries such as build and vendor directories are skipped
        by default, pass ``include_ignored=True`` to include them.
        """
        # The walk stats every entry, so it goes to a thread rather than the event loop.
        return await asyncio.to_thread(
            self._list,
            current(self.mounts),
            path,
            patterns,
            max_depth,
            max_results,
            include_ignored,
        )
