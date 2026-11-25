"""The engines of `Grep`, which count and read the matches of a pattern below a folder."""

import asyncio
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol

from pydantic import Field

from ...schema import Open, serializable
from ...subprocess import CommandFailed, RgMatch, require, rg_count, rg_search
from ...text import decode
from .. import Retry

__all__ = [
    "AnyGrepEngine",
    "GrepEngine",
    "GrepEngineKind",
    "GrepLine",
    "GrepMatch",
    "Python",
    "Ripgrep",
]


@dataclass(slots=True, frozen=True)
class GrepLine:
    """A single line in a match block."""

    line_number: int
    text: str
    is_match: bool
    match_start: int | None = None
    """Character offset of the first match in *text*, ``None`` for context."""


@dataclass(slots=True, frozen=True)
class GrepMatch:
    """A match block in a document with a path relative to the search root."""

    file_path: str
    lines: tuple[GrepLine, ...]


class GrepEngine(Protocol):
    """Counts and reads the matches of a pattern below a folder, which `Grep` calls.

    The `Python` and `Ripgrep` engines are the ones a spec names, and Python code passes any other.
    Each raises `Retry` for what the model can correct, such as a pattern that cannot be parsed
    or an output past the limit, which a narrower pattern avoids.
    """

    async def count(
        self,
        pattern: str,
        root: Path,
        *,
        glob: str | None,
        case_sensitive: bool,
        literal: bool,
        exclude_dirs: tuple[str, ...],
    ) -> dict[str, int]:
        """The number of matching lines by path relative to the root, for the files that have any."""
        ...

    async def search(
        self,
        pattern: str,
        root: Path,
        files: Sequence[str],
        *,
        context_lines: int,
        max_count: int,
        case_sensitive: bool,
        literal: bool,
    ) -> list[GrepMatch]:
        """The match blocks of the files, each with its context lines."""
        ...


def _compile(pattern: str, case_sensitive: bool, literal: bool) -> re.Pattern[str]:
    try:
        return re.compile(
            re.escape(pattern) if literal else pattern,
            0 if case_sensitive else re.IGNORECASE,
        )
    except re.error as error:
        raise Retry(f"Invalid pattern: {error}") from error


def _walk(root: Path, glob: str | None, exclude_dirs: tuple[str, ...]) -> Iterator[str]:
    """The files below a folder by relative path, skipping hidden entries, symlinks, and excluded directories.

    A glob with a slash is anchored at the folder and one without matches the name at any depth.
    """
    anchored = None if glob is None else glob if "/" in glob else f"**/{glob}"

    for directory, names, files in root.walk():
        names[:] = sorted(
            name
            for name in names
            if not name.startswith(".") and name not in exclude_dirs
        )
        relative = directory.relative_to(root)

        for name in sorted(files):
            local = (relative / name).as_posix()

            if (
                not name.startswith(".")
                and not (directory / name).is_symlink()
                and (anchored is None or PurePosixPath(local).full_match(anchored))
            ):
                yield local


def _lines(path: Path) -> list[str] | None:
    """The lines of a text file, `None` for a missing file or one that is no text."""
    try:
        decoded = decode(path.read_bytes())
    except OSError:
        return None

    if decoded is None:
        return None

    lines = decoded.text.split("\n")

    if not lines[-1]:
        lines.pop()

    return [line.removesuffix("\r") for line in lines]


def _blocks(
    lines: Sequence[str], regex: re.Pattern[str], context: int, max_count: int
) -> list[tuple[GrepLine, ...]]:
    """The match blocks of lines, where runs of matches with their context lines are one block each."""
    starts: dict[int, int] = {}

    for index, line in enumerate(lines):
        if len(starts) >= max_count:
            break

        if found := regex.search(line):
            starts[index] = found.start()

    ranges: list[list[int]] = []

    for index in starts:
        first, last = max(index - context, 0), min(index + context, len(lines) - 1)

        if context and ranges and first <= ranges[-1][1] + 1:
            ranges[-1][1] = last
        else:
            ranges.append([first, last])

    return [
        tuple(
            GrepLine(
                line_number=number + 1,
                text=lines[number],
                is_match=number in starts,
                match_start=starts.get(number),
            )
            for number in range(first, last + 1)
        )
        for first, last in ranges
    ]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Python:
    """Searches with Python's regular expressions, which needs nothing installed.

    The files are read whole, so it suits folders of notes and documents and `Ripgrep` large trees.
    A glob supports `*`, `?`, `[]`, and `**`.
    """

    kind: Literal["python"] = field(default="python", repr=False)

    async def count(
        self,
        pattern: str,
        root: Path,
        *,
        glob: str | None,
        case_sensitive: bool,
        literal: bool,
        exclude_dirs: tuple[str, ...],
    ) -> dict[str, int]:
        regex = _compile(pattern, case_sensitive, literal)

        def count() -> dict[str, int]:
            counts: dict[str, int] = {}

            for local in _walk(root, glob, exclude_dirs):
                lines = _lines(root / local)

                if lines and (found := sum(bool(regex.search(line)) for line in lines)):
                    counts[local] = found

            return counts

        return await asyncio.to_thread(count)

    async def search(
        self,
        pattern: str,
        root: Path,
        files: Sequence[str],
        *,
        context_lines: int,
        max_count: int,
        case_sensitive: bool,
        literal: bool,
    ) -> list[GrepMatch]:
        regex = _compile(pattern, case_sensitive, literal)

        def search() -> list[GrepMatch]:
            return [
                GrepMatch(file_path=local, lines=block)
                for local in files
                for block in _blocks(
                    _lines(root / local) or (), regex, context_lines, max_count
                )
            ]

        return await asyncio.to_thread(search)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Ripgrep:
    """Searches with the `rg` binary of ripgrep, which must be installed and is much faster on large trees.

    It reads legacy encodings and skips hidden files and binary files like ripgrep does.
    """

    kind: Literal["ripgrep"] = field(default="ripgrep", repr=False)

    def __post_init__(self) -> None:
        require("rg")

    async def count(
        self,
        pattern: str,
        root: Path,
        *,
        glob: str | None,
        case_sensitive: bool,
        literal: bool,
        exclude_dirs: tuple[str, ...],
    ) -> dict[str, int]:
        try:
            return await rg_count(
                pattern,
                root,
                glob=glob,
                case_sensitive=case_sensitive,
                literal=literal,
                exclude_dirs=exclude_dirs,
            )

        except CommandFailed as error:
            raise Retry(str(error)) from error

    async def search(
        self,
        pattern: str,
        root: Path,
        files: Sequence[str],
        *,
        context_lines: int,
        max_count: int,
        case_sensitive: bool,
        literal: bool,
    ) -> list[GrepMatch]:
        try:
            blocks = await rg_search(
                pattern,
                root,
                files,
                context_lines=context_lines,
                max_count=max_count,
                case_sensitive=case_sensitive,
                literal=literal,
            )

        except CommandFailed as error:
            raise Retry(str(error)) from error

        return [_grep_match(block) for block in blocks]


def _grep_match(block: RgMatch) -> GrepMatch:
    return GrepMatch(
        file_path=block.path,
        lines=tuple(
            GrepLine(
                line_number=line.line_number,
                text=line.text,
                is_match=line.is_match,
                match_start=line.match_start,
            )
            for line in block.lines
        ),
    )


type GrepEngineKind = Annotated[Python | Ripgrep, Field(discriminator="kind")]
"""The engines of `Grep` a spec names, which count and read the matches of a pattern below a folder."""

type AnyGrepEngine = Open[GrepEngineKind, GrepEngine]
"""A `GrepEngineKind` or any other `GrepEngine` of Python code, which serialized forms and JSON Schemas only know as `GrepEngineKind`."""
