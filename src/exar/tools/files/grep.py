"""`Grep`, which finds the lines of files that match a pattern."""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass, field, replace
from itertools import groupby
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field
from pydantic_ai import ToolReturn

from ...memory import Mount
from ...memory.mounts import Mounts, current, narrow
from ...schema import PythonOnly, serializable
from ...text import LineClip, hint_suffix
from .. import BaseTool, Retry
from ..common import (
    DEFAULT_EXCLUDE_DIRS,
    FullLinesArg,
    Hint,
    IncludeIgnoredArg,
    is_in_excluded_dir,
    line_clip,
    no_hint,
    omission_hints,
)
from .engines import AnyGrepEngine, GrepLine, GrepMatch, Python

__all__ = [
    "BLOCK_SEP",
    "GROUP_SEP",
    "MAX_RESULTS",
    "Grep",
    "GrepCaseSensitiveArg",
    "GrepContextArg",
    "GrepGlobArg",
    "GrepLiteralArg",
    "GrepMaxResultsArg",
    "GrepOutputMode",
    "GrepOutputModeArg",
    "GrepPatternArg",
    "format_matches",
]

BLOCK_SEP = "\n---\n"
"""Separator placed between distinct blocks in formatted output."""

GROUP_SEP = "\n--\n"
"""Lighter rule than `BLOCK_SEP` for sub-blocks under one heading.

Marks a gap between discontiguous runs that share a heading without repeating the heading.
"""


MAX_RESULTS = 10_000
"""Ceiling on `max_results`, which bounds what a search holds in memory."""

_MAX_CONTEXT = 100
"""Ceiling on `context`, since every block holds its context lines too."""

type GrepOutputMode = Literal["content", "files_with_matches", "count"]

GrepPatternArg = Annotated[
    str,
    Field(
        description=(
            "Text or regular expression pattern to search for. Write it in the "
            "language the documents are likely written in, and retry in another "
            "language (e.g. English or the user's) when nothing matches."
        ),
    ),
]
GrepGlobArg = Annotated[
    str | None,
    Field(
        description=(
            "Optional glob restricting which files are searched, matched "
            "against workspace-relative paths (e.g. `*.md` or `reports/*.txt`). "
            "Lead with a scope prefix to restrict the operation to that one scope; "
            "without a prefix it spans every scope you can access."
        ),
    ),
]
GrepContextArg = Annotated[
    int,
    Field(
        description="Number of context lines to include before and after a match.",
        ge=0,
        le=_MAX_CONTEXT,
    ),
]
GrepMaxResultsArg = Annotated[
    int,
    Field(
        description=(
            "Maximum number of matching lines to return, or of files in the "
            "`files_with_matches` and `count` modes."
        ),
        ge=1,
        le=MAX_RESULTS,
    ),
]
GrepCaseSensitiveArg = Annotated[
    bool,
    Field(
        description=(
            "When true, match case exactly.  Defaults to case-insensitive search."
        ),
    ),
]
GrepLiteralArg = Annotated[
    bool,
    Field(
        description=(
            "When true, treat `pattern` as a literal string rather than a "
            "regular expression.  Use this to search for characters like "
            "`.`, `(`, or `?` without escaping them."
        ),
    ),
]
GrepOutputModeArg = Annotated[
    GrepOutputMode,
    Field(
        description=(
            "Output mode: `content` returns match blocks with context, "
            "`files_with_matches` returns only the filenames that contain "
            "a match, and `count` returns per-file match counts.  Use the "
            "lighter modes to scope a broad search before drilling into "
            "specific matches."
        ),
    ),
]


@dataclass(slots=True, frozen=True)
class _Hit:
    """A file with matches, under the mount that found it."""

    mount: Mount
    local: str
    count: int
    """How many of its lines match."""

    @property
    def file_path(self) -> str:
        """The file as every result spells it."""
        return self.mount.render(self.local)


def _never(local: str, matched: Container[str], /) -> bool:
    return False


def _cut(block: GrepMatch, left: int) -> tuple[GrepMatch, int]:
    """A block up to a number of matching lines, cut after its last kept one, and what is left.

    Examples:
        >>> line = GrepLine(1, "x", is_match=True)
        >>> block, left = _cut(GrepMatch("a", (line, line)), 1)
        >>> len(block.lines), left
        (1, 0)
    """
    lines: list[GrepLine] = []

    for line in block.lines:
        if line.is_match:
            if not left:
                break

            left -= 1

        lines.append(line)

    return replace(block, lines=tuple(lines)), left


def format_matches(matches: Sequence[GrepMatch], clip: LineClip) -> str:
    """Renders matches with each document path once as a heading.

    Every line then carries only its number with `:` for matches and `-` for context.
    Distinct documents are split by `BLOCK_SEP` and discontiguous blocks within a document by `GROUP_SEP`.
    """
    parts: list[str] = []
    prev_file: str | None = None

    for m in matches:
        if prev_file is None:
            head = f"{m.file_path}\n"
        elif m.file_path != prev_file:
            head = f"{BLOCK_SEP}{m.file_path}\n"
        else:
            head = GROUP_SEP

        prev_file = m.file_path

        for i, line in enumerate(m.lines):
            prefix = head if i == 0 else "\n"
            mark = ":" if line.is_match else "-"

            # A match is clipped around its hit, so the cut never takes it.
            text = (
                clip.around(line.text, line.match_start or 0)
                if line.is_match
                else clip(line.text)
            )
            parts.append(f"{prefix}{line.line_number}{mark}{text}")

    return "".join(parts)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Grep(BaseTool):
    """Searches the files below mounts for a pattern, a tool of effect `"read"`.

    Every mode counts the matches of every file in one parallel run per mount,
    and the content mode then reads the lines of only the files it shows, in path order up to `max_results`,
    so what a search holds is bounded by what it renders while its hints still know the total.
    A matching line is clipped around its match, so the cut never takes the hit itself,
    and a context line from its start, and `full_lines` lifts the clip from both.

    Args:
        mounts: The mounts to search, a sequence or a function of the run.
        engine: Scans the files, pure Python or ripgrep as a `GrepEngineKind` of the spec, or any `GrepEngine` of Python code.
        max_line_chars: Characters per line before it is clipped, which keeps long lines from flooding the context.
        exclude_dirs: Names of directories that no search enters unless `include_ignored` is set.
        binary_suffixes: Suffixes of files whose bytes are no text, which a glob must not name alone,
            since a search never opens them and would answer no matches for a file it never read.
        hidden: What `include_ignored` reveals, as the hint on hidden files names it.
        guidance: What to add to the description before how it searches, such as when to prefer another tool.
        notes: What to add to the description after it, such as which duplicate files are left out.
        hint: What to add to a refusal of a glob, a tool to use instead.
        shadowed: Whether a file with matches is dropped for another with matches that represents it,
            given the file and every file of the root with matches, such as an original and its converted text.

    Examples:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "a.md").write_text("one\\ntwo\\nthree\\n")
        >>> grep = Grep(mounts=[Mount(source=folder, target="/notes")])
        >>> print(asyncio.run(grep("T[WH]", context=0)).return_value)
        /notes/a.md
        2:two
        --
        3:three
    """

    kind: Literal["grep"] = field(default="grep", repr=False)
    mounts: Mounts
    engine: AnyGrepEngine = field(default_factory=Python)
    name: str = "grep"
    max_line_chars: int = 200
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS
    binary_suffixes: frozenset[str] = frozenset()
    hidden: str = "common build and vendor directories"
    hint: PythonOnly[Hint] = no_hint
    shadowed: PythonOnly[Callable[[str, Container[str]], bool]] = _never

    def _binary_glob(self, glob: str | None) -> str:
        """The refusal of a glob that can only match files whose bytes are no text, empty for any other."""
        if not glob or PurePosixPath(glob).suffix.lower() not in self.binary_suffixes:
            return ""

        return (
            f"'{glob}' matches only files whose bytes are not text, which a search "
            f"never opens, so it can only ever report no matches.{self.hint('binary', glob)}"
        )

    async def _hits(
        self,
        mount: Mount,
        pattern: str,
        glob: str | None,
        case_sensitive: bool,
        literal: bool,
        ignoring: bool,
    ) -> tuple[list[_Hit], int]:
        """The files with matches in one mount, and how many of them it hid.

        One run per mount, each anchored in its own folder, so an unprefixed `reports/*.txt`
        names that subdirectory of every mount the call spans,
        while a prefixed one has already narrowed the mounts to the one it names.
        The hidden count covers what the mount hides and the post-filter drops,
        and not the directories the engine never enters, so it is a floor.
        """
        root = mount.root()

        if not root.exists():
            return [], 0

        exclude = self.exclude_dirs if ignoring else ()

        counts = await self.engine.count(
            pattern,
            root,
            glob=glob,
            case_sensitive=case_sensitive,
            literal=literal,
            exclude_dirs=exclude,
        )

        allowed = {
            local: count for local, count in counts.items() if not mount.excludes(local)
        }
        hits: list[_Hit] = []
        hidden = 0

        for local, count in allowed.items():
            if self.shadowed(local, allowed):
                continue

            if ignoring and (is_in_excluded_dir(local, exclude) or mount.hides(local)):
                hidden += 1
                continue

            hits.append(_Hit(mount, local, count))

        return hits, hidden

    async def _blocks(
        self,
        hits: Sequence[_Hit],
        pattern: str,
        context: int,
        case_sensitive: bool,
        literal: bool,
        limit: int,
    ) -> list[GrepMatch]:
        """The match blocks of the leading hits, up to a limit of matching lines.

        The counts say which files the limit reaches, so only those are read for their lines,
        and the last one is cut after its last line shown.
        """
        leading: list[_Hit] = []
        reached = 0

        for hit in hits:
            if reached >= limit:
                break

            leading.append(hit)
            reached += hit.count

        groups = [
            (mount, [hit.local for hit in group])
            for mount, group in groupby(leading, key=lambda hit: hit.mount)
        ]
        found = await asyncio.gather(
            *(
                self.engine.search(
                    pattern,
                    mount.root(),
                    files,
                    context_lines=context,
                    max_count=limit,
                    case_sensitive=case_sensitive,
                    literal=literal,
                )
                for mount, files in groups
            )
        )
        blocks: defaultdict[str, list[GrepMatch]] = defaultdict(list)

        for (mount, _files), matches in zip(groups, found, strict=True):
            for match in matches:
                block = replace(match, file_path=mount.render(match.file_path))
                blocks[block.file_path].append(block)

        kept: list[GrepMatch] = []
        left = limit

        for hit in leading:
            for block in blocks[hit.file_path]:
                if left:
                    block, left = _cut(block, left)
                    kept.append(block)

        return kept

    async def __call__(
        self,
        pattern: GrepPatternArg,
        glob: GrepGlobArg = None,
        context: GrepContextArg = 2,
        max_results: GrepMaxResultsArg = 50,
        case_sensitive: GrepCaseSensitiveArg = False,
        literal: GrepLiteralArg = False,
        output_mode: GrepOutputModeArg = "content",
        include_ignored: IncludeIgnoredArg = False,
        full_lines: FullLinesArg = False,
    ) -> ToolReturn[list[GrepMatch]]:
        """Search documents for a pattern.

        Defaults to case-insensitive regex search.  Set ``literal=True``
        to disable regex interpretation, or ``case_sensitive=True`` for
        exact case matching.  Use ``output_mode`` to switch between full
        match context, filenames only, or per-file counts.
        """
        if refusal := self._binary_glob(glob):
            raise Retry(refusal)

        ignoring = not include_ignored
        # A prefix of a mount on the glob narrows the search to that mount.
        # The pattern stays raw, since it matches file content, which is never normalized.
        scoped, glob = narrow(current(self.mounts), glob)
        results = await asyncio.gather(
            *(
                self._hits(mount, pattern, glob, case_sensitive, literal, ignoring)
                for mount in scoped
            )
        )
        hits = sorted(
            (hit for batch, _hidden in results for hit in batch),
            key=lambda hit: hit.file_path,
        )
        hints: list[str] = []

        # The structured data is what backs the text, since a client cites what it is given:
        # the blocks shown in content mode, and in the counting modes, which show no line, the files listed.
        if output_mode == "content":
            clip = line_clip(self.max_line_chars, full_lines)
            data = await self._blocks(
                hits, pattern, context, case_sensitive, literal, max_results
            )
            body = format_matches(data, clip)
            total = sum(hit.count for hit in hits)
            hints += clip.hints()
        else:
            listed = hits[:max_results]
            data = [GrepMatch(hit.file_path, ()) for hit in listed]
            body = "\n".join(
                hit.file_path
                if output_mode == "files_with_matches"
                else f"{hit.file_path}: {hit.count}"
                for hit in listed
            )
            total = len(hits)

        hints += omission_hints(
            hidden=sum(hidden for _batch, hidden in results),
            max_results=max_results,
            total=total,
            what=self.hidden,
        )

        return ToolReturn(
            return_value=(body or "(no matches)") + hint_suffix(hints), metadata=data
        )
