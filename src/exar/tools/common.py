"""What the file tools share: the arguments they take and the refusals of a path.

The functions resolve a path a model wrote to the mount and file it names, and turn what is wrong with it into `Retry`,
with the wording a host can extend through a `Hint`.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Annotated, Literal

from pydantic import Field

from ..memory.mounts import Mount, entry_stat, resolve, targets
from ..text import (
    MAX_BYTES_PER_CHAR,
    NOT_TEXT_REASON,
    Decoded,
    LineClip,
    decode,
    limit_hints,
    nfc,
    os_reason,
    plural,
)
from . import Retry

__all__ = [
    "DEFAULT_EXCLUDE_DIRS",
    "MAX_DOCUMENT_CHARS",
    "FullLinesArg",
    "Hint",
    "IncludeIgnoredArg",
    "OffsetArg",
    "Refusal",
    "check_read_budget",
    "excluded_dirs",
    "is_in_excluded_dir",
    "line_clip",
    "near_miss_hint",
    "no_hint",
    "omission_hints",
    "read_bytes_or_retry",
    "read_text_or_retry",
    "refuse_nul",
    "resolve_file_or_retry",
    "workspace_root_hint",
]

DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    "node_modules",
    ".git",
    "dist",
    "build",
    ".next",
    "coverage",
    "__pycache__",
)
"""Build and vendor directory names skipped by default in file listings and searches."""

MAX_DOCUMENT_CHARS = 5_000_000
"""Default cap on the characters of one document a tool decodes whole."""

IncludeIgnoredArg = Annotated[
    bool,
    Field(
        description=(
            "When true, include the build and vendor directories "
            "(node_modules, .git, dist, ...) skipped by default."
        ),
    ),
]
FullLinesArg = Annotated[
    bool,
    Field(
        description=(
            "When true, return every line in full instead of clipping it to a "
            "per-line budget, for content whose tail carries meaning such as a "
            "wide table or a long matched line."
        ),
    ),
]
OffsetArg = Annotated[
    int,
    Field(description="1-based line to start reading from.", ge=1),
]


type Refusal = Literal["directory", "format", "binary", "media"]
"""What a tool refused: a directory where a file is needed, a file of a format it does not take, a glob or file of binary files, or a file that cannot be shown."""

type Hint = Callable[[Refusal, str], str]
"""What a host adds to a refusal of a path, such as the tool to use instead, starting with a space, empty for none.

The refusal of a file that cannot be shown ends with a full stop unless the hint replaces it.
"""


def no_hint(refusal: Refusal, path: str, /) -> str:
    """The hint that adds nothing."""
    return ""


def excluded_dirs(
    include_ignored: bool, /, exclude: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS
) -> tuple[str, ...]:
    """The directory names to skip, none when ignored files are included.

    Examples:
        >>> excluded_dirs(False)[:2], excluded_dirs(True)
        (('node_modules', '.git'), ())
    """
    return () if include_ignored else exclude


def is_in_excluded_dir(rel_path: str, exclude_dirs: tuple[str, ...], /) -> bool:
    """Whether a relative path lies inside any excluded directory name.

    Examples:
        >>> is_in_excluded_dir("a/node_modules/b.js", DEFAULT_EXCLUDE_DIRS), is_in_excluded_dir("a/b.js", ())
        (True, False)
    """
    return bool(exclude_dirs) and any(
        excluded in rel_path.split("/") for excluded in exclude_dirs
    )


def workspace_root_hint(mounts: Sequence[Mount], file_path: str, /) -> str:
    """Names the targets of the mounts when a path leads with none of them, else nothing.

    A caller that dropped the prefix is told which mounts exist rather than left to guess another spelling.

    Examples:
        >>> from pathlib import Path
        >>> mounts = [Mount(source=Path("/a"), target="/notes")]
        >>> workspace_root_hint(mounts, "/other/a.md")
        ' This tool addresses /notes; give the full path, leading with one of them.'
        >>> workspace_root_hint(mounts, "/notes/a.md")
        ''
    """
    if resolve(mounts, file_path) is not None:
        return ""

    names = targets(mounts)

    if not names:
        return ""

    return f" This tool addresses {', '.join(names)}; give the full path, leading with one of them."


def near_miss_hint(absolute: Path, /) -> str:
    """Names the sibling that a missed path almost matched, compared case-folded and in Unicode NFC.

    A refusal of a path copied from a listing one step earlier is otherwise unactionable.
    """
    target = nfc(absolute.name).casefold()

    try:
        siblings = sorted(
            name
            for path in absolute.parent.iterdir()
            if (name := path.name) != absolute.name and nfc(name).casefold() == target
        )
    except OSError:
        return ""

    if not siblings:
        return ""

    return f" A file with an equivalent name exists: {', '.join(siblings)}."


def refuse_nul(path: str, /) -> None:
    r"""Refuses a path with a NUL byte, which no file name can hold and the OS would raise on.

    Raises:
        Retry: If the path holds a NUL byte.

    Examples:
        >>> refuse_nul("a/b.md")
        >>> refuse_nul("a\x00b.md")
        Traceback (most recent call last):
        exar.tools.Retry: 'a\x00b.md' holds a NUL byte, which no file name can.
    """
    if "\0" in path:
        raise Retry(f"{path!r} holds a NUL byte, which no file name can.")


def resolve_file_or_retry(
    mounts: Sequence[Mount], file_path: str, hint: Hint = no_hint
) -> tuple[Mount, str, Path]:
    """The mount, canonical local path, and file of a path to an existing file, or a correctable refusal.

    The globs of the mount are asked about the canonical path and not the spelling the caller sent,
    so no alias, such as `a/../b.md` or a symlink, reaches a file that the mount excludes.
    A path that names no mount is told which mounts exist, a missing file which sibling it almost matched,
    and a directory is refused with the `hint` of the host.

    Raises:
        Retry: If the path names no accessible file.
    """
    refuse_nul(file_path)
    resolved = resolve(mounts, file_path)
    found = None if resolved is None else resolved[0].canonical(resolved[1])
    accessible = (
        None
        if resolved is None or found is None or resolved[0].excludes(found[0])
        else (resolved[0], *found)
    )
    status = entry_stat(accessible[2]) if accessible is not None else None

    if status is not None and S_ISDIR(status.st_mode):
        raise Retry(
            f"'{file_path}' is a directory, not a document.{hint('directory', file_path)}"
        )

    if accessible is None or status is None or not S_ISREG(status.st_mode):
        extra = (
            workspace_root_hint(mounts, file_path)
            if accessible is None
            else near_miss_hint(accessible[2])
        )

        raise Retry(f"'{file_path}' not found.{extra}")

    return accessible


def check_read_budget(canonical: str, size: int, limit: int, /) -> None:
    """Refuses a file too large to decode into a budget of characters, before reading it.

    A file over `MAX_BYTES_PER_CHAR` times the limit cannot fit,
    and anything smaller is left to the exact check of the caller on the decoded text.

    Raises:
        Retry: If the file cannot fit.
    """
    if size <= limit * MAX_BYTES_PER_CHAR:
        return

    raise Retry(
        f"'{canonical}' is too large to read here ({size} bytes, and one "
        f"document may hold at most {limit} characters)."
    )


def read_bytes_or_retry(
    mount: Mount,
    local: str,
    file_path: str,
    check: Callable[[int], None] | None = None,
) -> bytes:
    """Reads a resolved file through no symlink, see `Mount.read_bytes`, or a correctable refusal.

    Blocking, so async callers run it in a thread.

    Raises:
        Retry: If the file is no longer a regular file below the mount or `check` refuses it.
    """
    try:
        return mount.read_bytes(local, check)
    except OSError as error:
        raise Retry(f"'{file_path}' cannot be read: {os_reason(error)}.") from error


def read_text_or_retry(
    mount: Mount, local: str, file_path: str, max_chars: int, hint: str = ""
) -> Decoded:
    """Decodes a resolved file, or refuses one too large for the budget or no text.

    Legacy encodings are decoded rather than refused, so callers can report the source encoding.
    Blocking, so async callers run it in a thread.

    Raises:
        Retry: If the file cannot be read, is too large, or is no text.
    """
    data = read_bytes_or_retry(
        mount,
        local,
        file_path,
        lambda size: check_read_budget(file_path, size, max_chars),
    )
    decoded = decode(data)

    if decoded is None:
        raise Retry(f"'{file_path}' {NOT_TEXT_REASON}.{hint}")

    return decoded


def omission_hints(
    *,
    hidden: int = 0,
    deeper: int = 0,
    max_depth: int | None = None,
    max_results: int | None = None,
    shown: int = 0,
    total: int | None = None,
    what: str = "common build and vendor directories",
) -> list[str]:
    """Says what a search or listing left out, so a short result is not read as all.

    Each hint is what changing one argument would reveal: `hidden` entries that `include_ignored` would expose,
    described by `what`, `deeper` ones that a larger `max_depth` would, and the rest past `max_results`.
    A call that knows its `total` says how many there are,
    and one that stopped once `shown` reached the cap can only say there may be more.
    Render them with `exar.text.hint_suffix`.

    Examples:
        >>> omission_hints(hidden=1, max_results=5, total=9)
        ['1 hidden entry (common build and vendor directories), pass include_ignored=True to reveal them', 'showing 5 of 9, raise max_results to see more']
        >>> omission_hints(max_results=5, shown=5)
        ['reached max_results=5, there may be more']
    """
    hints: list[str] = []

    if hidden:
        hints.append(
            f"{hidden} hidden {plural(hidden, 'entry', 'entries')} ({what}), pass include_ignored=True to reveal them"
        )

    if deeper:
        hints.append(
            f"{deeper} {plural(deeper, 'entry', 'entries')} below max_depth={max_depth}, raise it or pass a "
            "directory as path to see them"
        )

    if max_results is None:
        return hints

    if total is not None:
        hints += limit_hints(max_results, total, "raise max_results to see more")
    elif shown >= max_results:
        hints.append(f"reached max_results={max_results}, there may be more")

    return hints


def line_clip(max_chars: int, full_lines: bool, noun: str = "lines") -> LineClip:
    """The `exar.text.LineClip` of a tool, which keeps every line whole with `full_lines` and names that argument.

    Examples:
        >>> line_clip(4, False)("abcdef"), line_clip(4, True)("abcdef")
        ('abc…', 'abcdef')
    """
    return LineClip(
        None if full_lines else max_chars,
        noun,
        "pass full_lines=true to see them whole",
    )
