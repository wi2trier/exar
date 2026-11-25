"""Mounts, the folders that a `FileMemory` and the file tools of `exar.tools.files` share, and the functions over them.

A `Mount` is a pure access record: a folder, how its files are rendered, which of them exist and which are hidden, and whether a memory may write it.
What a memory keeps per mount, such as its index and trust, lives in the memory.
Its methods are pure: `root` resolves the folder, `render` maps a path local to the mount to its address,
`canonical` finds the spelling that the disk holds, refusing symlinks, `excludes` and `hides` apply its globs, and `walk` lists the entries below a folder.
What spans mounts is a function over a sequence of them, so the same list serves a memory, tools, and a host that writes files itself:
`resolve` and `claim` find the mount and the local path that an address names, `narrow` those of a filter argument, and `targets` lists the mount points.
A tool takes `Mounts`, a sequence or a function of the run that returns one when the tool is called (`current`).
The functions `render` and `strip` map between an address and a path local to a target, for a host that addresses folders it mounts for no tool.
A remote store is mirrored into a local folder by the host, so every mount is a folder on disk.
"""

import errno
import glob
import os
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import BinaryIO

from pydantic.json_schema import SkipJsonSchema

from ..schema import serializable
from ..text import nfc
from ..typing import Decision

__all__ = [
    "Listdir",
    "Mount",
    "Mounts",
    "canonical",
    "check_pattern",
    "claim",
    "current",
    "entry_stat",
    "fold",
    "folds_case",
    "list_names",
    "narrow",
    "open_beneath",
    "render",
    "resolve",
    "respell",
    "strip",
    "targets",
]

_SIGILS = ("/", "~", "@")
"""What a path leads with that names a mount or the file system, so no implied mount takes it."""


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class Mount:
    """A folder of files that memories and tools address, search, and read.

    The globs have the semantics of `PurePosixPath.full_match` on paths relative to the folder,
    where one without a slash matches a name at any depth, one with a slash is anchored at the folder, and a leading slash anchors a single name and one that leads with `!` re-includes what an earlier one matched.
    A path is matched with every folder above it, so a glob that matches a folder matches everything below it.

    Args:
        source: The folder where the files live.
        target: The mount point that the addresses of its files lead with, such as `/notes` or `~`, none if empty.
        implied: Whether it takes a relative path that leads with no target, which a mount without target does anyway.
        write: Whether writes of a memory are refused, need approval, or are allowed.
        include: Globs of the only entries that exist for memories and tools, every entry if empty, applied before `exclude`.
        exclude: Globs of the entries that do not exist for memories and tools, by default the hidden ones.
        hidden: Globs of the entries that tools skip unless they are asked to include what is ignored.

    Examples:
        >>> mount = Mount(source=Path("/notes"), target="/notes", hidden=("**/*.assets/**",))
        >>> mount.render("a.md"), mount.excludes(".git/config"), mount.hides("a.assets/b.png")
        ('/notes/a.md', True, True)
        >>> only = Mount(source=Path("/notes"), include=("*.md",))
        >>> only.excludes("a.md"), only.excludes("a.pdf")
        (False, True)
    """

    source: Path
    target: str = ""
    implied: bool = False
    write: Decision = "deny"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = (".*",)
    hidden: tuple[str, ...] = ()

    def root(self) -> Path:
        """The folder, resolved."""
        return self.source.resolve()

    def render(self, local: str, /) -> str:
        """The address of a local path below the target, the inverse of `resolve`, see the function `render`."""
        return render(self.target, local)

    def implies(self, address: str, /) -> bool:
        """Whether an address that leads with no target is shorthand for a path of the mount."""
        return not self.target or (
            self.implied and bool(address) and not address.startswith(_SIGILS)
        )

    def excludes(self, local: str, /) -> bool:
        """Whether a local path does not exist for the tools, which no argument reveals.

        It exists if the `include` globs match it, any entry without them, and no `exclude` glob does.

        Examples:
            >>> mount = Mount(source=Path("/a"), exclude=("docs/*", "!docs/public"))
            >>> mount.excludes("docs/a.md"), mount.excludes("docs/public/a.md"), mount.excludes("a.md")
            (True, False, False)
        """
        return (bool(self.include) and not _within(self.include, local)) or _within(
            self.exclude, local
        )

    def hides(self, local: str, /) -> bool:
        """Whether a local path is skipped unless the tools are asked to include what is ignored."""
        return _within(self.hidden, local)

    def canonical(
        self, local: str, listdir: Listdir | None = None, /
    ) -> tuple[str, Path] | None:
        """The spelling of a local path as the folder holds it and its file, see the function `canonical`."""
        return canonical(self.root(), local, listdir)

    def read_bytes(
        self, local: str, check: Callable[[int], None] | None = None, /
    ) -> bytes:
        """The content of the regular file at a local path, opened through no symlink, see `open_beneath`.

        The `check` is called with the size of the opened file before it is read, to refuse one too large.
        Blocking.

        Raises:
            OSError: If the file is no longer a regular file below the folder.
        """
        with open_beneath(self.root(), local) as file:
            if check is not None:
                check(os.fstat(file.fileno()).st_size)

            return file.read()

    def walk(
        self, folder: Path, pattern: str | None = None, /, *, dirs: bool = False
    ) -> Iterator[tuple[str, os.stat_result]]:
        """The entries below a folder of the mount by path local to it and own stat, in path order.

        The files are listed, and the folders with `dirs`, without those that the mount excludes and without symlinks,
        since a symlink is an alias for a file that the mount was never asked about.
        The stat is handed on because the size and date of a listing come off exactly that stat.
        A pattern has the semantics of `Path.rglob`, matched against the paths relative to the folder.
        The walk never leaves the folder nor follows a symlink, so no pattern reaches an entry outside the mount,
        and one that is absolute or has a `..` segment matches nothing, which `check_pattern` refuses up front.
        """
        matches = (
            None
            if pattern is None
            else re.compile(
                glob.translate(f"**/{pattern}", recursive=True, include_hidden=True)
            ).fullmatch
        )
        base = self.root()
        found: list[tuple[str, os.stat_result]] = []

        for directory, folders, files in folder.walk():
            relative = _prefix(directory.relative_to(folder))
            local_dir = _prefix(directory.relative_to(base))

            for name in (*folders, *files):
                if matches is not None and not matches(relative + name):
                    continue

                status = entry_stat(directory / name)

                if status is None:
                    continue

                is_dir = S_ISDIR(status.st_mode)

                if (is_dir and not dirs) or (
                    not is_dir and not S_ISREG(status.st_mode)
                ):
                    continue

                local = local_dir + name

                if not self.excludes(local):
                    found.append((local, status))

        yield from sorted(found, key=lambda entry: entry[0].split("/"))


type Mounts = Sequence[Mount] | SkipJsonSchema[Callable[[], Sequence[Mount]]]
"""The mounts of a tool, a sequence, which serializes, or a function of the run, such as the workspaces of its user."""


def current(mounts: Mounts, /) -> Sequence[Mount]:
    """The mounts as they are now, those a function of the run returns.

    Examples:
        >>> mounts = [Mount(source=Path("/notes"))]
        >>> current(mounts) == current(lambda: mounts) == mounts
        True
    """
    return mounts if isinstance(mounts, Sequence) else mounts()


def _prefix(relative: Path, /) -> str:
    """The POSIX spelling of a relative folder that a name is appended to, empty for the folder itself."""
    return "" if relative == Path() else f"{relative.as_posix()}/"


def check_pattern(pattern: str | None, /) -> None:
    """Refuses a glob that could reach beyond the folder it is matched in.

    Raises:
        ValueError: If the pattern is absolute or has a `..` segment.

    Examples:
        >>> check_pattern("docs/**/*.md")
        >>> check_pattern("../*")
        Traceback (most recent call last):
        ValueError: The pattern '../*' must be relative and must not contain '..'.
    """
    if pattern is None:
        return

    path = PurePosixPath(pattern)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"The pattern {pattern!r} must be relative and must not contain '..'."
        )


def render(target: str, local: str, /) -> str:
    """The address of a local path below a target, the bare target for an empty path, the path itself without target.

    Examples:
        >>> render("~", "x.md"), render("~", ""), render("", "x.md")
        ('~/x.md', '~', 'x.md')
    """
    if not target:
        return local

    return f"{target}/{local}" if local else target


def strip(target: str, address: str, /) -> str | None:
    """The local remainder of an address that leads with a target, empty for the bare target, else `None`, the inverse of `render`.

    Examples:
        >>> strip("~", "~/x.md"), strip("~", "~"), strip("~", "~x/y.md"), strip("", "x.md")
        ('x.md', '', None, None)
    """
    if not target:
        return None

    if address == target:
        return ""

    head = f"{target}/"

    return address.removeprefix(head) if address.startswith(head) else None


def claim(mounts: Sequence[Mount], raw: str, /) -> tuple[Mount, str] | None:
    """The mount that claims an address by the longest target, and the local remainder, empty for a bare target.

    Examples:
        >>> mounts = [Mount(source=Path("/a"), target="~"), Mount(source=Path("/b"), target="@team")]
        >>> claim(mounts, "@team/x.md") == (mounts[1], "x.md"), claim(mounts, "x.md")
        (True, None)
    """
    claims = [
        (mount, local)
        for mount in mounts
        if (local := strip(mount.target, raw)) is not None
    ]

    return max(claims, key=lambda found: len(found[0].target), default=None)


def resolve(mounts: Sequence[Mount], raw: str, /) -> tuple[Mount, str] | None:
    """The mount an address names and its local path, one that leads with no target falling back to the first mount that implies it.

    The address is folded to Unicode NFC, the funnel every path passes through,
    so a decomposed spelling still resolves to the canonically named file.

    Examples:
        >>> mounts = [Mount(source=Path("/a"), target="~", implied=True), Mount(source=Path("/b"), target="@team")]
        >>> resolve(mounts, "x.md") == (mounts[0], "x.md"), resolve(mounts, "@team/x.md") == (mounts[1], "x.md")
        (True, True)
        >>> resolve(mounts, "@ghost/x.md"), resolve(mounts, "/etc/x")
        (None, None)
    """
    raw = nfc(raw)
    claimed = claim(mounts, raw)

    if claimed is not None and claimed[1]:
        return claimed

    return next(((mount, raw) for mount in mounts if mount.implies(raw)), None)


def narrow(
    mounts: Sequence[Mount], raw: str | None, /
) -> tuple[Sequence[Mount], str | None]:
    """Narrows the mounts to the one that a filter argument, such as a glob, leads with the mount point of.

    The remainder is `None` for a bare target.
    Without a known mount point the mounts and the argument pass through, so an unprefixed filter spans every mount.
    The argument is folded to Unicode NFC like every path.

    Examples:
        >>> mounts = [Mount(source=Path("/a"), target="~"), Mount(source=Path("/b"), target="@team")]
        >>> narrow(mounts, "~/reports") == ([mounts[0]], "reports"), narrow(mounts, "reports") == (mounts, "reports")
        (True, True)
    """
    if not raw:
        return mounts, raw

    raw = nfc(raw)
    claimed = claim(mounts, raw)

    if claimed is None:
        return mounts, raw

    mount, local = claimed

    return [mount], local or None


def targets(mounts: Sequence[Mount], /) -> list[str]:
    """The targets that an address may lead with, none for a mount without one.

    Examples:
        >>> targets([Mount(source=Path("/a"), target="~"), Mount(source=Path("/b"))])
        ['~']
    """
    return [mount.target for mount in mounts if mount.target]


def _matched(patterns: Sequence[str], path: PurePosixPath, /) -> bool:
    """Whether the last glob that matches a path excludes it, which one that leads with `!` does not."""
    matched = False

    for pattern in patterns:
        negated = pattern.startswith("!")
        glob = pattern.removeprefix("!")

        if path.full_match(glob.removeprefix("/") if "/" in glob else f"**/{glob}"):
            matched = not negated

    return matched


def _within(patterns: Sequence[str], local: str, /) -> bool:
    """Whether the globs match a path or a folder above it."""
    path = PurePosixPath(local)

    return bool(patterns) and any(
        _matched(patterns, folder) for folder in (path, *path.parents) if folder.parts
    )


def entry_stat(path: Path, /) -> os.stat_result | None:
    """The own stat of a path, or `None` when it names no addressable entry, a symlink counting as absent."""
    try:
        status = path.lstat()
    except OSError:
        return None

    return None if S_ISLNK(status.st_mode) else status


type Listdir = Callable[[Path], Sequence[str]]
"""Lists the names in a folder, which `respell` takes so that a caller can cache them."""

_FOLDING: dict[int, bool] = {}
"""Whether each file system folds case, by device, which is as many as are mounted."""


def fold(name: str, /) -> str:
    return nfc(name).casefold()


def _probe_folding(directory: Path, /) -> bool:
    for candidate in (directory, *directory.parents):
        swapped = (
            candidate.with_name(candidate.name.swapcase())
            if candidate.name
            else candidate
        )

        if swapped == candidate:
            continue

        try:
            return swapped.samefile(candidate)
        except OSError:
            return False

    return False


def folds_case(directory: Path, /) -> bool:
    """Whether the file system holding a folder treats names that differ in case as one.

    Probed once per device, read-only: the nearest existing ancestor whose name has a case is looked up under its swapped spelling,
    which a case-insensitive file system such as macOS or Windows resolves to the same inode.
    Every later call costs one `stat`, however many folders ask.
    """
    for candidate in (directory, *directory.parents):
        try:
            device = candidate.stat().st_dev
        except OSError:
            continue

        folded = _FOLDING.get(device)

        if folded is None:
            folded = _FOLDING[device] = _probe_folding(candidate)

        return folded

    return False


def list_names(directory: Path, /) -> Sequence[str]:
    """The names in a folder, none when it cannot be listed."""
    try:
        return [entry.name for entry in directory.iterdir()]
    except OSError:
        return ()


def respell(root: Path, local: str, listdir: Listdir | None = None, /) -> str:
    """A local path with each existing segment spelled the way the disk spells it.

    On a case-insensitive file system `A.md` opens `a.md`, so a path spelled differently would otherwise become a second name for one file,
    and a glob would be asked about a name that the file does not have.
    A missing segment keeps its spelling, and so does every path on a case-sensitive file system.
    """
    if not local or not folds_case(root):
        return local

    listdir = listdir or list_names
    spelled = PurePosixPath()

    for part in PurePosixPath(local).parts:
        key = fold(part)
        spelled /= next(
            (name for name in listdir(root / spelled) if fold(name) == key), part
        )

    return str(spelled)


def canonical(
    root: Path, local: str, listdir: Listdir | None = None, /
) -> tuple[str, Path] | None:
    """The spelling of a local path as the disk holds it, and its file, the one place where a spelling becomes an identity.

    `..` segments are folded away and each existing segment is respelled,
    so the name that a glob is asked about is the file that the operation would touch.
    It is `None` if the path leaves the folder, is the folder itself, which names no entry,
    or passes through a symlink, which is an alias that the mount was never asked about and is never followed.
    """
    root = root.resolve()
    absolute = Path(os.path.normpath(root / local))

    if not absolute.is_relative_to(root) or absolute == root:
        return None

    spelled = respell(root, absolute.relative_to(root).as_posix(), listdir)
    current = root

    for part in PurePosixPath(spelled).parts:
        current /= part

        try:
            status = current.lstat()
        except FileNotFoundError:
            break
        except OSError:
            return None

        if S_ISLNK(status.st_mode):
            return None

    return spelled, root / spelled


def open_beneath(root: Path, local: str, /) -> BinaryIO:
    """Opens the regular file at a local path below a folder for reading, following no symlink in any of its segments.

    Each folder is opened relative to the one above it, so a segment swapped for a symlink meanwhile is refused too,
    and the file is opened without blocking and checked to be regular, so a FIFO or a device never stalls the reader.

    Raises:
        OSError: If a segment is missing or a symlink (`ELOOP`), the path leaves the folder (`EINVAL`),
            or the file is no regular file (`EINVAL`).

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     _ = Path(folder, "a.txt").write_text("hi")
        ...     with open_beneath(Path(folder), "a.txt") as file:
        ...         file.read()
        b'hi'
    """
    parts = PurePosixPath(local).parts

    if not parts or any(part in ("/", ".", "..") for part in parts):
        raise OSError(
            errno.EINVAL, "The path does not name a file below the folder", local
        )

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    folder = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)

    try:
        for part in parts[:-1]:
            below = os.open(part, flags | os.O_DIRECTORY, dir_fd=folder)
            os.close(folder)
            folder = below

        handle = os.open(parts[-1], flags | os.O_NONBLOCK, dir_fd=folder)
    finally:
        os.close(folder)

    try:
        if not S_ISREG(os.fstat(handle).st_mode):
            raise OSError(errno.EINVAL, "The path does not name a regular file", local)

        os.set_blocking(handle, True)

        return os.fdopen(handle, "rb")
    except BaseException:
        os.close(handle)
        raise
