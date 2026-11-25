"""The subprocess runner and the typed async wrappers of the command line tools `rg` and `jq` that share it.

`run` stops a child at the time, CPU, memory, file size, and output of its `ProcessLimits`,
and `require` checks that a binary is installed, so a tool that needs one fails when it is built and not on every call.
The wrappers raise a `CommandFailed` the model can read for every failure of their binary.
"""

import asyncio
import base64
import contextlib
import functools
import json
import logging
import os
import re
import shutil
import signal
import sys
from collections.abc import Callable, Container, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    RootModel,
    TypeAdapter,
    ValidationError,
)

from .helpers import shield_to_completion
from .text import decode, os_reason

__all__ = [
    "CommandFailed",
    "OutputTooLarge",
    "ProcessLimits",
    "RgLine",
    "RgMatch",
    "SubprocessError",
    "SubprocessResult",
    "jq_filter",
    "require",
    "rg_count",
    "rg_search",
    "run",
    "sanitized_env",
]

logger = logging.getLogger(__name__)


@functools.cache
def require(binary: str, /) -> str:
    """The path of an installed binary, looked up once per process unless it is missing.

    Raises:
        FileNotFoundError: If the binary is not on the `PATH`.

    Examples:
        >>> require("exar-missing-binary")
        Traceback (most recent call last):
        FileNotFoundError: exar-missing-binary is not installed
    """
    if (path := shutil.which(binary)) is None:
        raise FileNotFoundError(f"{binary} is not installed")

    return path


@dataclass(slots=True, frozen=True, kw_only=True)
class ProcessLimits:
    """What a subprocess may consume before it is stopped, each unlimited for ``None``."""

    timeout: float | None = 30.0
    """Seconds of wall time."""

    cpu: int | None = 30
    """Seconds of CPU time, enforced by the kernel as ``RLIMIT_CPU``."""

    memory: int | None = 4 * 1024**3
    """Bytes of address space, enforced as ``RLIMIT_AS``, except on macOS, which refuses to set it."""

    files: int | None = None
    """Bytes of any one file the child writes, enforced as ``RLIMIT_FSIZE``, past which a write fails and the child is killed."""

    output: int | None = 16 * 1024**2
    """Bytes of standard output the parent buffers, past which the child is killed."""

    def command(self, args: Sequence[str]) -> list[str]:
        """*args* run through `sh`, which sets the kernel limits and then becomes the command.

        A shell sets them rather than a ``preexec_fn``, which would run Python
        between fork and exec in a threaded process and could deadlock on a lock
        another thread held at the fork.  A limit the shell fails to set, such as
        one a seccomp filter denies, stops it with the code 125, which wrappers
        such as `env` and `timeout` exit with for their own failure, and the
        shell's message, so the command never runs without it.

        Examples:
            >>> ProcessLimits(cpu=2, memory=None, files=1024).command(["jq", "."])
            ['/bin/sh', '-c', 'ulimit -t 2 || exit 125; ulimit -f 2 || exit 125; unset PWD OLDPWD SHLVL; exec "$@"', 'sh', 'jq', '.']
        """
        memory = None if self.memory is None or sys.platform == "darwin" else self.memory
        # `ulimit -v` counts KiB, while `ulimit -f` counts the blocks of 512 bytes POSIX
        # specifies, which the bash 3.2 of macOS reads as KiB, doubling the limit there.
        settings = [
            f"ulimit -{flag} {value} || exit 125"
            for flag, value in (
                ("t", self.cpu),
                ("v", None if memory is None else max(memory // 1024, 1)),
                ("f", None if self.files is None else max(self.files // 512, 1)),
            )
            if value is not None
        ]

        # The variables a shell exports of its own, which the command must not see.
        script = "; ".join([*settings, "unset PWD OLDPWD SHLVL", 'exec "$@"'])

        return ["/bin/sh", "-c", script, "sh", *args]


_DEFAULT_LIMITS = ProcessLimits()
"""The limits of a call that names none, shared since they are frozen."""


@dataclass(slots=True, frozen=True)
class SubprocessResult:
    """Captured output of a completed subprocess."""

    stdout: bytes
    stderr: bytes
    returncode: int

    @property
    def stdout_text(self) -> str:
        """Decode stdout as UTF-8."""
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        """Decode stderr as UTF-8."""
        return self.stderr.decode("utf-8", errors="replace")

    def stdout_json[T](self, type_: type[T]) -> T:
        """Parse stdout as JSON and validate it against *type_*."""
        return TypeAdapter(type_).validate_json(self.stdout)

    def stdout_ndjson(self) -> Iterator[JsonValue]:
        """Parse stdout as newline-delimited JSON (one object per line).

        Split on the raw bytes rather than :attr:`stdout_text`: a large result
        set would otherwise be copied twice at full size, and ``str`` splits on
        more separators than JSON treats as line breaks.
        """
        for line in self.stdout.splitlines():
            if line.strip():
                yield json.loads(line)


class SubprocessError(Exception):
    """Raised when a subprocess exits with an unexpected return code."""

    def __init__(
        self,
        result: SubprocessResult,
    ) -> None:
        self.result = result
        super().__init__(
            f"Subprocess exited with code {result.returncode}\n"
            f"stderr: {result.stderr_text}"
        )


class CommandFailed(Exception):
    """Raised by the wrappers of a command line tool when it fails, worded for the model.

    Its own type, so a caller turns only this into a retry,
    never a bug that surfaces as some other error.
    """


class OutputTooLarge(Exception):
    """Raised when a subprocess writes more than `ProcessLimits.output` bytes, after it is killed."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"output exceeds {limit} bytes")


_CHUNK = 64 * 1024
_STDERR_BYTES = 64 * 1024
"""Bytes of standard error kept, the rest is read and dropped so the child never blocks on it."""


def _kill(proc: asyncio.subprocess.Process, *, group: bool) -> None:
    """Kill a child, with the process group it leads if *group*, so its own children die too.

    The group is killed even once the child exited, since a child of its own may still hold its output,
    and the id of a group is not reused while any member of it lives.
    """
    # macOS refuses with EPERM once only the exited leader is left.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        if group:
            os.killpg(proc.pid, signal.SIGKILL)
        elif proc.returncode is None:
            proc.kill()


async def _feed(stream: asyncio.StreamWriter | None, data: bytes | None) -> None:
    """Write *data* to the child and close its input, which a child that stopped reading may refuse."""
    if stream is None:
        return

    try:
        if data:
            stream.write(data)
            await stream.drain()

        stream.close()
        await stream.wait_closed()

    except BrokenPipeError, ConnectionResetError:
        pass


async def _collect(
    stream: asyncio.StreamReader | None,
    limit: int | None,
    *,
    overflow: Callable[[], None] | None = None,
) -> tuple[bytes, bool]:
    """The first *limit* bytes of *stream* and whether it held more.

    Past *limit*, *overflow* is called and reading stops, or without one the
    rest is read and dropped, so the child never blocks on a full pipe.
    """
    if stream is None:
        return b"", False

    buffer = bytearray()
    overflowed = False

    while chunk := await stream.read(_CHUNK):
        kept = chunk if limit is None else chunk[: max(limit - len(buffer), 0)]
        buffer += kept

        if len(kept) < len(chunk):
            overflowed = True

            if overflow is not None:
                overflow()
                break

    return bytes(buffer), overflowed


async def run(
    args: Sequence[str | Path],
    *,
    stdin: bytes | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    allowed_returncodes: Container[int] = (),
    limits: ProcessLimits = _DEFAULT_LIMITS,
    own_group: bool = True,
) -> SubprocessResult:
    """Run a CLI command asynchronously and capture its output.

    The child is killed when the call fails, times out, is cancelled, or the output overflows,
    as a whole process group if it leads one of its own.
    At most `limits.output` bytes of standard output are buffered and the
    first 64 KiB of standard error are kept.

    Args:
        args: Command and arguments.
        stdin: Bytes to pipe to stdin.
        cwd: Working directory.
        env: Environment of the child, the server's own when ``None``.
        check: Raise SubprocessError on unexpected return code.
        allowed_returncodes: Additional return codes besides 0 that are
            acceptable (e.g., rg returns 1 for "no matches").
        limits: Time, CPU, memory, and output the child may use.
        own_group: Whether the child leads a process group of its own, killed with its children.
            Otherwise it joins the group of the caller and is killed alone,
            such as in a worker of `exar.ingest.ProcessPool`, whose group dies with the worker,
            so the child never outlives a worker that is killed.

    Raises:
        TimeoutError: If the child outlives `limits.timeout`, after it is killed.
        OutputTooLarge: If it writes more than `limits.output` bytes, after it is killed.
    """
    proc = await asyncio.create_subprocess_exec(
        *limits.command([str(a) for a in args]),
        stdin=asyncio.subprocess.PIPE
        if stdin is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        # Not `process_group`, which uvloop rejects.
        start_new_session=own_group,
    )

    try:
        async with asyncio.timeout(limits.timeout):
            (stdout, overflowed), (stderr, _), _ = await asyncio.gather(
                _collect(
                    proc.stdout,
                    limits.output,
                    overflow=lambda: _kill(proc, group=own_group),
                ),
                _collect(proc.stderr, _STDERR_BYTES),
                _feed(proc.stdin, stdin),
            )
            returncode = await proc.wait()

        if overflowed and limits.output is not None:
            raise OutputTooLarge(limits.output)

    except BaseException:
        # Never leave the child running when the await is interrupted.
        # A cancelled job or any error must reclaim the process,
        # and reaping it to completion, even through a further cancel,
        # keeps it from lingering as a zombie with its pipes still open.
        _kill(proc, group=own_group)
        await shield_to_completion(proc.wait())

        raise

    result = SubprocessResult(stdout=stdout, stderr=stderr, returncode=returncode)

    if check and returncode != 0 and returncode not in allowed_returncodes:
        raise SubprocessError(result)

    return result


@dataclass(slots=True, frozen=True)
class RgLine:
    """A single line in a match block with its match/context classification."""

    line_number: int
    text: str
    is_match: bool
    match_start: int | None = None
    """Character offset of the first match in *text*, ``None`` for context."""


@dataclass(slots=True, frozen=True)
class RgMatch:
    """A match block from ripgrep, at least one matching line plus any
    surrounding context lines that ripgrep grouped with it."""

    # Relative to the search root, which is where ripgrep was run.
    path: str
    lines: tuple[RgLine, ...]


def _decode(raw: bytes) -> str | None:
    """Bytes ripgrep reported as text, ``None`` where they are not text at all."""
    decoded = decode(raw)
    return decoded.text if decoded is not None else None


class _RgText(BaseModel):
    """ripgrep's "arbitrary data" object, in either form it is emitted in.

    ripgrep fills in ``text`` only for valid UTF-8 and falls back to base64
    ``bytes`` for everything else, such as a line out of a Latin-1 file or a
    filename the filesystem never validated.
    """

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    bytes: str | None = None

    def decode(self) -> str | None:
        """Return the value as text, decoding the base64 fallback.

        The fallback goes through the shared decoder, so legacy Western
        content reads back as text instead of being lost.  ``None`` means
        neither form was filled in, or the bytes are not text at all.
        """
        if self.text is not None:
            return self.text
        if self.bytes is None:
            return None

        try:
            raw = base64.b64decode(self.bytes)
        except ValueError:
            return None

        return _decode(raw)


class _RgSubmatch(BaseModel):
    """Where one match sits in its line, as a byte offset."""

    model_config = ConfigDict(frozen=True)

    start: int


class _RgLineData(BaseModel):
    """The payload of a ``match`` or ``context`` event."""

    model_config = ConfigDict(frozen=True)

    path: _RgText
    lines: _RgText
    line_number: int
    submatches: tuple[_RgSubmatch, ...] = ()

    def match_start(self) -> int | None:
        """The first match's offset in characters rather than ripgrep's bytes.

        ``None`` for a context line, and for a line ripgrep could only hand
        over as bytes, whose decoded characters no longer line up with them.
        """
        if not self.submatches or self.lines.text is None:
            return None

        start = self.submatches[0].start

        if self.lines.text.isascii():
            return start

        return len(self.lines.text.encode()[:start].decode(errors="ignore"))


class _RgLineEvent(BaseModel):
    """An event carrying one matching or surrounding line."""

    model_config = ConfigDict(frozen=True)

    type: Literal["match", "context"]
    data: _RgLineData


class _RgBoundaryEvent(BaseModel):
    """An event that only delimits blocks, whose payload is of no interest."""

    model_config = ConfigDict(frozen=True)

    type: Literal["begin", "end", "context_separator", "summary"]


class _RgEvent(RootModel[_RgLineEvent | _RgBoundaryEvent]):
    """Either kind of event, by its `type`."""


def _parse_event(raw: JsonValue) -> _RgLineEvent | _RgBoundaryEvent | None:
    """Validate one decoded JSON line into a typed event.

    ``None`` for anything this wrapper does not model, which the caller
    skips: an event a future ripgrep adds, or a record whose payload does not
    fit, costs that line rather than every result of the search.
    """
    try:
        return _RgEvent.model_validate(raw).root
    except ValidationError:
        return None


_FILES_PER_RUN = 1_000
"""Most files one search names, which keeps its command line far below ``ARG_MAX``."""


def _matching(*, case_sensitive: bool, literal: bool) -> list[str]:
    """The flags that decide how the pattern matches."""
    args: list[str] = []

    if not case_sensitive:
        args.append("--ignore-case")

    if literal:
        args.append("--fixed-strings")

    return args


async def rg_count(
    pattern: str,
    root: Path,
    *,
    glob: str | None = None,
    case_sensitive: bool = False,
    literal: bool = False,
    exclude_dirs: tuple[str, ...] = (),
    limits: ProcessLimits = _DEFAULT_LIMITS,
) -> dict[str, int]:
    """Count the matching lines of every file below *root*, by path relative to it.

    ripgrep runs with *root* as its working directory rather than being
    handed it as an argument, because that is what anchors a glob: ripgrep
    matches a glob carrying a slash against the path relative to the working
    directory, so ``reports/*.txt`` passed beside an absolute root was tested
    against ``/…/workspace/user:x/reports/note.txt`` and could never match,
    while a slashless ``*.md`` matched by basename and appeared to work.  The
    reported paths come back relative to *root* for the same reason.

    Args:
        pattern: Regex pattern to search for.
        root: Directory to search, which is also what every glob and every
            reported path is relative to.
        glob: Only search files matching this glob, either by basename
            (``"*.md"``) or anchored at *root* (``"reports/*.txt"``).
        case_sensitive: When ``False`` (the default), search
            case-insensitively.  When ``True``, match case exactly.
        literal: When ``True``, treat *pattern* as a fixed string rather
            than a regular expression.
        exclude_dirs: Directory names to skip anywhere in the tree
            (e.g. ``("node_modules", ".git")``).
        limits: Time, CPU, memory, and output ripgrep may use.

    Raises:
        CommandFailed: If ripgrep fails, mostly on a pattern it cannot parse,
            or its output exceeds the limit.
    """
    args: list[str | Path] = [
        "rg",
        "--no-config",
        "--count",
        "--null",
        *_matching(case_sensitive=case_sensitive, literal=literal),
    ]

    if glob:
        args.extend(["--glob", glob])

    for excluded in exclude_dirs:
        args.extend(["--glob", f"!**/{excluded}/**"])

    args.extend(["--", pattern, "."])

    # rg exits 1 when there are no matches, which is no error, and 2 for a file
    # it could not read as well as for a pattern it could not parse, which only
    # the former survives with counts.  An explicitly given target is searched
    # whatever the ignore files say, and `.` is as explicit as an absolute path,
    # so a workspace under an ignored directory (`data/` in the dev tree) stays searchable.
    try:
        result = await run(
            args,
            allowed_returncodes=(1, 2),
            cwd=root,
            env=sanitized_env(),
            limits=limits,
        )

    except OutputTooLarge as exc:
        raise CommandFailed(
            f"rg failed: {exc}, narrow the pattern or the glob"
        ) from exc

    except (SubprocessError, TimeoutError, OSError) as exc:
        raise CommandFailed(_failure("rg", exc)) from exc

    if result.returncode == 2:
        if not result.stdout:
            raise CommandFailed(f"rg failed: {result.stderr_text.strip()}")

        logger.warning(
            "rg skipped files below %s: %s", root, result.stderr_text.strip()
        )

    counts: dict[str, int] = {}

    # `--null` ends a path with NUL and a count with a newline, which a path may hold too,
    # so the output is read as `path NUL count NL` pairs rather than as lines.
    raw, *fields = result.stdout.split(b"\0")

    for field in fields:
        count, _, following = field.partition(b"\n")

        # `.` is the target, so every path arrives spelled `./<relative>`.
        if (path := _decode(raw)) is not None:
            counts[path.removeprefix("./")] = int(count)

        raw = following

    return counts


def _block(path: str | None, lines: list[RgLine]) -> Iterator[RgMatch]:
    """The block of *lines* under *path*, or none before the first line."""
    if lines and path is not None:
        yield RgMatch(path=path, lines=tuple(lines))


def _blocks(result: SubprocessResult, *, context_lines: int) -> Iterator[RgMatch]:
    """The match blocks of one ``--json`` run."""
    block_lines: list[RgLine] = []
    block_path: str | None = None

    for raw in result.stdout_ndjson():
        event = _parse_event(raw)

        if not isinstance(event, _RgLineEvent):
            if event is None:
                logger.debug("Skipping unmodelled ripgrep event: %r", raw)

            yield from _block(block_path, block_lines)
            block_lines = []
            block_path = None
            continue

        line_path = event.data.path.decode()
        text = event.data.lines.decode()

        if line_path is None or text is None:
            logger.debug("Skipping undecodable ripgrep line: %r", raw)
            continue

        is_match = event.type == "match"

        # Without --context, consecutive matches have no separator,
        # so flush the previous block before starting a new one.
        if is_match and context_lines == 0:
            yield from _block(block_path, block_lines)
            block_lines = []

        block_path = line_path
        block_lines.append(
            RgLine(
                line_number=event.data.line_number,
                # Strip the terminator in both spellings: a CRLF source would
                # otherwise carry a carriage return into every formatted line.
                text=text.rstrip("\r\n"),
                is_match=is_match,
                match_start=event.data.match_start(),
            )
        )

    yield from _block(block_path, block_lines)


def _failure(binary: str, error: SubprocessError | TimeoutError | OSError, /) -> str:
    """Why a run of a binary failed, as the model reads it.

    Examples:
        >>> _failure("jq", TimeoutError())
        'jq failed: timed out'
    """
    match error:
        case SubprocessError():
            return f"{binary} failed: {error.result.stderr_text.strip()}"

        case TimeoutError():
            return f"{binary} failed: timed out"

        case OSError():
            return f"{binary} failed: {os_reason(error)}"


async def rg_search(
    pattern: str,
    root: Path,
    files: Sequence[str],
    *,
    context_lines: int = 0,
    max_count: int | None = None,
    case_sensitive: bool = False,
    literal: bool = False,
    limits: ProcessLimits = _DEFAULT_LIMITS,
) -> list[RgMatch]:
    """Search the *files* below *root* for *pattern* using ripgrep.

    The files are named rather than walked, since :func:`rg_count` already
    found which ones match, so a caller reads the lines of only those it shows.
    They are searched in parallel and come back in no particular order.

    Args:
        pattern: Regex pattern to search for.
        root: Directory the *files* and every reported path are relative to.
        files: The files to search.
        context_lines: Number of context lines before and after each match.
        max_count: Most matching lines to report per file, all for ``None``.
        case_sensitive: When ``False`` (the default), search
            case-insensitively.  When ``True``, match case exactly.
        literal: When ``True``, treat *pattern* as a fixed string rather
            than a regular expression.
        limits: Time, CPU, memory, and output each run of ripgrep may use.

    Returns:
        Match blocks parsed from ripgrep's JSON output, each path relative to
        *root*.  Each block contains one or more lines, and ``is_match``
        distinguishes matching lines from surrounding context.

    Raises:
        CommandFailed: If ripgrep cannot run, fails, times out, or its output exceeds the limit.
    """
    args: list[str | Path] = [
        "rg",
        "--no-config",
        "--json",
        *_matching(case_sensitive=case_sensitive, literal=literal),
    ]

    if context_lines > 0:
        args.extend(["--context", str(context_lines)])

    if max_count is not None:
        args.extend(["--max-count", str(max_count)])

    # rg exits 2 for a file it could not read, here one that vanished since
    # it was counted, which costs that file's lines alone.
    try:
        results = await asyncio.gather(
            *(
                run(
                    [*args, "--", pattern, *batch],
                    allowed_returncodes=(1, 2),
                    cwd=root,
                    env=sanitized_env(),
                    limits=limits,
                )
                for batch in batched(files, _FILES_PER_RUN)
            )
        )

    except OutputTooLarge as exc:
        raise CommandFailed(
            f"rg failed: {exc}, narrow the pattern or show fewer lines"
        ) from exc

    except (SubprocessError, TimeoutError, OSError) as exc:
        raise CommandFailed(_failure("rg", exc)) from exc

    return [
        block
        for result in results
        for block in _blocks(result, context_lines=context_lines)
    ]


def sanitized_env(*names: str) -> dict[str, str]:
    """The `PATH` and the *names* of the server's environment that are set, for a tool that must not see the rest.

    The rest of the environment holds the secrets of the server, such as the URL of its database.

    Examples:
        >>> list(sanitized_env("EXAR_UNSET_VARIABLE"))
        ['PATH']
    """
    return {
        name: value
        for name in ("PATH", *names)
        if (value := os.environ.get(name)) is not None
    }


# A module directive is jq's only way to read a file, and an absolute
# ``search`` path reaches any one on disk.  Directives are bare top-level
# keywords, so a field (``.include``), a variable, or a string never matches.
_MODULE_DIRECTIVE = re.compile(r'(?<![\w.$"])(?:import|include)(?!\w)')


async def jq_filter(
    filter_expr: str, document: str, *, limits: ProcessLimits = _DEFAULT_LIMITS
) -> list[JsonValue]:
    """Run a jq filter against a JSON *document*.

    The document is passed as the text it was read as rather than as a parsed
    object: jq parses it either way, so decoding it here only to re-encode it
    would hold a large file in memory twice and move a malformed-JSON error out
    of jq's message and into a traceback of our own.

    The filter sees the document alone: module directives are refused, and jq
    runs without the server's environment, which ``$ENV`` would otherwise expose.

    Args:
        filter_expr: A jq filter expression (e.g. ``".[] | .name"``).
        document: JSON text to filter.
        limits: Time, CPU, memory, and output jq may use.

    Returns:
        List of parsed JSON values produced by the filter.

    Raises:
        CommandFailed: If the document or the jq expression is invalid, or jq
            cannot run, fails, times out, or its output exceeds the limit.
    """
    if _MODULE_DIRECTIVE.search(filter_expr):
        raise CommandFailed("jq failed: import and include are not available")

    try:
        result = await run(
            ["jq", "-c", "--", filter_expr],
            stdin=document.encode("utf-8"),
            env=sanitized_env(),
            limits=limits,
        )
    except OutputTooLarge as exc:
        raise CommandFailed(f"jq failed: {exc}, narrow the filter") from exc

    # jq exits 2 for usage errors, 3 for compile errors, and 5 on bad input.
    except (SubprocessError, TimeoutError, OSError) as exc:
        raise CommandFailed(_failure("jq", exc)) from exc

    return list(result.stdout_ndjson())
