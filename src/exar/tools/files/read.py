"""`Read`, which reads documents as numbered windows of lines."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic_ai import ToolReturn

from ...memory import Mount, digest
from ...memory.mounts import Mounts, current
from ...schema import PythonOnly, serializable
from ...text import fit_window, hint_suffix
from .. import BaseTool, BatchData, Retry
from ..batch import BatchShare, batch_field, serve_batch, tool_return
from ..common import (
    MAX_DOCUMENT_CHARS,
    FullLinesArg,
    Hint,
    OffsetArg,
    line_clip,
    no_hint,
    read_text_or_retry,
    resolve_file_or_retry,
)

__all__ = [
    "DocumentRange",
    "DocumentRead",
    "Read",
    "ReadFilePathArg",
    "ReadLimitArg",
    "ReadsArg",
]

ReadFilePathArg = Annotated[
    str,
    Field(description="Full workspace path of the document to operate on."),
]
ReadLimitArg = Annotated[
    int | None,
    Field(
        description=(
            "Maximum number of lines to return. When omitted, reads as many as "
            "the character budget allows."
        ),
        ge=1,
    ),
]


@dataclass(slots=True, frozen=True)
class DocumentRead:
    """One document to read, and the window of lines to read from it."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    file_path: ReadFilePathArg
    offset: OffsetArg = 1
    limit: ReadLimitArg = None

    @property
    def key(self) -> str:
        """The path, plus the window when it is not the default one.

        Examples:
            >>> DocumentRead("~/a.md").key
            '~/a.md'
            >>> DocumentRead("~/a.md", offset=40, limit=10).key
            '~/a.md (offset=40, limit=10)'
        """
        window = [f"offset={self.offset}"] if self.offset != 1 else []
        window += [f"limit={self.limit}"] if self.limit is not None else []

        return f"{self.file_path} ({', '.join(window)})" if window else self.file_path


ReadsArg = Annotated[
    list[DocumentRead],
    batch_field(
        "Documents to read, each with an optional window. Read every document "
        "you need in one call rather than one call each, since the output budget is "
        "shared between them."
    ),
]


@dataclass(slots=True, frozen=True)
class DocumentRange:
    """A range of lines from a document."""

    file_path: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    content_hash: str
    """Fingerprint of the *full* document, for ``expected_hash`` on a later edit."""


def _no_media(file_path: str, /) -> str | None:
    return None


def _no_advice(mount: Mount, local: str, /) -> str:
    return ""


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Read(BaseTool):
    """Reads documents below mounts as windows of numbered lines, a tool of effect `"read"`.

    Two budgets sit on different axes.
    `max_chars` bounds the window of lines selected, measured as rendered with their numbers,
    so an agent can set it to what a tool return shows whole, since a read resumes from an offset and not from a saved copy.
    `max_line_chars` clips each line, so one very long line, such as an embedded image or a minified bundle, cannot flood the context,
    and `full_lines` lifts the clip for content whose tail carries meaning.
    The typed `content` keeps the lines the model was shown unclipped, and a call of several documents splits `max_chars` between them.
    A document longer than `max_document_chars` is refused by its size before it is read.

    Args:
        mounts: The mounts of the documents, a sequence or a function of the run.
        max_chars: Characters of numbered lines one call shows.
        max_line_chars: Characters per line before it is clipped.
        max_document_chars: Most characters of one document, which is decoded whole.
        viewer: Name of the tool that shows the files `media_type` names.
        hint: What to add to a refusal of a directory or of a file that is no text, a tool to use instead.
        media_type: The media type of a file that the `viewer` shows and this tool refuses, `None` for one it reads.
        advice: What to tell about a file it has read, given its mount and local path, empty for nothing.

    Examples:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "a.md").write_text("one\\ntwo\\nthree\\n")
        >>> read = Read(mounts=[Mount(source=folder)])
        >>> print(asyncio.run(read([DocumentRead("a.md", offset=2)])).return_value)
        ==> a.md (offset=2) <==
        lines 2-3 of 3 (hash b6285c57e879):
        2: two
        3: three
    """

    kind: Literal["read"] = field(default="read", repr=False)
    mounts: Mounts
    name: str = "read"
    max_chars: int = 40_000
    max_line_chars: int = 2000
    max_document_chars: int = MAX_DOCUMENT_CHARS
    viewer: str = "view"
    hint: PythonOnly[Hint] = no_hint
    media_type: PythonOnly[Callable[[str], str | None]] = _no_media
    advice: PythonOnly[Callable[[Mount, str], str]] = _no_advice

    def _read(
        self,
        mounts: Sequence[Mount],
        item: DocumentRead,
        full_lines: bool,
        max_chars: int,
    ) -> ToolReturn[DocumentRange]:
        """Decodes the file and renders the requested window of lines."""
        mount, local, _absolute = resolve_file_or_retry(
            mounts, item.file_path, self.hint
        )
        # Reported as the document is named, which a client keys its coverage by.
        file_path = mount.render(local)
        hint = self.hint("binary", file_path)

        # The file asked for is read as text and never silently swapped for another.
        if (media_type := self.media_type(file_path)) is not None:
            raise Retry(
                f"'{file_path}' is a {media_type} binary, use {self.viewer} "
                f"to send it to a vision model.{hint}"
            )

        # Legacy encodings are decoded rather than refused, and the encoding is reported.
        decoded = read_text_or_retry(
            mount, local, file_path, self.max_document_chars, hint
        )
        file_hash = digest(decoded.text)[:12]
        clip = line_clip(self.max_line_chars, full_lines)
        window = fit_window(
            decoded.text.splitlines(), item.offset, max_chars, clip, item.limit
        )
        result = DocumentRange(
            file_path=file_path,
            start_line=window.start,
            end_line=window.end,
            total_lines=window.total,
            content=window.content,
            content_hash=file_hash,
        )

        if not window.total:
            return tool_return(result, "(empty file)")

        if past_end := window.past_end("file"):
            return tool_return(result, past_end)

        hints = [advice] if (advice := self.advice(mount, local)) else []
        hints += clip.hints()
        hints += window.hints()
        source = decoded.encoding
        encoding = f", decoded from {source}" if source else ""

        return tool_return(
            result,
            f"lines {result.start_line}-{result.end_line} of "
            f"{result.total_lines} (hash {file_hash}{encoding}):"
            f"\n{window.rendered}{hint_suffix(hints)}",
        )

    async def __call__(
        self, reads: ReadsArg, full_lines: FullLinesArg = False
    ) -> ToolReturn[BatchData[DocumentRange]]:
        """Read the content of one or more documents.

        Each read returns the lines from ``offset`` (1-indexed) up to
        ``limit`` lines, each prefixed with its line number, under a header
        naming the document.  The lines are also clamped by a per-call
        character budget shared by the reads, and the output reports how many
        lines remain so the caller can issue a follow-up with a higher
        ``offset``.  Long
        lines are clipped unless ``full_lines`` is set, and the output says
        when that happened so the caller can ask for them whole.  A file
        stored in a legacy encoding is decoded transparently, with the
        source encoding named next to the hash.  A read that fails is
        reported in place without failing the others.
        """
        mounts = current(self.mounts)

        async def read(
            item: DocumentRead, share: BatchShare
        ) -> ToolReturn[DocumentRange]:
            return await asyncio.to_thread(
                self._read, mounts, item, full_lines, share.of(self.max_chars)
            )

        return await serve_batch(
            reads, read, key=lambda item: item.key, concurrency=self.concurrency
        )
