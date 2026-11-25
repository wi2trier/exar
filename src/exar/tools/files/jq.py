"""`Jq`, which filters JSON documents with `jq`."""

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue
from pydantic_ai import ToolReturn

from ...memory.mounts import Mount, Mounts, current
from ...schema import PythonOnly, serializable
from ...subprocess import CommandFailed, jq_filter, require
from ...text import hint_suffix
from .. import BaseTool, BatchData, Retry
from ..batch import BatchShare, batch_field, serve_batch, tool_return
from ..common import (
    MAX_DOCUMENT_CHARS,
    Hint,
    no_hint,
    read_text_or_retry,
    resolve_file_or_retry,
)

__all__ = ["SHAPE_FILTER", "Jq", "JqFilePathsArg", "JqFilterArg", "JqResult"]

SHAPE_FILTER = (
    'def shape: if type == "object" '
    "then (to_entries | map({(.key): (.value | type)}) | add // {}) "
    'elif type == "array" '
    "then {array: length, element: (if length > 0 then (.[0] | shape) else null end)} "
    "else type end; shape"
)
"""The filter a call of `Jq` with no filter runs: the document's keys and their types.

The cheap first call, since a filter cannot be written without knowing the keys of a document,
and the only other way to learn them is `.`, which returns the whole file.
"""


_JSON_SUFFIXES = frozenset({".json"})


JqFilePathsArg = Annotated[
    list[str],
    batch_field(
        "Full workspace paths of the JSON documents to filter, each with the same filter."
    ),
]
JqFilterArg = Annotated[
    str | None,
    Field(
        description=(
            "jq filter expression to run against each document, e.g. "
            "`.items | map(.name)`. Omit it to get the top-level keys and "
            "their types (for an array, its length and the shape of its first "
            "element), which is the cheap first call."
        ),
    ),
]


@dataclass(slots=True, frozen=True)
class JqResult:
    """The values a filter produced, one entry per output jq emitted."""

    file_path: str
    filter: str
    values: tuple[JsonValue, ...] = ()
    encoding: str | None = None


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Jq(BaseTool):
    """Filters JSON documents below mounts with the `jq` binary, which must be installed, a tool of effect `"read"`.

    The filter runs over the whole document and returns only what it selects.
    A call without a filter returns the top-level keys and their types (`SHAPE_FILTER`).
    Each document is filtered on its own, so one that fails is reported under its path.

    Args:
        mounts: The mounts of the documents, a sequence or a function of the run.
        max_document_chars: Most characters of one document, which is decoded whole.
        hint: What to add to a refusal of a directory or a file of another format, a tool to use instead.

    Examples:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> from exar.memory import Mount
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "a.json").write_text('{"items": [{"name": "x"}]}')
        >>> jq = Jq(mounts=[Mount(source=folder)])
        >>> print(asyncio.run(jq(["a.json"], ".items | map(.name)")).return_value)
        ==> a.json <==
        ["x"]
    """

    kind: Literal["jq"] = field(default="jq", repr=False)
    mounts: Mounts
    name: str = "jq"
    max_document_chars: int = MAX_DOCUMENT_CHARS
    hint: PythonOnly[Hint] = no_hint

    def __post_init__(self) -> None:
        require("jq")

    async def _filter(
        self, mounts: Sequence[Mount], file_path: str, filter: str | None
    ) -> ToolReturn[JqResult]:
        mount, local, _absolute = resolve_file_or_retry(mounts, file_path, self.hint)

        if PurePosixPath(file_path).suffix.lower() not in _JSON_SUFFIXES:
            raise Retry(
                f"'{file_path}' is not a JSON document. Filterable formats: "
                f"{', '.join(sorted(_JSON_SUFFIXES))}.{self.hint('format', file_path)}"
            )

        decoded = await asyncio.to_thread(
            read_text_or_retry, mount, local, file_path, self.max_document_chars
        )

        try:
            values = await jq_filter(filter or SHAPE_FILTER, decoded.text)
        except CommandFailed as error:
            raise Retry(str(error)) from error

        result = JqResult(
            file_path=file_path,
            filter=filter or SHAPE_FILTER,
            values=tuple(values),
            encoding=decoded.encoding,
        )
        body = "\n".join(json.dumps(value, default=str) for value in result.values)
        hints = ["shape only, pass a filter to select values"] if filter is None else []

        return tool_return(result, (body or "(no values)") + hint_suffix(hints))

    async def __call__(
        self, file_paths: JqFilePathsArg, filter: JqFilterArg = None
    ) -> ToolReturn[BatchData[JqResult]]:
        """Filter one or more JSON documents with a jq expression.

        Prefer this over reading a JSON document: the filter runs over the
        whole file and returns only what it selects, where a line read spends
        the context on the records the question does not need.  Call it without
        a filter first to learn the top-level keys and their types.  Each
        document is filtered on its own and reported under its path.
        """
        mounts = current(self.mounts)

        async def run(file_path: str, _share: BatchShare) -> ToolReturn[JqResult]:
            return await self._filter(mounts, file_path, filter)

        return await serve_batch(file_paths, run, key=str, concurrency=self.concurrency)
