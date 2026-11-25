"""Tools that search and read the files below mounts, configured as serializable classes.

Each tool is a keyword-only dataclass like the measures of cbrkit: its fields are plain data, `__call__` is the tool,
and `tool` turns it into the plain function that agent frameworks register, with an effect of `"read"`.
The `mounts` of the files are plain data, or a function of the run for a host that varies them per run,
and the wording of a tool is data too: its `description` and the descriptions of its `arguments` replace the neutral defaults.
Results are pydantic-ai `ToolReturn`s, the text for the model and the typed data for programs and clients.
The tools need the `pydantic-ai` extra, and the ones that run a binary need it installed: `Ripgrep` needs `rg`, `Jq` needs `jq`.

- `Grep`: Finds the lines of files that match a pattern, with an `engine` that is pure Python (`Python`) or `rg` (`Ripgrep`).
- `Jq`: Filters JSON documents with `jq` instead of reading them.
- `Read`: Reads documents as numbered windows of lines within a budget of characters.
- `Ls`: Lists the files and folders of a tree with their sizes and dates, by pattern or depth.
- `file_tools`: The tools of a `FileMemory`.
"""

from dataclasses import replace

from ...errors import Conflict, Refused
from ...memory import FileMemory, memory_tools
from ...text import hint_suffix, limit_hints, replace_once
from ...typing import Changes, Tool, Update
from .. import as_tool, retry_on
from .engines import (
    AnyGrepEngine,
    GrepEngine,
    GrepEngineKind,
    GrepLine,
    GrepMatch,
    Python,
    Ripgrep,
)
from .grep import (
    BLOCK_SEP,
    GROUP_SEP,
    Grep,
    GrepCaseSensitiveArg,
    GrepContextArg,
    GrepGlobArg,
    GrepLiteralArg,
    GrepMaxResultsArg,
    GrepOutputMode,
    GrepOutputModeArg,
    GrepPatternArg,
    format_matches,
)
from .jq import SHAPE_FILTER, Jq, JqFilePathsArg, JqFilterArg, JqResult
from .ls import (
    DocumentSummary,
    Ls,
    LsMaxDepthArg,
    LsMaxResultsArg,
    LsPathArg,
    LsPatternsArg,
)
from .read import (
    DocumentRange,
    DocumentRead,
    Read,
    ReadFilePathArg,
    ReadLimitArg,
    ReadsArg,
)

__all__ = [
    "BLOCK_SEP",
    "GROUP_SEP",
    "SHAPE_FILTER",
    "AnyGrepEngine",
    "DocumentRange",
    "DocumentRead",
    "DocumentSummary",
    "Grep",
    "GrepCaseSensitiveArg",
    "GrepContextArg",
    "GrepEngine",
    "GrepEngineKind",
    "GrepGlobArg",
    "GrepLine",
    "GrepLiteralArg",
    "GrepMatch",
    "GrepMaxResultsArg",
    "GrepOutputMode",
    "GrepOutputModeArg",
    "GrepPatternArg",
    "Jq",
    "JqFilePathsArg",
    "JqFilterArg",
    "JqResult",
    "Ls",
    "LsMaxDepthArg",
    "LsMaxResultsArg",
    "LsPathArg",
    "LsPatternsArg",
    "Python",
    "Read",
    "ReadFilePathArg",
    "ReadLimitArg",
    "ReadsArg",
    "Ripgrep",
    "file_tools",
    "format_matches",
]


def file_tools(
    memory: FileMemory, /, name: str = "files", max_results: int = 100
) -> list[Tool]:
    """The tools of `exar.memory.memory_tools` plus tools to list, grep, and edit files, named `<name>_list` and so on.

    The `grep` is a `Grep` over the mounts of the memory.

    Args:
        memory: The files.
        name: Prefix of the tool names.
        max_results: Most keys that `list` returns.

    Examples:
        >>> [tool.__name__ for tool in file_tools(FileMemory(mounts=[]))][3:]
        ['files_list', 'files_grep', 'files_edit']
    """

    async def list_(path: str | None = None) -> str:
        """Lists the keys of the files, one per line.

        Args:
            path: Folder to list, every mount if omitted.
        """
        keys = await memory.keys(path)

        return (
            "\n".join(keys[:max_results])
            + hint_suffix(limit_hints(max_results, len(keys), "narrow the path"))
            or "No files."
        )

    async def edit(key: str, old: str, new: str) -> str:
        """Replaces the one occurrence of a text in a file.

        Args:
            key: Key of the file.
            old: Text to replace, which must occur exactly once.
            new: Replacement.
        """
        with retry_on(Refused, Conflict, ValueError):
            document = (await memory.read([key])).get(key)

            if document is None:
                raise ValueError(f"{key!r} does not exist")

            text, _ = replace_once(document.text, old, new)
            change = Update(
                key=key,
                value=replace(document, text=text),
                reason="Edited",
            )
            await memory.write(Changes(items=[change]), expected={key: document})

        return f"Edited {key}."

    grep = Grep(mounts=memory.mounts, name=f"{name}_grep")

    return [
        *memory_tools(memory, name),
        as_tool(list_, name=f"{name}_list", effect="read"),
        grep.tool(),
        as_tool(edit, name=f"{name}_edit", effect="write"),
    ]
