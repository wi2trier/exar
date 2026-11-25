"""Tools that expose memories to language models, see `exar.tools.files.file_tools` for files."""

from collections.abc import Mapping, Sequence
from types import FunctionType

from ..errors import Conflict, Refused
from ..tools import effect, retry_on
from ..typing import Changes, Hit, Memory, Tool

__all__ = ["memory_tools"]


def memory_tools[K, V](memory: Memory[K, V], /, name: str = "memory") -> list[Tool]:
    """Tools to search, read, and write a memory, named `<name>_search`, `<name>_read`, and `<name>_write`.

    Refused and conflicting writes raise `exar.tools.Retry`, so the model can correct itself.

    Examples:
        >>> from exar.memory import InMemoryMemory
        >>> [tool.__name__ for tool in memory_tools(InMemoryMemory[str, str](), "notes")]
        ['notes_search', 'notes_read', 'notes_write']
    """

    @effect("read")
    async def search(queries: list[str], limit: int = 10) -> Sequence[Hit[K, V]]:
        """Searches the memory, the most relevant entries first.

        Args:
            queries: What to look for, several phrasings or aspects whose rankings are fused into one.
            limit: Most entries to return.
        """
        return await memory.search(*queries, limit=limit)

    @effect("read")
    async def read(keys: list[K]) -> Mapping[K, V]:
        """Reads entries of the memory by key, leaving out missing ones.

        Args:
            keys: Keys of the entries.
        """
        with retry_on(Refused, Conflict, ValueError):
            return await memory.read(keys)

    @effect("write")
    async def write(changes: Changes[K, V]) -> Changes[K, V]:
        """Adds, updates, and removes entries of the memory, each with the reason for the change.

        Args:
            changes: The changes, applied together or not at all.
        """
        with retry_on(Refused, Conflict, ValueError):
            return await memory.write(changes)

    return _prefixed(name, {"search": search, "read": read, "write": write})


def _prefixed(name: str, tools: Mapping[str, FunctionType]) -> list[Tool]:
    """Names each tool `<name>_<suffix>` by its suffix."""
    for suffix, tool in tools.items():
        tool.__name__ = tool.__qualname__ = f"{name}_{suffix}"

    return list(tools.values())
