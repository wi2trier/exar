"""Memories that agents search, read, and write, see the protocol `exar.typing.Memory`.

- `InMemoryMemory`: Entries in process, searched by a text measure, such as the scratch memory of a run.
- `FileMemory`: Files of mounted folders (`Mount`) as `Document`s with frontmatter metadata, searched by per mount indexes.
- `GatedMemory`: Applies a policy to the writes of another memory.

A `GatedMemory` wraps any of them, `MemoryKind` is the union of its members and `AnyMemory` also admits any other `Memory`,
such as the memories over cbrkit storages of `exar.cbrkit`.
Memories without an index score their entries by an `exar.typing.Measure`, `word_coverage` by default, or any pair measure of cbrkit.

Short-term and long-term memory are two instances of the protocol,
such as a `FileMemory` over a scratch folder and one over shared folders.
Promoting knowledge is a write to the long-term memory,
whose metadata keeps its confidence and provenance and whose changes keep their reasons.
`memory_tools` exposes a memory to language models as plain async tools, and `exar.tools.files.file_tools` the files of a `FileMemory`.

Modules:
- `_common`: `digest`, `word_coverage`, and the ranking of searches.
- `_in_memory`: `InMemoryMemory` and `next_keys`, the default keys of its new entries.
- `_gated`: `GatedMemory`, `MemoryKind`, and `AnyMemory`.
- `_document`: `Document`.
- `mounts`: `Mount`, the folder that memories and file tools share, and the functions over a sequence of mounts.
- `_files`: `FileMemory`.
- `_tools`: `memory_tools`.

Examples:
    >>> import asyncio
    >>> memory = InMemoryMemory[str, str](entries={"a": "Paris is the capital of France", "b": "Berlin"})
    >>> [(hit.key, hit.score) for hit in asyncio.run(memory.search("capital of France"))]
    [('a', 1.0), ('b', 0.0)]
"""

from ._common import digest, word_coverage
from ._document import Document
from ._files import FileMemory
from ._gated import AnyMemory, GatedMemory, MemoryKind
from ._in_memory import InMemoryMemory, next_keys
from ._tools import memory_tools
from .mounts import Mount

__all__ = [
    "AnyMemory",
    "Document",
    "FileMemory",
    "GatedMemory",
    "InMemoryMemory",
    "MemoryKind",
    "Mount",
    "digest",
    "memory_tools",
    "next_keys",
    "word_coverage",
]
