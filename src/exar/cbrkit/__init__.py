"""cbrkit as the case-based reasoning of exar, needs the `cbrkit` extra.

- `StorageMemory`: Any cbrkit storage whose current cases cbrkit retrievers score, such as a case base on disk.
- `IndexMemory`: A cbrkit storage that searches its own text index, such as a vector database of typed rows.
- `Adapt`: A cbrkit adaptation, such as a `cbrkit.adapt.llm.Rewrite`, as an `exar.typing.Adaptation`.

The memories translate the changes, hits, conflicts, and refusals of cbrkit into those of exar,
so hosts catch the errors of `exar.errors` only.
They are configured in Python only, a `GatedMemory` wraps them like any other `exar.typing.Memory`.
Any pair measure of cbrkit, such as `cbrkit.sim.strings.Levenshtein()`, also scores an `exar.memory.InMemoryMemory`.
`exar.bpmn` and the storage of runs and reuse of processes of `exar.experience` build on cbrkit as well.
"""

from ._adapt import Adapt
from ._memory import IndexMemory, StorageMemory

__all__ = [
    "Adapt",
    "IndexMemory",
    "StorageMemory",
]
