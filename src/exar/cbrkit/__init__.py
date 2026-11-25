"""cbrkit as the case-based reasoning of exar, needs the `cbrkit` extra.

- `StorageMemory`: Any cbrkit storage whose current cases cbrkit retrievers score, such as a case base on disk.
- `IndexMemory`: A cbrkit storage that searches its own text index, such as a vector database of typed rows.
- `Adapt`: A cbrkit adaptation, such as a `cbrkit.adapt.llm.Rewrite`, as an `exar.typing.Adaptation`.
- `RetrievalMetric`: The ranking metrics of cbrkit as a pydantic-evals evaluator, also needs the `eval` extra.

The memories translate the changes, hits, conflicts, and refusals of cbrkit into those of exar,
so hosts catch the errors of `exar.errors` only.
They are configured in Python only, a `GatedMemory` wraps them like any other `exar.typing.Memory`.
Any pair measure of cbrkit, such as `cbrkit.sim.strings.Levenshtein()`, also scores an `exar.memory.InMemoryMemory`.
`exar.bpmn` and the storage of runs and reuse of processes of `exar.experience` build on cbrkit as well.
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_exports
from ._adapt import Adapt
from ._memory import IndexMemory, StorageMemory

if TYPE_CHECKING:
    from ._eval import RetrievalMetric

__all__ = [
    "Adapt",
    "IndexMemory",
    "RetrievalMetric",
    "StorageMemory",
]

lazy_exports(__name__)
