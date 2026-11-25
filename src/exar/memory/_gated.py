"""A memory whose writes a policy checks."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field, GetPydanticSchema

from ..policy import AnyPolicy
from ..schema import Open, serializable
from ..typing import Changes, Hit, ListableMemory, Memory
from ._common import _authorize
from ._files import FileMemory
from ._in_memory import InMemoryMemory

__all__ = ["AnyMemory", "GatedMemory", "MemoryKind"]


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class GatedMemory[K, V]:
    """Applies a policy to every change written to a memory, searching, reading, and listing stay unchecked.

    Each change is an action of kind `"tool"` named `name` with effect `"write"` and its key as target,
    and the approver of the run answers the changes the policy asks about (`exar.agents.approve`).

    Args:
        memory: The gated memory, a `MemoryKind` of the spec or any `Memory` of Python code.
        policy: Decides on every change.
        name: Name of the memory in the actions, for policies by name.

    Examples:
        >>> import asyncio
        >>> from exar.typing import Add
        >>> from exar.memory import InMemoryMemory
        >>> from exar.policy import EffectPolicy
        >>> memory = GatedMemory(memory=InMemoryMemory[int, str](), policy=EffectPolicy(write="deny"))
        >>> asyncio.run(memory.keys())
        []
        >>> asyncio.run(memory.write(Changes(items=[Add(value="x", reason="note")])))
        Traceback (most recent call last):
        ...
        exar.errors.Refused: The policy denied tool 'memory'
    """

    kind: Literal["gated"] = field(default="gated", repr=False)
    # Python code sees a memory of its keys and values, serialized forms an `AnyMemory`.
    memory: Annotated[
        Memory[K, V], GetPydanticSchema(lambda _, handler: handler(AnyMemory))
    ]
    policy: AnyPolicy
    name: str = "memory"

    async def search(self, *queries: str, limit: int = 10) -> Sequence[Hit[K, V]]:
        """Searches the gated memory."""
        return await self.memory.search(*queries, limit=limit)

    async def read(self, keys: Collection[K], /) -> Mapping[K, V]:
        """Reads from the gated memory."""
        return await self.memory.read(keys)

    async def keys(self, /) -> Collection[K]:
        """Keys of all entries of the gated memory.

        Raises:
            TypeError: If the gated memory lists no keys, see `exar.typing.ListableMemory`.
        """
        if not isinstance(self.memory, ListableMemory):
            raise TypeError(f"{type(self.memory).__name__} lists no keys")

        return await self.memory.keys()

    async def write(
        self, changes: Changes[K, V], /, *, expected: Mapping[K, V] | None = None
    ) -> Changes[K, V]:
        """Writes the changes once the policy allows or the approver of the run grants each of them.

        Raises:
            exar.errors.Refused: If a change is refused, then nothing is written.
        """
        await _authorize(changes, self.policy, self.name)

        return await self.memory.write(changes, expected=expected)


type MemoryKind = Annotated[
    InMemoryMemory[Any, Any] | FileMemory | GatedMemory[Any, Any],
    Field(discriminator="kind"),
]
"""The memories of a spec, which a `GatedMemory` wraps."""

type AnyMemory = Open[MemoryKind, Memory[Any, Any]]
"""A `MemoryKind` or any other `Memory` of Python code, which serialized forms and JSON Schemas only know as `MemoryKind`."""
