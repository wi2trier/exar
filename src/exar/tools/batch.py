"""Results of tools that serve a list argument item by item, as `pydantic_ai.ToolReturn`s of text and typed data.

`serve_batch` runs a tool on every distinct item with `exar.tools.run_batch`,
hands each item its `BatchShare` of the call's budgets, and presents the results side by side.
"""

from collections.abc import Awaitable, Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import Field
from pydantic.fields import FieldInfo
from pydantic_ai import ToolReturn
from pydantic_ai.messages import UserContent

from ..text import as_text
from . import BatchData, ItemFailure, render_batch, run_batch

__all__ = [
    "BatchShare",
    "ToolResult",
    "as_return",
    "batch_field",
    "serve_batch",
    "tool_return",
    "unpack",
]


def tool_return[T](
    data: T, text: str | None = None, files: Sequence[UserContent] = ()
) -> ToolReturn[T]:
    """The two channels of a tool's result: `text` and `files` for the model, `data` for programs and the client.

    Without `text`, the model reads `data` itself, a string as it is and anything else as JSON.
    Returning this is optional, a tool may return a plain value the model reads as it is.
    Either way every consumer reads a result with `unpack`, and the framework integrations of exar
    (`exar.pydantic_ai.toolset`, `exar.mcp.server`) turn a plain value into a `ToolReturn` through `as_return`,
    as pydantic-ai does, so hooks and history only ever see the rich form.

    Examples:
        >>> tool_return({"a": 1}).return_value, tool_return("x", "y").metadata
        ('{"a":1}', 'x')
    """
    value = as_text(data) if text is None else text

    return ToolReturn(return_value=[value, *files] if files else value, metadata=data)


def as_return(result: object, /) -> ToolReturn[object]:
    """A tool's result as a `ToolReturn`, a plain value being its `return_value` as pydantic-ai makes it.

    Examples:
        >>> as_return("x"), as_return(as_return("x")).return_value
        (ToolReturn(return_value='x'), 'x')
    """
    return result if isinstance(result, ToolReturn) else ToolReturn(result)


@dataclass(slots=True, frozen=True)
class ToolResult:
    """What a tool's result carries, read the same way whether the tool returned a plain value or a `ToolReturn`."""

    text: str
    """What the model reads first, a string as it is and any other value as JSON."""

    files: list[UserContent]
    """The items after the text, the files of the result and the notes of guards."""

    data: object
    """The typed data for programs and the client, the return value itself for a result without metadata."""


def unpack(result: object, /) -> ToolResult:
    """The text, files, and data of a tool's result, the counterpart of `float` for a `cbrkit` similarity.

    Examples:
        >>> unpack("x")
        ToolResult(text='x', files=[], data='x')
        >>> unpack(tool_return({"a": 1}, "one", ["note"]))
        ToolResult(text='one', files=['note'], data={'a': 1})
    """
    returned = as_return(result)
    files: list[UserContent]

    match returned.return_value:
        case str() as text:
            files = []

        case [str() as text, *rest] if isinstance(returned.return_value, list):
            files = cast(list[UserContent], rest)

        case value:
            text, files = as_text(value), []

    data = returned.return_value if returned.metadata is None else returned.metadata

    return ToolResult(text, files, data)


def batch_field(description: str, *, max_items: int | None = None) -> FieldInfo:
    """The `Field` of a list argument: `Annotated[list[X], batch_field(...)]`.

    A one-item list is how a single item is asked for, so there is no separate scalar variant of the argument and no empty list to answer.
    The list is uncapped unless `max_items` is given,
    since `exar.tools.BaseTool` caps the list arguments of its tool by its own `max_items`, while a plain function tool passes its cap here.
    """
    return Field(description=description, min_length=1, max_length=max_items)


@dataclass(slots=True, frozen=True)
class BatchShare:
    """Which item of how many a batch runner is serving.

    A batch spends one call's budgets, so each item takes its share of them rather than the whole:
    `of` splits any per-call budget evenly, the remainder going to the first items.
    """

    index: int
    count: int

    def of(self, total: int) -> int:
        """This item's share of `total`.

        Examples:
            >>> [BatchShare(i, 3).of(10) for i in range(3)]
            [4, 3, 3]
        """
        base, extra = divmod(total, self.count)

        return base + (self.index < extra)


async def serve_batch[I, R](
    items: Iterable[I],
    run: Callable[[I, BatchShare], Awaitable[ToolReturn[R]]],
    *,
    key: Callable[[I], str],
    concurrency: int = 1,
    identity: Callable[[R], Hashable] | None = None,
    shares: Callable[[I], bool] | None = None,
) -> ToolReturn[BatchData[R]]:
    """Serves every distinct item with `exar.tools.run_batch` and presents the results side by side.

    Each item is handed its `BatchShare` to split the call's budgets with,
    among the items `shares` picks when given, the others being handed them whole.
    `identity` names what makes two results the same, for items that only turn out to be
    duplicates once served, such as two URLs redirecting to one page.
    Each item is rendered under its key and the files are concatenated in order.

    Raises:
        exar.tools.Retry: If every item failed.
    """
    unique: dict[str, I] = {}

    for item in items:
        unique.setdefault(key(item), item)

    sharing = [name for name, item in unique.items() if shares is None or shares(item)]
    split = {
        name: BatchShare(index, len(sharing)) for index, name in enumerate(sharing)
    }

    async def serve(item: I) -> ToolReturn[R]:
        return await run(item, split.get(key(item), BatchShare(0, 1)))

    batch = dict(
        await run_batch(unique.values(), serve, key=key, concurrency=concurrency)
    )
    seen: dict[Hashable, str] = {}

    for name, outcome in batch.items():
        if isinstance(outcome, ToolReturn) and identity is not None:
            first = seen.setdefault(identity(outcome.metadata), name)

            if first != name:
                batch[name] = ItemFailure(item=name, reason=f"same result as {first}")

    return tool_return(
        tuple(
            outcome if isinstance(outcome, ItemFailure) else outcome.metadata
            for outcome in batch.values()
        ),
        render_batch(batch, lambda outcome: unpack(outcome).text),
        [
            file
            for outcome in batch.values()
            if isinstance(outcome, ToolReturn)
            for file in unpack(outcome).files
        ],
    )
