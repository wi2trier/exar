"""Notes that tell a pydantic-ai model the state of its run while keeping the prompt cache.

State that changes between turns would rewrite the cached prefix if it were rendered into fresh instructions,
so a note appends its change to the history instead.
`RunNote` renders instructions once per run, which append a changed rendering, or rewrite the prefix as fixed text.
Text for the model is rendered by a `Render` from the run, which a host overrides to localize it, here and in the guards.
`ToolNote` reports state as tool calls the model did not make, with their results,
enqueued where the compaction window does not show them yet.
Instructions carry the authority of the operator, so state written by others,
such as a memory the model wrote or content of users, belongs in a tool result.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Self

from pydantic import JsonValue
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    InstructionChangePolicy,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    post_compaction_window,
)
from pydantic_ai.models import ModelRequestContext

from ..helpers import frozendict

__all__ = [
    "Read",
    "Reads",
    "Render",
    "RunNote",
    "ToolNote",
    "enqueue_reads",
    "latest_read",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Read:
    """A tool call reported to the model with its result.

    Args:
        tool: Name of the tool.
        content: Result of the call.
        args: Arguments of the call.
    """

    tool: str
    content: str
    args: Mapping[str, JsonValue] = field(default_factory=frozendict)


type Reads[D] = Callable[
    [RunContext[D], Sequence[ModelMessage]], Awaitable[Iterable[Read]]
]
"""Loads what to report from the run and the messages of its compaction window."""

type Render[D, *Args] = Callable[[RunContext[D], *Args], str]
"""Renders text for the model from the run and *Args*, which a host overrides to localize it."""


def latest_read(
    messages: Iterable[ModelMessage],
    tool: str,
    args: Mapping[str, JsonValue] = frozendict(),
    /,
) -> str | None:
    """The result of the latest call of a tool with these arguments that the messages answer.

    Examples:
        >>> call = ModelResponse(parts=[ToolCallPart("read_memory", {}, "1")])
        >>> result = ModelRequest(parts=[ToolReturnPart("read_memory", "- likes tea", "1")])
        >>> latest_read([call, result], "read_memory"), latest_read([call], "read_memory")
        ('- likes tea', None)
    """
    calls: set[str] = set()
    latest = None

    for message in messages:
        for part in message.parts:
            if (
                isinstance(part, ToolCallPart)
                and part.tool_name == tool
                and part.args_as_dict() == args
            ):
                calls.add(part.tool_call_id)

            elif isinstance(part, ToolReturnPart) and part.tool_call_id in calls:
                latest = part.model_response_str()

    return latest


def enqueue_reads[D](
    ctx: RunContext[D],
    window: Sequence[ModelMessage],
    reads: Iterable[Read],
    metadata: Mapping[str, JsonValue] | None = None,
) -> None:
    """Enqueues the *reads* whose result the compaction *window* does not show, as one response of tool calls and one request of their results."""
    unseen = [
        read
        for read in reads
        if latest_read(window, read.tool, read.args) != read.content
    ]

    if not unseen:
        return

    calls = [ToolCallPart(read.tool, dict(read.args)) for read in unseen]
    results = [
        ToolReturnPart(read.tool, read.content, call.tool_call_id)
        for read, call in zip(unseen, calls, strict=True)
    ]
    stamp = None if metadata is None else dict(metadata)
    ctx.enqueue(
        ModelResponse(parts=calls, metadata=stamp),
        ModelRequest(parts=results, metadata=stamp),
    )


@dataclass(kw_only=True)
class ToolNote[D](AbstractCapability[D]):
    """Reports reads before a model request whose compaction window shows other results or none.

    A read whose result the window shows already is skipped,
    the others are enqueued as one response of tool calls and one request of their results, which the history keeps.
    So the state a window started with stays in the cached prefix, and a change is appended.
    For a report once per window, `reads` returns nothing where `latest_read` finds a result.

    Args:
        reads: Loads what to report from the run and its compaction window.
        metadata: Metadata of the enqueued messages, such as a key a transcript hides them by.

    Examples:
        >>> import asyncio
        >>> from pydantic_ai import Agent
        >>> from pydantic_ai.models.test import TestModel
        >>> async def memory(ctx: RunContext[None], window: Sequence[ModelMessage]) -> list[Read]:
        ...     return [Read(tool="read_memory", content="- likes tea")]
        >>> agent = Agent(TestModel(call_tools=[]), capabilities=[ToolNote(reads=memory)])
        >>> result = asyncio.run(agent.run("Hi"))
        >>> [latest_read(result.all_messages(), "read_memory"), len(result.all_messages())]
        ['- likes tea', 4]
    """

    reads: Reads[D]
    metadata: Mapping[str, JsonValue] | None = None

    async def before_model_request(
        self, ctx: RunContext[D], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Enqueues the reads the window does not show."""
        window = post_compaction_window(ctx.messages)
        enqueue_reads(ctx, window, await self.reads(ctx, window), self.metadata)

        return request_context


@dataclass(kw_only=True)
class RunNote[D](AbstractCapability[D]):
    """Instructions rendered once per run that append their changes to the cached prefix.

    The state it renders is fixed for a run, so it is rendered once rather than before every request,
    in a worker thread if it blocks, such as by reading files,
    and an `on_change="append"` block extends the cached prefix where it changed between runs instead of rewriting it.
    An empty rendering contributes no block, and the `id` of the capability addresses the block, which it needs to append.
    With `on_change="rewrite"`, it is fixed text that rewrites the prefix when it changes,
    for what changes only along with the prefix, such as the tools a run may call,
    which pydantic-ai places with the other fixed text rather than behind the cache boundary as it would an instruction callable.

    Args:
        render: Renders the instructions from the run.
        blocking: Whether the rendering blocks, such as by reading files, so it runs in a worker thread.
        on_change: Whether a changed rendering appends to the history or rewrites the prefix.

    Examples:
        >>> import asyncio
        >>> from pydantic_ai import Agent
        >>> from pydantic_ai.models.test import TestModel
        >>> def mode(ctx: RunContext[str]) -> str:
        ...     return f"The run may {ctx.deps}."
        >>> agent = Agent(TestModel(), deps_type=str, capabilities=[RunNote[str](render=mode, id="mode")])
        >>> asyncio.run(agent.run("Hi", deps="read")).all_messages()[0].instructions
        'The run may read.'
    """

    render: Render[D]
    blocking: bool = False
    on_change: InstructionChangePolicy = "append"
    text: str = field(default="", init=False)

    async def for_run(self, ctx: RunContext[D]) -> Self:
        """Renders the instructions of the run."""
        note = replace(self)
        note.text = (
            await asyncio.to_thread(self.render, ctx)
            if self.blocking
            else self.render(ctx)
        )

        return note

    def get_instructions(self) -> InstructionPart | None:
        """The rendering as a block of the `on_change` policy, none while it is empty."""
        return (
            InstructionPart(self.text, on_change=self.on_change) if self.text else None
        )
