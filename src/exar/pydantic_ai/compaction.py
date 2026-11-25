"""Overflow-triggered, prefix-cache-preserving compaction of pydantic-ai histories.

The summary is asked for as one more turn of the conversation it summarizes,
under the same instructions, tools, and model, so everything ahead of the prompt is the block
the provider just cached and only the prompt is prefilled.
The tools stay declared and calls are refused by a usage limit,
since withdrawing them would change the cached block,
and a model that calls one anyway is asked again with them withheld, see `run_summary`.
A transcript under other instructions, as `SummarizingCompaction` of pydantic-ai-harness renders,
shares nothing with the cache and pays a full prefill of a nearly full window.

Compaction runs only after the provider refused a request, never on a forecast,
since the window and its reserve are not reliably known for self-hosted endpoints.
The continuation is the larger request, so room is made by dropping messages from the tail,
which keeps the cached head.
Where to cut comes from `ModelResponse.usage`, the largest request the endpoint accepted being a floor
on its window, so no tokenizer or configured window is needed.
A cut lands only where every tool call is answered, since providers reject a history ending on an open one.
A refused summary sheds its output cap plus a prompt reserve and asks again.

The summary is stored as a response holding one provider-less `CompactionPart`, keeping every message,
and announced as a `Compacted` event in the event stream of the run, so a user interface shows it live.
Adapters that ignore compaction boundaries, such as `OpenAIChatModel` for vLLM, llama.cpp, and SGLang,
replay the whole history, so `CompactionWindow` sends only the window from the latest compaction on.
It is a workaround until pydantic-ai issue #7289 is resolved.
"""

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from typing import NamedTuple

from pydantic_ai import AgentRun, AgentRunResult, ModelRequestNode, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    PrepareTools,
)
from pydantic_ai.exceptions import (
    ModelContextWindowExceededError,
    ModelHTTPError,
    OutputTokenLimitExceeded,
)
from pydantic_ai.messages import (
    CapabilityEvent,
    CompactionPart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryFeedbackPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    post_compaction_window,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from .notes import Render

__all__ = [
    "COMPACTION",
    "COMPACT_PROMPT",
    "CompactOnOverflow",
    "Compacted",
    "CompactionWindow",
    "Overflow",
    "Summarizer",
    "SummaryFailed",
    "SummaryTurn",
    "compaction_cut",
    "compaction_window",
    "is_compaction",
    "is_context_overflow",
    "is_server_overflow",
    "run_summary",
    "summarize",
]

# Section structure adapted from pi-mono (MIT), https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/compaction/compaction.ts
COMPACT_PROMPT = """\
Summarize the conversation as a handover note for continuing the work in a fresh context.

Use this exact structure, keeping every heading even where a section is empty:

## Goal
[What the user is trying to accomplish, in one or two sentences.]

## Constraints and decisions
- [Preferences, requirements, and choices made, each with its reason, otherwise "(none)"]

## Progress
### Done
- [Finished work and established findings, otherwise "(none)"]
### In progress
- [Work underway or partially applied, otherwise "(none)"]
### Blocked
- [Blockers, failures, and unknowns, otherwise "(none)"]

## Files
- [Full path or name: why it matters, otherwise "(none)"]

## Next steps
1. [The immediate concrete action, otherwise "(none)"]

Keep every section to terse bullets rather than prose, and reproduce paths, quoted passages, names, and figures exactly as they appeared.
Answer from the conversation above alone, and do not call any tools.
Return only the summary, with no commentary around it."""
"""The request appended to the conversation as its last user turn.

Bounded by structure rather than a word count, which would make the model drop sections.
Its closing sentences keep the model from calling the tools that stay declared.
Write it in the language of the conversation, or the continuation switches language.
"""

SUMMARY_MAX_TOKENS = 8192
PROMPT_RESERVE_TOKENS = 1024
"""What a refused summary sheds on top of its output cap, covering the prompt and counting drift."""
MAX_ATTEMPTS = 3
"""Prefixes to offer before giving up, more would mean the reported counts bear no relation to the provider's."""
COMPACTION = "compaction"
"""Metadata key marking a summary run, which is never compacted itself."""

type Summarizer[D] = Callable[[RunContext[D], Sequence[ModelMessage]], Awaitable[str]]
"""Asks for the summary of a prefix of a running agent's history, raising `SummaryFailed` where its run failed."""

type Overflow = Callable[[Exception, Sequence[ModelMessage]], bool]
"""Whether an error of a run, with the messages the run ended on, means the request did not fit."""

type OpenRun[D] = Callable[
    [Sequence[AbstractCapability[D]]], AbstractAsyncContextManager[AgentRun[D, str]]
]
"""Opens a summary run with extra capabilities, such as by `Agent.iter`."""

# llama.cpp rejects an oversized prompt with this error type.
LLAMACPP_OVERFLOW_TYPE = "exceed_context_size_error"
# SGLang's wordings, whose 400 body carries no stable identifier,
# the second is the KV pool bound below the context length.
SGLANG_OVERFLOW_PHRASES = (
    "the model's context length",
    "exceeds the maximum allowed length",
)


def is_context_overflow(error: Exception, messages: Sequence[ModelMessage], /) -> bool:
    """Whether an error of pydantic-ai, with the messages its run ended on, means the request did not fit the context window.

    An exceeded output limit counts only where the window left no room for output,
    which an endpoint answers with a response cut for length before it generated anything,
    while a response that thought, wrote, or called a tool spent its cap instead.

    Examples:
        >>> from pydantic_ai.messages import ThinkingPart
        >>> is_context_overflow(ModelContextWindowExceededError("m", "too long"), []), is_context_overflow(ValueError(), [])
        (True, False)
        >>> cut = OutputTokenLimitExceeded("cut")
        >>> is_context_overflow(cut, [ModelResponse(parts=[], finish_reason="length")])
        True
        >>> is_context_overflow(cut, [ModelResponse(parts=[ThinkingPart("hm")], finish_reason="length")])
        False
    """
    match error, messages:
        case ModelContextWindowExceededError(), _:
            return True
        case OutputTokenLimitExceeded(), [*_, ModelResponse() as response]:
            return (
                response.finish_reason == "length"
                and not response.usage.output_tokens
                and all(
                    isinstance(part, TextPart) and not part.content
                    for part in response.parts
                )
            )
        case _:
            return False


def is_server_overflow(error: Exception, /) -> bool:
    """Whether an error is an overflow that pydantic-ai does not recognize, from llama.cpp or SGLang.

    Combine it with `is_context_overflow` for self-hosted endpoints.

    Examples:
        >>> is_server_overflow(ModelHTTPError(400, "m", provider_error_type=LLAMACPP_OVERFLOW_TYPE))
        True
    """
    match error:
        case ModelHTTPError(status_code=400, provider_error_type=kind, body=body):
            return kind == LLAMACPP_OVERFLOW_TYPE or any(
                phrase in str(body) for phrase in SGLANG_OVERFLOW_PHRASES
            )
        case _:
            return False


class _Cut(NamedTuple):
    keep: int
    size: int


def _is_retry(part: ModelRequestPart | ModelResponsePart) -> bool:
    """Whether a part asks the model to retry, the legacy `RetryPromptPart` included.

    One built in memory or handed back as a deferred result can still be in a history,
    and it is told by its kind, since its class is deprecated.
    """
    return isinstance(part, RetryFeedbackPart) or part.part_kind == "retry-prompt"


def _called_tool(messages: Sequence[ModelMessage]) -> bool:
    """Whether the last response calls a tool."""
    response = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, ModelResponse)
        ),
        None,
    )

    return response is not None and any(
        isinstance(part, ToolCallPart) for part in response.parts
    )


def _answered_call(part: ModelRequestPart | ModelResponsePart) -> str | None:
    """The tool call a part answers, as pydantic-ai pairs them, `None` for a part that answers none.

    A legacy `RetryPromptPart` bound to a tool answers its call like a `ToolReturnPart`.
    """
    if isinstance(part, ToolReturnPart) or (
        part.part_kind == "retry-prompt" and part.tool_name is not None
    ):
        return part.tool_call_id

    return None


def _cuts(messages: Sequence[ModelMessage]) -> list[_Cut]:
    """The prefixes that can be closed off, ascending, with the tokens they measured when last sent.

    A prefix closes where every tool call is answered and it does not end on a fresh prompt,
    which a summary would drop.
    Input plus output of a response is what the following request sent,
    so a prefix ending in tool returns understates by those returns, which a refusal then corrects.
    """
    cuts: list[_Cut] = []
    open_calls: set[str] = set()
    size = 0

    for keep, message in enumerate(messages, start=1):
        answered = False

        for part in message.parts:
            if isinstance(part, ToolCallPart):
                open_calls.add(part.tool_call_id)
            elif (answer := _answered_call(part)) is not None:
                open_calls.discard(answer)
                answered = True
            elif _is_retry(part):
                answered = True

        if isinstance(message, ModelResponse):
            size = message.usage.total_tokens or size

        if (answered or isinstance(message, ModelResponse)) and not open_calls:
            cuts.append(_Cut(keep, size))

    return cuts


def _plan(messages: Sequence[ModelMessage], max_tokens: int) -> list[int]:
    """How many leading messages each attempt keeps, longest first.

    The first attempt keeps everything the endpoint has served, since an overflow is refused
    before anything is prefilled, and each refusal sheds what the request needs.
    """
    cuts = [cut for cut in _cuts(messages) if cut.size] or [_Cut(len(messages), 0)]
    plan = [cuts[-1].keep]
    budget = cuts[-1].size
    reserve = max_tokens + PROMPT_RESERVE_TOKENS

    for _ in range(MAX_ATTEMPTS - 1):
        smaller = next((c for c in reversed(cuts) if c.size + reserve <= budget), None)

        if smaller is None or smaller.keep >= plan[-1]:
            break

        plan.append(smaller.keep)
        budget = smaller.size

    return plan


@dataclass
class SummaryFailed(Exception):
    """A summary run that failed, with the messages it ended on, which tell whether it overflowed."""

    error: Exception
    messages: Sequence[ModelMessage]


def _withhold_tools[D](
    ctx: RunContext[D], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    return []


async def run_summary[D](open_run: OpenRun[D], /) -> AgentRunResult[str]:
    """Runs a summary turn, and once more with the tools withheld where it called one.

    The tools stay declared for the cached prefix, so a run that calls one anyway
    is refused by its `tool_calls_limit` of zero and pays a full prefill once.

    Raises:
        SummaryFailed: The error of the run, with the messages it ended on.
    """
    for capabilities in ((), (PrepareTools[D](_withhold_tools),)):
        failed: SummaryFailed | None = None

        # Raised outside the run, whose exit would replace the error with the one it caught.
        async with open_run(capabilities) as run:
            try:
                async for _ in run:
                    pass
            except Exception as error:  # noqa: BLE001, judged below with the messages of the run
                failed = SummaryFailed(error, run.all_messages())

        if failed is not None and (capabilities or not _called_tool(failed.messages)):
            raise failed from failed.error

        if run.result is not None:
            return run.result

    raise RuntimeError("The summary run ended without a result")


async def summarize(
    messages: Sequence[ModelMessage],
    ask: Callable[[Sequence[ModelMessage]], Awaitable[str]],
    /,
    *,
    max_tokens: int = SUMMARY_MAX_TOKENS,
    is_overflow: Overflow = is_context_overflow,
) -> str:
    """Summarizes a conversation, shedding messages from its tail while the summary overflows.

    Args:
        messages: The conversation.
        ask: Asks for the summary as the next turn of a prefix of the conversation,
            raising `SummaryFailed` where its run failed, as `run_summary` does.
        max_tokens: Output cap of the summary.
        is_overflow: Whether an error of `ask`, with the messages its run ended on, means the request did not fit.

    Raises:
        Exception: The error of the last attempt, or of any attempt that did not overflow.
    """
    *retries, final = _plan(messages, max_tokens)

    for cut in retries:
        try:
            return await ask(messages[:cut])
        except SummaryFailed as failed:
            if not is_overflow(failed.error, failed.messages):
                raise failed.error from None
        # Any error can mean an overflow, which `is_overflow` decides.
        except Exception as error:
            if not is_overflow(error, ()):
                raise

    try:
        return await ask(messages[:final])
    except SummaryFailed as failed:
        raise failed.error from None


def is_compaction(message: ModelMessage, /) -> bool:
    """Whether a message is a compaction, a response holding a `CompactionPart`."""
    return isinstance(message, ModelResponse) and any(
        isinstance(part, CompactionPart) for part in message.parts
    )


def _is_prompt(message: ModelMessage) -> bool:
    parts = message.parts if isinstance(message, ModelRequest) else ()

    return any(isinstance(part, UserPromptPart) for part in parts) and not any(
        isinstance(part, ToolReturnPart) or _is_retry(part) for part in parts
    )


def _boundaries(messages: Sequence[ModelMessage]) -> list[int]:
    """Where a compaction can go, ascending.

    Not right after the latest compaction, which leaves nothing new to summarize,
    and not past the latest prompt before a model answered it.
    """
    start = len(messages) - len(post_compaction_window(messages))
    prompt = max(
        (i for i, message in enumerate(messages) if _is_prompt(message)), default=-1
    )
    answered = next(
        (
            i
            for i, message in enumerate(messages)
            if i > prompt and isinstance(message, ModelResponse) and message.model_name
        ),
        len(messages),
    )

    return [
        cut.keep
        for cut in _cuts(messages)
        if cut.keep > start + 1 and (cut.keep <= prompt or cut.keep > answered)
    ]


def compaction_cut(messages: Sequence[ModelMessage], /) -> int | None:
    """Where a compaction of a history goes, `None` where none can.

    It goes after the history where that closes it, or ahead of a trailing prompt,
    which is never summarized and follows the summary.

    Examples:
        >>> from pydantic_ai.messages import TextPart
        >>> ask = ModelRequest(parts=[UserPromptPart("hi")])
        >>> answer = ModelResponse(parts=[TextPart("hello")], model_name="m")
        >>> compaction_cut([ask, answer]), compaction_cut([ask, answer, ask]), compaction_cut([ask])
        (2, 2, None)
    """
    boundaries = _boundaries(messages)

    if len(messages) in boundaries:
        return len(messages)

    if (
        messages
        and isinstance(messages[-1], ModelRequest)
        and len(messages) - 1 in boundaries
    ):
        return len(messages) - 1

    return None


def _compact_prompt[D](ctx: RunContext[D], /) -> str:
    return COMPACT_PROMPT


@dataclass(frozen=True, slots=True, kw_only=True)
class SummaryTurn[D]:
    """Asks the running agent for the summary as the next turn of its own conversation.

    It reuses the agent, dependencies, and model of the run, not instructions, toolsets,
    or capabilities passed to the run alone, which a custom `Summarizer` reproduces.
    The summary's usage is added to the run's.

    Args:
        prompt: Renders the request for the summary from the run, in the language of its conversation.
        model_settings: Settings of the summary request, which leave the cached prefix intact.
    """

    prompt: Render[D] = _compact_prompt
    model_settings: ModelSettings = field(
        default_factory=lambda: ModelSettings(
            max_tokens=SUMMARY_MAX_TOKENS, thinking=False
        )
    )

    async def __call__(
        self, ctx: RunContext[D], messages: Sequence[ModelMessage], /
    ) -> str:
        agent = ctx.agent

        if agent is None:
            raise RuntimeError("The summary turn needs the agent of the run")

        prompt = self.prompt(ctx)
        result = await run_summary(
            lambda capabilities: agent.iter(
                prompt,
                output_type=str,
                message_history=messages,
                deps=ctx.deps,
                model=ctx.model if isinstance(ctx.model, Model) else None,
                model_settings=self.model_settings,
                usage_limits=UsageLimits(request_limit=1, tool_calls_limit=0),
                metadata={COMPACTION: True},
                capabilities=capabilities,
            )
        )
        ctx.usage.incr(result.usage)

        return result.output.strip()


@dataclass(kw_only=True)
class Compacted(CapabilityEvent, namespace="exar"):
    """A run compacted its history into the summary of `part`."""

    part: CompactionPart


def _retry(request: ModelRequest) -> ModelRequest:
    """A refused request without its instruction state, which the retry after a compaction records afresh."""
    return replace(
        request,
        parts=[
            part
            for part in request.parts
            if not isinstance(part, InstructionDeltaPart | InstructionBaselinePart)
        ],
    )


@dataclass(kw_only=True)
class CompactOnOverflow[D](AbstractCapability[D]):
    """Compacts a history the context window refused, then retries the request.

    The compaction goes at `compaction_cut`.
    After refused tool returns, the retry is an empty request continuing from the summary,
    ahead of a refused prompt, the retry sends the prompt again.
    A summary run, marked by the metadata key `"compaction"`, is never compacted itself.
    Each compaction is announced as a `Compacted` event.

    Args:
        summarizer: Asks for the summary of a prefix of the history.
        max_tokens: Output cap of the summary, which a refused summary sheds.
        is_overflow: Whether an error, with the messages of the run, means the request did not fit,
            such as `lambda e, m: is_context_overflow(e, m) or is_server_overflow(e)` for self-hosted endpoints.
    """

    summarizer: Summarizer[D] = field(default_factory=SummaryTurn[D])
    max_tokens: int = SUMMARY_MAX_TOKENS
    is_overflow: Overflow = is_context_overflow

    async def on_node_run_error(
        self, ctx: RunContext[D], *, node: AgentNode[D], error: Exception
    ) -> NodeResult[D]:
        """Compacts and retries an overflowing model request, re-raises anything else."""
        if (
            not isinstance(node, ModelRequestNode)
            or not self.is_overflow(error, ctx.messages)
            or (ctx.metadata or {}).get(COMPACTION)
        ):
            raise error

        history = list(ctx.messages)

        # A stream refused on opening leaves an empty response behind.
        if history and isinstance(history[-1], ModelResponse) and not history[-1].parts:
            history.pop()

        cut = compaction_cut(history)

        if cut is None or not isinstance(refused := history[-1], ModelRequest):
            raise error

        async def ask(messages: Sequence[ModelMessage]) -> str:
            return await self.summarizer(ctx, messages)

        summary = await summarize(
            history[:cut], ask, max_tokens=self.max_tokens, is_overflow=self.is_overflow
        )
        part = CompactionPart(content=summary)
        compaction = ModelResponse(parts=[part])

        if cut < len(history):
            # The retry appends the refused request again.
            history[-1] = compaction
            retry = _retry(refused)
        else:
            history.append(compaction)
            retry = ModelRequest(parts=[])

        ctx.messages[:] = history
        await ctx.emit(Compacted(part=part))

        return replace(node, request=retry)


def compaction_window(messages: Sequence[ModelMessage], /) -> list[ModelMessage]:
    """The messages from the latest compaction on, its summary sent as a request.

    A provider's own compaction is left to its adapter.

    Examples:
        >>> from pydantic_ai.messages import TextPart
        >>> old = ModelRequest(parts=[UserPromptPart("old")])
        >>> summary = ModelResponse(parts=[CompactionPart(content="sum")])
        >>> new = ModelRequest(parts=[UserPromptPart("new")])
        >>> [type(m).__name__ for m in compaction_window([old, summary, new])]
        ['ModelRequest', 'ModelRequest']
    """
    match window := post_compaction_window(messages):
        case [
            ModelResponse(
                parts=[CompactionPart(provider_name=None) as summary, *rest]
            ) as head,
            *tail,
        ]:
            kept = [replace(head, parts=rest)] if rest else []

            return [ModelRequest(parts=[summary]), *kept, *tail]
        case _:
            return window


@dataclass(kw_only=True)
class CompactionWindow[D](AbstractCapability[D]):
    """Sends each request the window from the latest compaction on, keeping the stored history whole.

    A workaround until pydantic-ai honors compaction boundaries for every adapter (issue #7289).
    """

    async def before_model_request(
        self, ctx: RunContext[D], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Replaces the messages of the request with their compaction window."""
        request_context.messages = compaction_window(request_context.messages)

        return request_context
