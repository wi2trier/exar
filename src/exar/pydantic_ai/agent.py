"""Runs of natively configured pydantic-ai agents as exar agents, with their usage and spans in the runs of exar.

A pydantic-ai agent is configured with the API of pydantic-ai, including the capabilities of pydantic-ai-harness
such as `SubAgents`, `TaskControl`, and `ApprovalReviewer`, or declaratively from an `AgentSpec`
with `pydantic_ai.Agent.from_file`.
`PydanticAiAgent` turns a function that runs it on a task, with every option of `Agent.run` native,
into an `exar.typing.Agent`, so workflows, processes, and `exar.agents.run` run it,
and `traced_events` traces a streamed run that the host drives, such as one a UI adapter streams.
Instrumentation is global: `Agent.instrument_all(instrumentation())` makes every run of pydantic-ai,
subagents included, emit its OpenTelemetry spans into `exar.otel.provider`,
which nest below the span of the invocation in the trace.
Usage is counted apart from the spans by the capability `CountUsage` of each agent, as every model request returns,
so it holds whether or not the spans are sampled or record content.
Approvals stay native: a tool that requires approval ends the run with `DeferredToolRequests`
if the output type allows it, and `EffectGate` derives the approvals from an `exar.policy` policy.
"""

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from functools import cache

from opentelemetry.trace import TracerProvider
from pydantic_ai import (
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    InstrumentationSettings,
    ModelResponse,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai_harness.spend_limits import PriceFunc

from exar.agents import Invocation, RunConfig, session, spend, usage
from exar.otel import Usage, provider
from exar.typing import Effect, Trace

__all__ = [
    "CountUsage",
    "Events",
    "PydanticAiAgent",
    "instrumentation",
    "traced_events",
    "usage_of",
]

_logger = logging.getLogger(__name__)

type Events[O] = AsyncIterable[AgentStreamEvent | AgentRunResultEvent[O]]
"""The events of a streamed run of pydantic-ai, ending with its result."""


@cache
def _instrumentation(
    provider: TracerProvider, include_content: bool, include_binary_content: bool
) -> InstrumentationSettings:
    return InstrumentationSettings(
        tracer_provider=provider,
        include_content=include_content,
        include_binary_content=include_binary_content,
        version=6,
    )


def instrumentation(
    *, include_content: bool = True, include_binary_content: bool = True
) -> InstrumentationSettings:
    """Instrumentation of pydantic-ai into the tracer provider of exar with the defaults of pydantic-ai.

    Apply it globally with `Agent.instrument_all(instrumentation())`, so the runs of every agent,
    subagents of pydantic-ai-harness included, reach the traces of exar.
    It uses version 6 of the data format of pydantic-ai, whose tool results have the role `tool`.
    Usage does not depend on it, see `CountUsage`.

    Args:
        include_content: Whether the spans record the messages, prompts, and tool arguments and results.
        include_binary_content: Whether the recorded content includes binary data, such as images, base64 encoded.
    """
    return _instrumentation(provider(), include_content, include_binary_content)


def usage_of(response: ModelResponse, /, price: PriceFunc | None = None) -> Usage:
    """The usage of one model request by its response, see `CountUsage`.

    Args:
        response: The response.
        price: Prices the response before pydantic-ai does, as for the `SpendLimits` of pydantic-ai-harness,
            where `None` falls through to the cost of pydantic-ai.

    Examples:
        >>> from decimal import Decimal
        >>> from pydantic_ai import RequestUsage
        >>> response = ModelResponse(parts=[], usage=RequestUsage(input_tokens=3), model_name="test")
        >>> usage_of(response)
        Usage(input_tokens=3, output_tokens=0, cache_read_input_tokens=0, cache_write_input_tokens=0, reasoning_output_tokens=0, requests=1, cost=None)
        >>> usage_of(response, price=lambda response: Decimal("0.5")).cost
        Decimal('0.5')
    """
    tokens = response.usage
    cost = None if price is None else price(response)

    return Usage(
        input_tokens=tokens.input_tokens,
        output_tokens=tokens.output_tokens,
        cache_read_input_tokens=tokens.cache_read_tokens,
        cache_write_input_tokens=tokens.cache_write_tokens,
        reasoning_output_tokens=tokens.output_reasoning_tokens,
        requests=1,
        cost=tokens.cost if cost is None else cost,
    )


@dataclass(kw_only=True)
class CountUsage[D](AbstractCapability[D]):
    """Counts the usage of every model request toward the exar run the agent runs in, see `exar.agents.spend`.

    It counts each response as it returns, independently of the spans, so sampling and content settings change nothing,
    and what `exar.agents.usage`, the `exar.usage.*` attributes, and `exar.agents.Limits.max_cost` see is complete.
    Add it once to every agent whose usage counts, subagents included, such as `Agent(..., capabilities=[CountUsage()])`,
    since a second one counts every request twice.
    The cost is what pydantic-ai prices the request at, the cost the Pydantic AI Gateway reports or the estimate of genai-prices,
    and unknown for a model it cannot price, such as a self-hosted one.
    A `price` prices a response first, with the `PriceFunc` of the `SpendLimits` of pydantic-ai-harness,
    such as by the cost OpenRouter bills in `response.provider_details["cost"]`,
    and its `None` falls through to the cost of pydantic-ai.

    Args:
        price: Prices a response before pydantic-ai does.

    Examples:
        >>> import asyncio
        >>> from pydantic_ai import Agent
        >>> from pydantic_ai.models.test import TestModel
        >>> from exar.agents import RunConfig, run_async
        >>> from exar.policy import allow_all
        >>> agent = Agent(TestModel(custom_output_text="Hi"), capabilities=[CountUsage()])
        >>> async def ask(task: str) -> str:
        ...     return (await agent.run(task)).output
        >>> trace = asyncio.run(run_async(ask, "Say hi", RunConfig(policy=allow_all)))
        >>> from exar.otel import usage
        >>> usage(trace.root).requests, usage(trace.root).cost
        (1, None)
    """

    price: PriceFunc | None = None

    async def after_model_request(
        self,
        ctx: RunContext[D],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Counts the usage of the response."""
        spend(usage_of(response, self.price))

        return response


def _check_counted(name: str, result: AgentRunResult[object]) -> None:
    """Logs a run whose model calls used tokens that no `CountUsage` counted."""
    used = any(
        isinstance(message, ModelResponse) and message.usage.total_tokens
        for message in result.new_messages()
    )

    if used and not usage().requests:
        _logger.debug(
            "The run of %r used tokens that no request counted, add `exar.pydantic_ai.CountUsage` to the agent",
            name,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PydanticAiAgent[I, O]:
    """A run of a pydantic-ai agent as an exar agent, whose output is the output of the run.

    The output is `DeferredToolRequests` if the run waits for approvals and the output type of the agent allows it.
    The usage is that of the model requests below the invocation, those of subagents running natively included,
    whether or not the run shares the `RunUsage` of a caller, as far as their agents count it with `CountUsage`.

    Args:
        run: Runs the agent on a task with every option of `Agent.run`, such as the deps, model,
            instructions, history, usage, usage limits, or conversation id.
        name: Name of the agent in traces.
        effect: What a run may change, which policies check when a workflow invokes it.

    Examples:
        >>> from pydantic_ai import Agent
        >>> from pydantic_ai.models.test import TestModel
        >>> from exar import agents
        >>> from exar.policy import allow_all
        >>> Agent.instrument_all(instrumentation())
        >>> greeter = Agent(TestModel(custom_output_text="Hi"), name="greeter", capabilities=[CountUsage()])
        >>> agent = PydanticAiAgent(run=lambda task: greeter.run(task, instructions="Be brief."), name="greeter")
        >>> trace = agents.run(agent, "Say hi", agents.RunConfig(policy=allow_all))
        >>> from exar.otel import operation, usage
        >>> trace.output, [operation(span) for span in trace.walk()], usage(trace.root).input_tokens
        ('Hi', ['invoke_agent', 'invoke_agent', 'chat'], 52)
    """

    run: Callable[[I], Awaitable[AgentRunResult[O]]]
    name: str
    effect: Effect = "execute"

    async def __call__(self, task: I, /) -> O:
        result = await self.run(task)
        _check_counted(self.name, result)

        return result.output


async def traced_events[I, O](
    events: Events[O] | AbstractAsyncContextManager[Events[O]],
    task: I,
    config: RunConfig,
    /,
    *,
    name: str,
    on_trace: Callable[[Trace[I, O]], object] | None = None,
) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[O]]:
    """Passes on the events of a streamed run as the root invocation of an exar run, see `exar.agents.session`.

    The session opens before the first event, so the run, which pydantic-ai starts on the first iteration,
    nests below it with its spans, and its output is the output of the final `AgentRunResultEvent`.
    By default it only traces: the spans reach the tracer provider and its exporters, but no trace is recorded in memory,
    and the usage of the run counts toward its limits only, so a host that needs it reads the `RunUsage` of the run.
    A host that wants the trace, such as to assess and retain the run as experience, passes `on_trace`,
    which records it and receives it once the run ended without an error.
    An exception of the run passes through after failing the invocation,
    so a UI adapter that iterates the events, such as `UIAdapter.transform_stream`, still reports it in-band.
    The run iterates the events in a task of its own, which buffers one event for the consumer,
    so its context, such as the current invocation and span, never leaks into the consumer,
    and closing or cancelling the iteration cancels the run.

    Args:
        events: The events, such as of `Agent.run_stream_events` or `UIAdapter.run_stream_native`,
            entered first if it is a context manager.
        task: The task, such as the prompt.
        config: The policy, approver, limits, and redaction of the run.
        name: Name of the root invocation.
        on_trace: Receives the trace of the run, which is recorded only if it is given.

    Examples:
        >>> import asyncio
        >>> from pydantic_ai import Agent
        >>> from pydantic_ai.models.test import TestModel
        >>> from exar.agents import RunConfig
        >>> from exar.policy import allow_all
        >>> agent = Agent(TestModel(custom_output_text="Hi"))
        >>> async def stream() -> list[str]:
        ...     run = agent.run_stream_events("Say hi")
        ...     return [type(event).__name__ async for event in traced_events(run, "Say hi", RunConfig(policy=allow_all), name="chat")]
        >>> asyncio.run(stream())[-1]
        'AgentRunResultEvent'
    """
    queue = asyncio.Queue[AgentStreamEvent | AgentRunResultEvent[O]](maxsize=1)

    async def produce() -> Invocation[I, O]:
        try:
            async with (
                session(
                    task, config, name=name, record=on_trace is not None
                ) as invocation,
                events
                if isinstance(events, AbstractAsyncContextManager)
                else nullcontext(events) as stream,
            ):
                async for event in stream:
                    if isinstance(event, AgentRunResultEvent):
                        invocation.output = event.result.output

                    await queue.put(event)
        finally:
            queue.shutdown()

        return invocation

    producer = asyncio.create_task(produce())

    try:
        while True:
            try:
                event = await queue.get()
            except asyncio.QueueShutDown:
                break

            yield event

        invocation = await producer
    finally:
        producer.cancel()
        _ = await asyncio.wait([producer])

        # An error of the run counts as retrieved, superseded by the one or the stop of the consumer.
        if not producer.cancelled():
            _ = producer.exception()

    if on_trace is not None:
        on_trace(invocation.trace)
