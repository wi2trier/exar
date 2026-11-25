from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from pydantic import TypeAdapter

from exar import agents, otel
from exar.agents import FunctionAgent, Limits, RunConfig, call, run_async
from exar.errors import Refused
from exar.otel import Usage
from exar.policy import allow_all
from exar.typing import Span
from exar.workflows import Chain


async def test_capture_composes_with_the_host_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exported = InMemorySpanExporter()
    host = TracerProvider()
    host.add_span_processor(SimpleSpanProcessor(exported))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: host)

    with trace.get_tracer("host").start_as_current_span("request"):
        recorded = await run_async(
            Chain(agents=[FunctionAgent(func=len)]), "abc", RunConfig(policy=allow_all)
        )

    assert otel.provider() is host
    assert [span.name for span in recorded.walk()] == [
        "invoke_workflow Chain",
        "invoke_agent len",
    ]
    assert {span.name for span in exported.get_finished_spans()} == {
        "request",
        "invoke_workflow Chain",
        "invoke_agent len",
    }


def test_provider_warns_once_about_head_sampling(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    host = TracerProvider(sampler=TraceIdRatioBased(0.1))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: host)

    otel.provider()
    otel.provider()

    assert ["tail sampling" in record.message for record in caplog.records] == [True]


# pydantic-ai also records the messages and the output of a run on its agent span.
CONTENT = (
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "pydantic_ai.all_messages",
    "final_result",
)


def redact(span: Span) -> Span:
    return replace(
        span,
        attributes={k: v for k, v in span.attributes.items() if k not in CONTENT},
    )


async def test_redact_drops_content() -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from exar.eval.export import chat_messages
    from exar.pydantic_ai import PydanticAiAgent, instrumentation

    Agent.instrument_all(instrumentation())
    agent = Agent(TestModel(custom_output_text="secret"), instructions="Be terse.")
    recorded = await run_async(
        PydanticAiAgent(run=agent.run, name="agent"),
        "my secret",
        RunConfig(policy=allow_all, redact=redact),
    )
    [chat] = [span for span in recorded.walk() if otel.operation(span) == "chat"]

    assert recorded.output == "secret"
    assert (
        otel.input_messages(chat),
        otel.output_messages(chat),
        otel.system_instructions(chat),
    ) == ([], [], [])
    assert chat_messages(recorded) == []
    assert b"secret" not in TypeAdapter(Span).dump_json(recorded.root)


async def test_max_cost_counts_model_calls_as_they_return() -> None:
    cost: list[Decimal | None] = [Decimal(1)]

    async def spend(task: int) -> int:
        agents.spend(Usage(requests=1, cost=cost[0]))

        return task

    async def twice(task: int) -> int:
        return await call(spend, await call(spend, task))

    trace = await run_async(
        twice, 1, RunConfig(policy=allow_all, limits=Limits(max_cost=2.0))
    )

    assert otel.usage(trace.root).cost == 2

    with pytest.raises(Refused, match="The cost of 1 reached the maximum of 1.0"):
        await run_async(
            twice, 1, RunConfig(policy=allow_all, limits=Limits(max_cost=1.0))
        )

    # A model without a price, such as a self-hosted one, counts toward no cost limit.
    cost[0] = None
    unpriced = await run_async(
        twice, 1, RunConfig(policy=allow_all, limits=Limits(max_cost=1.0))
    )

    assert (otel.usage(unpriced.root).requests, otel.usage(unpriced.root).cost) == (2, None)


async def test_a_trace_is_built_only_once_it_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[int] = []

    def tree(
        spans: Sequence[ReadableSpan],
        span_id: int,
        /,
        redact: otel.Redact | None = None,
    ) -> Span | None:
        built.append(1)

        return otel.tree(spans, span_id, redact)

    monkeypatch.setattr(agents, "tree", tree)

    def double(x: int) -> int:
        captures.append(len(otel._RECORDER._captures))  # pyright: ignore[reportPrivateUsage]

        return 2 * x

    captures: list[int] = []

    assert await call(FunctionAgent(func=double), 2) == 4
    assert built == [] and captures == [0]

    trace = await run_async(FunctionAgent(func=double), 2, RunConfig(policy=allow_all))

    assert trace.output == 4 and otel.agent_name(trace.root) == "double"
    assert built == [1] and captures == [0, 1]
