from collections.abc import Sequence
from decimal import Decimal

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from exar import agents, otel
from exar.agents import FunctionAgent, Limits, RunConfig, call, run_async
from exar.errors import Refused
from exar.otel import Usage
from exar.policy import allow_all
from exar.typing import Span


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
