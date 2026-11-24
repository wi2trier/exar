"""OpenTelemetry capture of runs: the spans of a run, recorded in memory into `exar.typing.Span` records,
and the GenAI conventions that read their payload, such as `usage` and `input_messages`.

exar never replaces the tracer provider of the host, such as the one of Logfire.
`provider` is the global provider if it accepts span processors, as the SDK's and Logfire's do,
with the processor of exar added once, so the spans of a run reach the exporters of the host as well.
Without such a provider, it is a dedicated SDK provider of exar that is never registered globally,
so a host may still configure its own provider later, and exar picks it up for the following runs.
Spans that a head sampler drops are never recorded, so they are missing from the captured trace too,
and with them the steps of the trajectories of `exar.experience`.
exar relies on the spans on purpose, as the GenAI conventions are the same across agent frameworks,
so `provider` logs a warning once for an SDK provider whose sampler does not record every root span.
Sampling belongs in the collector instead, such as its tail sampling, after exar recorded the spans.
exar records spans in memory only for the traces a caller reads, see `exar.agents`.

Message content follows the upstream settings only: pydantic-ai records it unless its `InstrumentationSettings`
say `include_content=False`, such as `exar.pydantic_ai.instrumentation(include_content=False)`, and exar records the tasks and outputs of its own spans as messages
as the `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` environment variable
of `opentelemetry.util.genai` asks for.
A `Redact` function, such as one passed to `exar.agents.run`, rewrites every span record before it joins a trace,
exar ships no detector of personal data.

Agent frameworks emit their spans into `provider`, such as pydantic-ai through `exar.pydantic_ai.instrumentation`,
and exar opens the spans of the agents and workflows it invokes with the GenAI telemetry handler of `handler`,
which also records the duration metrics of the GenAI conventions into the global meter provider.
`capture` records every span that ends in the trace of a span while the block runs,
and `tree` turns the recorded spans below a span into its `exar.typing.Span` record.

Usage is not read from the spans while a run runs, framework adapters count it as each model call returns,
see `exar.agents.spend`, so neither sampling nor the content settings change what a run spent.
"""

import logging
import threading
import weakref
from collections import defaultdict
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, Protocol, runtime_checkable

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import Sampler
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    SystemInstructionPart,
)
from pydantic import ConfigDict, TypeAdapter
from pydantic_core import from_json, to_json

from .typing import JsonEntry, Redact, Span

__all__ = [
    "MODEL_CALLS",
    "NAME",
    "SCOPE",
    "Capture",
    "Operation",
    "Usage",
    "agent_name",
    "agents",
    "attribute",
    "capture",
    "error_type",
    "handler",
    "input_messages",
    "invoked",
    "operation",
    "output_messages",
    "provider",
    "system_instructions",
    "tool_call_arguments",
    "tool_call_result",
    "tool_name",
    "tree",
    "usage",
    "watch",
    "workflow_name",
]

_logger = logging.getLogger(__name__)

SCOPE = "exar"
"""Name of the instrumentation scope of the spans that exar opens."""


@runtime_checkable
class _Processing(Protocol):
    def add_span_processor(self, span_processor: SpanProcessor, /) -> None: ...


@dataclass(slots=True, eq=False)
class Capture:
    """The spans recorded for a trace while a `capture` block runs."""

    spans: list[ReadableSpan] = field(default_factory=list[ReadableSpan])
    trace_id: int | None = None


class _Recorder(SpanProcessor):
    """Hands every ended span to the captures of its trace."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._captures: defaultdict[int, list[Capture]] = defaultdict(list)

    def watch(self, capture: Capture, trace_id: int) -> None:
        with self._lock:
            capture.trace_id = trace_id
            self._captures[trace_id].append(capture)

    def forget(self, capture: Capture) -> None:
        with self._lock:
            if capture.trace_id is None:
                return

            captures = self._captures[capture.trace_id]
            captures.remove(capture)

            if not captures:
                del self._captures[capture.trace_id]

    def on_end(self, span: ReadableSpan) -> None:
        if span.context is None:
            return

        with self._lock:
            for capture in self._captures.get(span.context.trace_id, ()):
                capture.spans.append(span)


_RECORDER = _Recorder()
_OWN = TracerProvider()
_ATTACHED: weakref.WeakSet[_Processing] = weakref.WeakSet()


def provider() -> trace.TracerProvider:
    """The tracer provider that exar records runs from, see the module.

    Examples:
        >>> provider() is provider()
        True
    """
    current = trace.get_tracer_provider()

    if not isinstance(current, _Processing):
        current = _OWN

    if current not in _ATTACHED:
        current.add_span_processor(_RECORDER)
        _ATTACHED.add(current)

        sdk = _sdk_provider(current)

        if sdk is not None and not _records_all(sdk.sampler):
            _logger.warning(
                "The sampler %s of the tracer provider drops spans, which removes steps from the traces "
                "and experience of exar, sample in the collector instead, such as by tail sampling.",
                sdk.sampler.get_description(),
            )

    return current


def _sdk_provider(provider: object) -> TracerProvider | None:
    """The SDK provider a provider is or proxies through its `provider`, as Logfire's does, None for another one."""
    while not isinstance(provider, TracerProvider):
        provider = getattr(provider, "provider", None)

        if provider is None:
            return None

    return provider


def _records_all(sampler: Sampler) -> bool:
    """Whether a sampler records every root span, probed with the one trace id that only a ratio of one samples.

    Examples:
        >>> from opentelemetry.sdk.trace.sampling import ALWAYS_ON, DEFAULT_ON, TraceIdRatioBased
        >>> [_records_all(sampler) for sampler in (ALWAYS_ON, DEFAULT_ON, TraceIdRatioBased(0.99))]
        [True, True, False]
    """
    return sampler.should_sample(None, (1 << 128) - 1, "probe").decision.is_recording()


def handler() -> TelemetryHandler:
    """The GenAI telemetry handler of exar over `provider`, with the content capturing mode of the environment."""
    return TelemetryHandler(
        tracer_provider=provider(), instrumentation_scope_name=SCOPE
    )


@contextmanager
def capture() -> Generator[Capture]:
    """Records the spans that end in a trace while the block runs, once `watch` named the trace.

    Examples:
        >>> with capture() as recorded:
        ...     with handler().workflow("work") as invocation:
        ...         watch(recorded, invocation.span)
        >>> [span.name for span in recorded.spans]
        ['invoke_workflow work']
    """
    recorded = Capture()

    try:
        yield recorded
    finally:
        _RECORDER.forget(recorded)


def watch(recorded: Capture, span: trace.Span, /) -> None:
    """Records the spans of the trace of a span into a capture."""
    _RECORDER.watch(recorded, span.get_span_context().trace_id)


_STATUS: dict[StatusCode, Literal["unset", "ok", "error"]] = {
    StatusCode.UNSET: "unset",
    StatusCode.OK: "ok",
    StatusCode.ERROR: "error",
}


def _time(nanoseconds: int | None) -> datetime:
    return datetime.fromtimestamp((nanoseconds or 0) / 1e9, UTC)


def _value(value: object) -> JsonEntry:
    if isinstance(value, str | bool | int | float):
        return value

    if isinstance(value, Sequence):
        return [_value(item) for item in value]

    return str(value)


def tree(
    spans: Sequence[ReadableSpan], span_id: int, /, redact: Redact | None = None
) -> Span | None:
    """The record of a span with every span below it, None if the span was not recorded.

    Args:
        spans: The recorded spans.
        span_id: The id of the span.
        redact: Rewrites every span record, its children first.
    """
    by_id = {span.context.span_id: span for span in spans if span.context is not None}
    children: defaultdict[int, list[ReadableSpan]] = defaultdict(list)

    for span in by_id.values():
        if span.parent is not None:
            children[span.parent.span_id].append(span)

    def record(span: ReadableSpan, span_id: int) -> Span:
        recorded = Span(
            name=span.name,
            scope=span.instrumentation_scope.name if span.instrumentation_scope else "",
            span_id=f"{span_id:016x}",
            parent_id=None if span.parent is None else f"{span.parent.span_id:016x}",
            start=_time(span.start_time),
            end=_time(span.end_time),
            status=_STATUS[span.status.status_code],
            error=span.status.description,
            attributes={
                key: _value(value) for key, value in (span.attributes or {}).items()
            },
            children=tuple(
                record(child, child.context.span_id)
                for child in sorted(
                    children[span_id], key=lambda child: child.start_time or 0
                )
                if child.context is not None
            ),
        )

        return recorded if redact is None else redact(recorded)

    root = by_id.get(span_id)

    return None if root is None else record(root, span_id)


type Operation = Literal["invoke_agent", "invoke_workflow"]
"""The `gen_ai.operation.name` of the spans exar opens for the agents it invokes."""

NAME: Mapping[str, str] = {
    "invoke_agent": "gen_ai.agent.name",
    "invoke_workflow": "gen_ai.workflow.name",
}
"""The attribute naming what a span of an `Operation` invokes, by the GenAI conventions."""

MODEL_CALLS = frozenset({"chat", "text_completion", "generate_content", "embeddings"})
"""The `gen_ai.operation.name` of the model calls of the GenAI conventions, the spans whose usage counts.

Agent, workflow, tool, and retrieval spans carry no usage of their own,
and an invocation consumes what the model calls below it consumed.
"""

_BYTES = ConfigDict(val_json_bytes="base64", ser_json_bytes="base64")
_INPUT = TypeAdapter(list[InputMessage], config=_BYTES)
_OUTPUT = TypeAdapter(list[OutputMessage], config=_BYTES)
_SYSTEM = TypeAdapter(list[SystemInstructionPart], config=_BYTES)


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Resources consumed, by the `gen_ai.usage.*` attributes of the GenAI conventions.

    The cost adds up the priced model calls, as pydantic-ai adds it up,
    so a call of unknown cost adds nothing to it, and it is `None` while no call is priced,
    which keeps an unknown cost distinct from a free one.

    Args:
        input_tokens: `input_tokens`, including the cached tokens.
        output_tokens: `output_tokens`.
        cache_read_input_tokens: `cache_read.input_tokens`.
        cache_write_input_tokens: `cache_write.input_tokens`, or `cache_creation.input_tokens` as pydantic-ai writes it.
        reasoning_output_tokens: `reasoning.output_tokens`.
        requests: Model calls.
        cost: Price in US dollars of the priced model calls, `None` if none is priced.

    Examples:
        >>> Usage(input_tokens=3, requests=1, cost=Decimal("0.5")) + Usage(input_tokens=2, requests=1)
        Usage(input_tokens=5, output_tokens=0, cache_read_input_tokens=0, cache_write_input_tokens=0, reasoning_output_tokens=0, requests=2, cost=Decimal('0.5'))
        >>> (Usage(requests=1) + Usage(requests=1)).cost is None
        True
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    requests: int = 0
    cost: Decimal | None = None

    def __add__(self, other: Usage, /) -> Usage:
        counts = {name: getattr(self, name) + getattr(other, name) for name in _COUNTED}
        costs = [cost for cost in (self.cost, other.cost) if cost is not None]

        return Usage(**counts, cost=sum(costs, Decimal(0)) if costs else None)

    def attributes(self) -> dict[str, int | float]:
        """The usage as the attributes `exar.usage.*` of an invocation span, nothing without model calls.

        They mirror `gen_ai.usage.*` in a namespace of their own,
        since backends that add up `gen_ai.usage.*` over the spans of a trace would count them twice.
        The counts leave out zeros, and the cost is left out only while unknown, see `counted`.

        Examples:
            >>> Usage(input_tokens=3, cache_write_input_tokens=1, requests=1, cost=Decimal("0.5")).attributes()
            {'exar.usage.input_tokens': 3, 'exar.usage.cache_write.input_tokens': 1, 'exar.usage.requests': 1, 'exar.usage.cost': 0.5}
        """
        if not self.requests:
            return {}

        values: dict[str, int | float] = {
            f"exar.usage.{key}": getattr(self, name)
            for name, key in _COUNTED.items()
            if getattr(self, name)
        }

        if self.cost is not None:
            values["exar.usage.cost"] = float(self.cost)

        return values

    @classmethod
    def counted[V](cls, attributes: Mapping[str, V], /) -> Usage | None:
        """The usage an invocation span counted (`attributes`), `None` if it counted no model call.

        Examples:
            >>> Usage.counted({"exar.usage.input_tokens": 3, "exar.usage.requests": 1})
            Usage(input_tokens=3, output_tokens=0, cache_read_input_tokens=0, cache_write_input_tokens=0, reasoning_output_tokens=0, requests=1, cost=None)
            >>> Usage.counted({"gen_ai.usage.input_tokens": 3}) is None
            True
        """
        if "exar.usage.requests" not in attributes:
            return None

        counts = {
            name: _int(attributes.get(f"exar.usage.{key}"))
            for name, key in _COUNTED.items()
        }

        return cls(**counts, cost=_decimal(attributes.get("exar.usage.cost")))

    @classmethod
    def of[V](cls, attributes: Mapping[str, V], /) -> Usage:
        """The usage of a span by its attributes, nothing unless it is a model call (`MODEL_CALLS`).

        The cost is the attribute `operation.cost`, as pydantic-ai and Logfire write it.
        Without it, the cost is unknown.

        Examples:
            >>> Usage.of({"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": 3, "operation.cost": 0.5})
            Usage(input_tokens=3, output_tokens=0, cache_read_input_tokens=0, cache_write_input_tokens=0, reasoning_output_tokens=0, requests=1, cost=Decimal('0.5'))
            >>> Usage.of({"gen_ai.operation.name": "invoke_agent", "gen_ai.usage.input_tokens": 3}).input_tokens
            0
        """
        if attributes.get("gen_ai.operation.name") not in MODEL_CALLS:
            return cls()

        def count(keys: tuple[str, ...]) -> int:
            return next(
                (
                    n
                    for key in keys
                    if (n := _int(attributes.get(f"gen_ai.usage.{key}")))
                ),
                0,
            )

        return replace(
            cls(requests=1, cost=_decimal(attributes.get("operation.cost"))),
            **{name: count(keys) for name, keys in _KEYS.items()},
        )


_KEYS = {
    "input_tokens": ("input_tokens",),
    "output_tokens": ("output_tokens",),
    "cache_read_input_tokens": ("cache_read.input_tokens",),
    "cache_write_input_tokens": (
        "cache_write.input_tokens",
        "cache_creation.input_tokens",
    ),
    "reasoning_output_tokens": ("reasoning.output_tokens",),
}

_COUNTED = {name: keys[0] for name, keys in _KEYS.items()} | {"requests": "requests"}


def _number(value: object) -> int | float | None:
    """The value if it is a number, `None` otherwise, booleans included."""
    return (
        value
        if isinstance(value, int | float) and not isinstance(value, bool)
        else None
    )


def _int(value: object) -> int:
    return int(number) if (number := _number(value)) is not None else 0


def _decimal(value: object) -> Decimal | None:
    """A cost attribute as a decimal, by its shortest representation so a float like 0.1 stays 0.1."""
    return Decimal(str(number)) if (number := _number(value)) is not None else None


def _conform(messages: JsonEntry) -> JsonEntry:
    """Conforms the message parts of pydantic-ai to the GenAI conventions.

    pydantic-ai names reasoning parts `thinking` and the response of tool call responses `result`.
    """
    if not isinstance(messages, list):
        return messages

    for message in messages:
        parts = message.get("parts") if isinstance(message, dict) else None

        for part in parts if isinstance(parts, list) else ():
            if not isinstance(part, dict):
                continue

            if part.get("type") == "thinking":
                part["type"] = "reasoning"

            if part.get("type") == "tool_call_response" and "response" not in part:
                part["response"] = part.pop("result", None)

    return messages


def _text(span: Span, key: str) -> str | None:
    value = span.attributes.get(key)

    return value if isinstance(value, str) else None


def attribute(span: Span, key: str, /) -> JsonEntry:
    """An attribute of a span, with JSON strings parsed, None if missing."""
    value = span.attributes.get(key)

    if not isinstance(value, str):
        return value

    try:
        return from_json(value)
    except ValueError:
        return value


def operation(span: Span, /) -> str | None:
    """`gen_ai.operation.name`, such as `chat`, `execute_tool`, or `invoke_agent`."""
    return _text(span, "gen_ai.operation.name")


def agent_name(span: Span, /) -> str | None:
    """`gen_ai.agent.name`."""
    return _text(span, "gen_ai.agent.name")


def workflow_name(span: Span, /) -> str | None:
    """`gen_ai.workflow.name`."""
    return _text(span, "gen_ai.workflow.name")


def error_type(span: Span, /) -> str | None:
    """`error.type`, such as the type of the exception that failed the span."""
    return _text(span, "error.type")


def invoked(span: Span, /) -> str | None:
    """The name of the agent or workflow an `invoke_agent` or `invoke_workflow` span invokes, see `NAME`."""
    key = NAME.get(operation(span) or "")

    return None if key is None else _text(span, key)


def tool_name(span: Span, /) -> str | None:
    """`gen_ai.tool.name`."""
    return _text(span, "gen_ai.tool.name")


def tool_call_arguments(span: Span, /) -> JsonEntry:
    """`gen_ai.tool.call.arguments`, parsed."""
    return attribute(span, "gen_ai.tool.call.arguments")


def tool_call_result(span: Span, /) -> JsonEntry:
    """`gen_ai.tool.call.result`, parsed."""
    return attribute(span, "gen_ai.tool.call.result")


def input_messages(span: Span, /) -> list[InputMessage]:
    """`gen_ai.input.messages`, empty if missing."""
    return _INPUT.validate_json(
        to_json(_conform(attribute(span, "gen_ai.input.messages")) or [])
    )


def output_messages(span: Span, /) -> list[OutputMessage]:
    """`gen_ai.output.messages`, empty if missing."""
    return _OUTPUT.validate_json(
        to_json(_conform(attribute(span, "gen_ai.output.messages")) or [])
    )


def system_instructions(span: Span, /) -> list[SystemInstructionPart]:
    """`gen_ai.system_instructions`, empty if missing."""
    return _SYSTEM.validate_json(
        to_json(attribute(span, "gen_ai.system_instructions") or [])
    )


def usage(span: Span, /) -> Usage:
    """What the model calls of a span and every span below it consumed, see `Usage.of`.

    It is the usage an invocation counted (`Usage.counted`), else that of a model call itself
    and of every span below it.
    """
    if (counted := Usage.counted(span.attributes)) is not None:
        return counted

    return sum((usage(child) for child in span.children), Usage.of(span.attributes))


def agents(span: Span, /) -> tuple[str, ...]:
    """Names of the agents and workflows invoked at or below a span, in order of their first invocation."""
    return tuple(
        dict.fromkeys(name for child in span.walk() if (name := invoked(child)) is not None)
    )
