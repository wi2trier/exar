"""Running agents on tasks and recording every nested invocation into a trace.

An agent is anything async-callable from a task to its output, see `exar.typing.Agent`.
`run` and `run_async` run a root agent under a `RunConfig`, a policy, an approver of the host, limits, and a redaction,
and return the `exar.typing.Trace` of the run, captured from OpenTelemetry, see `exar.otel`.
`session` is the run of an agent that the host drives, such as a streamed run of an agent framework,
whose output the host sets.
Agents invoke other agents with `call`, or `traced` for the trace of the invocation,
which checks the invocation against the policy and the limits of the run
and opens an `invoke_agent` or `invoke_workflow` span below the current one.
Only a trace that is read records its spans in memory: `run`, `run_async`, `session`, and `traced` record them
once per trace at the outermost invocation that wants them, and the trace of a nested invocation is its part of them,
while `call` and a `session` without `record` only emit them to the tracer provider.
The spans of agent frameworks, such as the model calls and tool executions of pydantic-ai,
nest below the span of their invocation, since OpenTelemetry propagates its context.
The run itself travels in a context variable as well, so workflows compose plain agent objects.

The spans are opened by the GenAI telemetry handler of OpenTelemetry (`exar.otel.handler`),
an `invoke_agent` span with `gen_ai.agent.name` or an `invoke_workflow` span with `gen_ai.workflow.name`,
the task and the output as text in `gen_ai.input.messages` and `gen_ai.output.messages`
if the content capturing mode asks for it, and an error as the status and `error.type`.
An invocation consumed what the model calls below it consumed, nested invocations included,
which framework adapters count as each call returns (`spend`), such as `exar.pydantic_ai.CountUsage`,
independently of the spans, so sampling and content settings change nothing about it.
exar sets it as `exar.usage.*` on the span of the invocation (`exar.otel.Usage.attributes`) and checks `Limits.max_cost` against it.
`decide`, `approve`, and `depth` answer the policy questions of framework adapters with the policy and the approver of the run.
Outside a run there is no policy, so an agent invoked there runs, but nothing below it is decided on.

Examples:
    >>> from exar.policy import allow_all
    >>> double = FunctionAgent(func=lambda x: 2 * x)
    >>> trace = run(double, 21, RunConfig(policy=allow_all))
    >>> from exar.otel import agent_name, operation
    >>> trace.output, agent_name(trace.root), operation(trace.root)
    (42, '<lambda>', 'invoke_agent')
"""

import asyncio
import functools
import typing
from collections.abc import AsyncGenerator, Callable
from contextlib import ExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime

from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text
from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from ._effect import declared, effect_of
from .errors import AgentFailed, Refused
from .helpers import get_name
from .otel import NAME, SCOPE, Capture, Operation, Usage, capture, handler, tree, watch
from .policy import authorize
from .schema import serializable
from .text import as_text
from .typing import (
    Action,
    Agent,
    Approve,
    Decision,
    Effect,
    Policy,
    Redact,
    Span,
    Trace,
)

__all__ = [
    "FunctionAgent",
    "Invocation",
    "Limits",
    "RunConfig",
    "approve",
    "call",
    "decide",
    "depth",
    "name_of",
    "operation_of",
    "run",
    "run_async",
    "session",
    "spend",
    "traced",
    "usage",
]


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class Limits:
    """Bounds of a run.

    Args:
        max_depth: Deepest nesting of agent invocations, 1 for the agents the root agent invokes.
        max_cost: Total `cost` of the model calls of the run, as `spend` counts them,
            beyond which no further agent is invoked.
            It counts only the calls pydantic-ai or the span prices, so it never refuses a model without a price,
            such as a self-hosted one, which `max_depth` and the limits of the framework bound instead.
    """

    max_depth: int = 5
    max_cost: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """How a host runs agents: what it allows, who approves, and the bounds of a run.

    Args:
        policy: Checks the invocations of agents and, through framework adapters, the tool calls,
            required so that a host never runs without one, `exar.policy.allow_all` allows everything.
        approve: Answers the actions the policy asks about, which are refused without it.
        limits: Bounds of the run.
        redact: Rewrites every span record before it joins a trace of the run, see `exar.otel`.
    """

    policy: Policy
    approve: Approve | None = None
    limits: Limits = Limits()
    redact: Redact | None = None


@dataclass(slots=True, kw_only=True)
class _Run:
    config: RunConfig
    spent: Usage = field(default_factory=Usage)


@dataclass(slots=True, kw_only=True)
class _Frame:
    """An agent invocation while it runs, with what the model calls below it consumed so far.

    `recorded` holds the spans of its trace if it or an invocation enclosing it records them.
    """

    run: _Run
    depth: int
    parent: _Frame | None = None
    recorded: Capture | None = None
    usage: Usage = field(default_factory=Usage)


_FRAME: ContextVar[_Frame | None] = ContextVar("exar_frame", default=None)


def depth() -> int:
    """The nesting depth of the running agent invocation, 0 for the root agent and outside a run."""
    frame = _FRAME.get()

    return 0 if frame is None else max(frame.depth, 0)


def usage() -> Usage:
    """What the model calls below the running agent invocation consumed so far, nothing outside a run.

    Examples:
        >>> usage()
        Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_write_input_tokens=0, reasoning_output_tokens=0, requests=0, cost=None)
    """
    frame = _FRAME.get()

    return Usage() if frame is None else frame.usage


def spend(usage: Usage, /) -> None:
    """Counts the usage of a model call toward the running invocation, every invocation enclosing it, and the run.

    Framework adapters call it once for each model call, as `exar.pydantic_ai.CountUsage` does,
    and an agent that calls models on its own should too.
    Outside a run it counts nothing.

    Examples:
        >>> import asyncio
        >>> from exar.policy import allow_all
        >>> async def chat(task: str) -> str:
        ...     spend(Usage(input_tokens=3, requests=1))
        ...     return task
        >>> from exar.otel import usage
        >>> usage(run(chat, "Hi", RunConfig(policy=allow_all)).root).input_tokens
        3
    """
    frame = _FRAME.get()

    while frame is not None:
        frame.usage += usage

        # The frame of a run itself, see `session`, which a run nested in another one has too.
        if frame.depth < 0:
            frame.run.spent += usage

        frame = frame.parent


def _no_policy(action: Action, /) -> Decision:
    """The policy outside a run, which has none to decide with."""
    raise RuntimeError(
        f"No policy decides on {action.kind} {action.name!r} outside a run, see `run` and `session`"
    )


def decide(action: Action, /) -> Decision:
    """The decision of the policy of the current run.

    Raises:
        RuntimeError: Outside a run, which has no policy.

    Examples:
        >>> decide(Action(kind="tool", name="rm", effect="execute"))
        Traceback (most recent call last):
        RuntimeError: No policy decides on tool 'rm' outside a run, see `run` and `session`
    """
    frame = _FRAME.get()

    return (_no_policy if frame is None else frame.run.config.policy)(action)


async def approve(action: Action, /) -> bool:
    """Asks the approver of the current run, False if it has none."""
    frame = _FRAME.get()

    if frame is None or frame.run.config.approve is None:
        return False

    return await frame.run.config.approve(action)


def name_of(agent: object, /) -> str:
    """The name of an agent, its `name` attribute or else `exar.helpers.get_name`.

    Examples:
        >>> name_of(FunctionAgent(func=len))
        'len'
    """
    name = getattr(agent, "name", None)

    return name if isinstance(name, str) else get_name(agent)


_OPERATIONS: tuple[Operation, ...] = typing.get_args(Operation.__value__)


def operation_of(agent: object, /) -> Operation:
    """Whether an agent is invoked as an agent or a workflow, its `operation` attribute or else `"invoke_agent"`.

    Raises:
        ValueError: If the declared operation is unknown.

    Examples:
        >>> operation_of(FunctionAgent(func=len))
        'invoke_agent'
    """
    return declared(
        getattr(agent, "operation", "invoke_agent"), _OPERATIONS, "operation", agent
    )


async def _check(run: _Run, action: Action) -> None:
    config = run.config

    if action.depth > config.limits.max_depth:
        raise Refused(
            f"Invoking {action.name!r} exceeds the maximum depth of {config.limits.max_depth}"
        )

    maximum, cost = config.limits.max_cost, run.spent.cost

    if maximum is not None and cost is not None and cost >= maximum:
        raise Refused(f"The cost of {cost} reached the maximum of {maximum}")

    await authorize(config.policy(action), action, config.approve)


@dataclass(slots=True, kw_only=True)
class Invocation[I, O]:
    """An agent invocation while it runs, which the host completes in a `session` by setting its `output`.

    Args:
        task: The task.
    """

    task: I
    _output: tuple[O] | None = field(default=None, init=False, repr=False)
    _ended: Callable[[], tuple[Span, str]] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def output(self) -> O:
        """The output, which the host sets before the invocation ends.

        Raises:
            RuntimeError: If it is not set.
        """
        if self._output is None:
            raise RuntimeError("The invocation has no output")

        return self._output[0]

    @output.setter
    def output(self, output: O) -> None:
        self._output = (output,)

    @property
    def trace(self) -> Trace[I, O]:
        """The trace of the invocation once it ended, built from the recorded spans on first access.

        Raises:
            RuntimeError: If the invocation has not ended or has no output.
        """
        root, id = self._spans()

        return Trace(task=self.task, output=self.output, root=root, id=id)

    @property
    def failed(self) -> Trace[I, None]:
        """The trace of the invocation once it ended without an output, whose spans record the error."""
        root, id = self._spans()

        return Trace(task=self.task, output=None, root=root, id=id)

    def _spans(self) -> tuple[Span, str]:
        if self._ended is None:
            raise RuntimeError("The invocation has not ended")

        return self._ended()


@asynccontextmanager
async def _invoke[I, O](
    invocation: Invocation[I, O],
    name: str,
    operation: Operation,
    effect: Effect,
    record: bool,
) -> AsyncGenerator[Invocation[I, O]]:
    """Opens an invocation below the current one, checked against the policy and limits below the root.

    Its trace is available once the block ended, also if the block raised.

    With `record`, it records the spans of its trace unless an enclosing invocation records them already.
    """
    parent = _FRAME.get() or _Frame(
        run=_Run(config=RunConfig(policy=_no_policy)), depth=-1
    )
    frame = _Frame(
        run=parent.run, depth=parent.depth + 1, parent=parent, recorded=parent.recorded
    )

    if frame.depth > 0:
        await _check(
            frame.run,
            Action(
                kind="agent",
                name=name,
                effect=effect,
                depth=frame.depth,
                details=to_jsonable_python(invocation.task, fallback=str),
            ),
        )

    telemetry = handler()
    started = datetime.now(UTC)
    token = _FRAME.set(frame)

    try:
        with (
            ExitStack() as stack,
            telemetry.workflow(name)
            if operation == "invoke_workflow"
            else telemetry.invoke_local_agent(agent_name=name) as genai,
        ):
            if record and frame.recorded is None:
                frame.recorded = stack.enter_context(capture())
                watch(frame.recorded, genai.span)

            if genai.should_capture_content:
                genai.input_messages = [
                    InputMessage(role="user", parts=[Text(as_text(invocation.task))])
                ]

            try:
                yield invocation

                if genai.should_capture_content:
                    genai.output_messages = [
                        OutputMessage(
                            role="assistant", parts=[Text(as_text(invocation.output))]
                        )
                    ]
            finally:
                genai.attributes.update(frame.usage.attributes())
                context = genai.span.get_span_context()
                identity: dict[str, JsonValue] = {
                    "gen_ai.operation.name": operation,
                    NAME[operation]: name,
                }
                attributes = identity | frame.usage.attributes()
                recorded = frame.recorded
                ended = datetime.now(UTC)

                # Built only for the callers that read the trace, most read the output only.
                @functools.cache
                def spans() -> tuple[Span, str]:
                    # A span that a sampler dropped or nothing recorded is the root alone, with its usage.
                    found = () if recorded is None else recorded.spans
                    root = tree(
                        found, context.span_id, frame.run.config.redact
                    ) or Span(
                        name=f"{operation} {name}",
                        scope=SCOPE,
                        span_id=f"{context.span_id:016x}",
                        start=started,
                        end=ended,
                        attributes=attributes,
                    )

                    return root, f"{context.trace_id:032x}"

                invocation._ended = spans
    finally:
        _FRAME.reset(token)


async def traced[I, O](
    agent: Agent[I, O], task: I, /, *, name: str | None = None
) -> Trace[I, O]:
    """Invokes an agent below the current one and returns the trace of the invocation.

    Below the root of a run, the invocation is checked against the limits and the policy of the run,
    as an action with the effect of the agent (`exar.tools.effect_of`) and its task as JSON as details.
    The trace holds the spans below the invocation, which it records unless an enclosing invocation does.
    Outside a run, the agent is the root of an invocation without a policy, below which nothing is decided on.
    An agent with the class attribute `operation = "invoke_workflow"`, such as a workflow, is recorded as one (`operation_of`).

    Args:
        agent: The agent.
        task: The task.
        name: Name of the invocation, see `name_of` for the default.

    Raises:
        exar.errors.Refused: If a limit, the policy, or the approver refuses the invocation.
        exar.errors.AgentFailed: If the agent raises, with its partial trace and the error as its cause.
            Refusals and cancellation pass through unchanged, and so does every error of `call`.
        RuntimeError: If it is below an invocation outside a run, which has no policy.
    """
    invocation = Invocation[I, O](task=task)

    try:
        await _invoked(invocation, agent, name, record=True)
    except Refused:
        raise
    except Exception as error:
        # Raised before the invocation opened, such as by a refusing check, nothing ran to trace.
        if invocation._ended is None:
            raise

        raise AgentFailed(invocation.failed) from error

    return invocation.trace


async def _invoked[I, O](
    invocation: Invocation[I, O], agent: Agent[I, O], name: str | None, *, record: bool
) -> None:
    async with _invoke(
        invocation,
        name or name_of(agent),
        operation_of(agent),
        effect_of(agent),
        record,
    ):
        invocation.output = await agent(invocation.task)


async def call[I, O](agent: Agent[I, O], task: I, /, *, name: str | None = None) -> O:
    """Invokes an agent below the current one and returns its output, see `traced`, recording no spans in memory.

    Examples:
        >>> from exar.policy import allow_all
        >>> increment = FunctionAgent(func=lambda y: y + 1)
        >>> async def twice(x: int) -> int:
        ...     return await call(increment, await call(increment, x))
        >>> trace = run(twice, 1, RunConfig(policy=allow_all))
        >>> from exar.otel import agents
        >>> trace.output, agents(trace.root)
        (3, ('twice', '<lambda>'))
    """
    invocation = Invocation[I, O](task=task)
    await _invoked(invocation, agent, name, record=False)

    return invocation.output


@asynccontextmanager
async def session[I, O](
    task: I,
    config: RunConfig,
    /,
    *,
    name: str,
    operation: Operation = "invoke_agent",
    record: bool = True,
) -> AsyncGenerator[Invocation[I, O]]:
    """Runs an agent that the host drives, such as a streamed run of an agent framework, and traces it.

    The block is the root invocation of a run under a config, like `run_async`:
    agents invoked with `call` inside it are checked and recorded below it,
    and the spans of agent frameworks nest below it.
    The host sets the output of the invocation, and its trace is complete once the block ends,
    with the usage of the model calls below it, see the module.
    Without `record`, the run only emits its spans to the tracer provider, and its trace is the root alone with its usage.
    If the block raises, the invocation is recorded as failed with what its model calls consumed until then.

    Args:
        task: The task.
        config: The policy, approver, limits, and redaction of the run.
        name: Name of the root invocation.
        operation: Whether the host runs an agent or a workflow.
        record: Whether the trace holds the spans below the root, which it records in memory while the block runs.

    Examples:
        >>> import asyncio
        >>> from exar.policy import allow_all
        >>> async def stream(prompt: str) -> Trace[str, str]:
        ...     async with session(prompt, RunConfig(policy=allow_all), name="chat") as turn:
        ...         turn.output = "".join(chunk for chunk in ("Hel", "lo"))
        ...     return turn.trace
        >>> trace = asyncio.run(stream("Hi"))
        >>> trace.output, trace.root.name
        ('Hello', 'invoke_agent chat')
    """
    run = _Run(config=config)
    outer = _FRAME.get()
    token = _FRAME.set(
        _Frame(
            run=run,
            depth=-1,
            parent=outer,
            recorded=None if outer is None else outer.recorded,
        )
    )

    try:
        invocation = Invocation[I, O](task=task)

        async with _invoke(invocation, name, operation, "execute", record):
            yield invocation
    finally:
        _FRAME.reset(token)


async def run_async[I, O](
    agent: Agent[I, O],
    task: I,
    config: RunConfig,
    /,
    *,
    name: str | None = None,
) -> Trace[I, O]:
    """Runs an agent on a task and returns the trace of the run, see `session`.

    Args:
        agent: The root agent.
        task: The task.
        config: The policy, approver, limits, and redaction of the run.
        name: Name of the root invocation, see `name_of` for the default.

    Raises:
        exar.errors.Refused: If a limit, the policy, or the approver refuses an invocation the agent depends on.
    """
    async with session(
        task, config, name=name or name_of(agent), operation=operation_of(agent)
    ) as invocation:
        invocation.output = await agent(task)

    return invocation.trace


def run[I, O](
    agent: Agent[I, O],
    task: I,
    config: RunConfig,
    /,
    *,
    name: str | None = None,
) -> Trace[I, O]:
    """Runs an agent on a task with `asyncio.run` around `run_async`.

    Raises:
        RuntimeError: If an event loop runs in the current thread, use `run_async` there.
    """
    return asyncio.run(run_async(agent, task, config, name=name))


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionAgent[I, O]:
    """Adapts a synchronous function of the task into an agent, such as a rule or a cbrkit phase.

    The function runs in a worker thread so it never blocks other agents.
    Its effect is the one the function declares, see `exar.tools.effect`.
    An async function of the task is an agent already.

    Args:
        func: Function of the task.
    """

    func: Callable[[I], O]

    @property
    def name(self) -> str:
        """The name of the function."""
        return get_name(self.func)

    @property
    def effect(self) -> Effect:
        """The effect of the function."""
        return effect_of(self.func)

    async def __call__(self, task: I, /) -> O:
        return await asyncio.to_thread(self.func, task)
