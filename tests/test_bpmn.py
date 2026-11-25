import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from cbrkit.model.graph import BpmnGraph, from_bpmn, parse_bpmn
from pydantic import BaseModel, ConfigDict

from exar.agents import FunctionAgent
from exar.bpmn import BpmnProcess, validate
from exar.typing import Agent

DATA = Path(__file__).parent / "data"


class State(BaseModel):
    """The state of every test process."""

    model_config = ConfigDict(frozen=True)

    task: str = ""
    score: float = 0.0
    published: bool = False
    verdict: str | None = None
    web: int = 0
    papers: int = 0
    summary: int = 0
    wants_mail: bool = False
    wants_chat: bool = False
    mail: bool = False
    chat: bool = False
    log: bool = False
    done: bool = False
    revisions: int = 0
    needs_revision: bool = False
    slow: bool = False
    error: str | None = None
    recovered: bool = False
    order: int = 0
    confirmed: bool = False
    ok: bool = False


type Handler = Callable[[State], Awaitable[object]]


@dataclass(slots=True, kw_only=True)
class FakeContext:
    """Records the invoked agents and answers sent messages with the given replies."""

    agents: Mapping[str, Handler] = field(default_factory=dict)
    replies: Mapping[str, object] = field(default_factory=dict[str, object])
    sent: list[tuple[str | None, object]] = field(
        default_factory=list[tuple[str | None, object]]
    )
    calls: list[str] = field(default_factory=list)
    inbox: asyncio.Queue[tuple[str, object]] = field(default_factory=asyncio.Queue)

    async def send(self, to: str | None, content: object, /) -> None:
        self.sent.append((to, content))

        if to is not None and to in self.replies:
            self.inbox.put_nowait((to, self.replies[to]))

    async def receive(self, /, *, timeout: float | None = None) -> tuple[str, object]:
        async with asyncio.timeout(timeout):
            return await self.inbox.get()

    def agent(self, name: str) -> Agent[State, object]:
        async def invoke(state: State, /) -> object:
            self.calls.append(name)

            return await self.agents[name](state)

        return invoke


def returns(value: object, delay: float = 0) -> Handler:
    async def handler(state: State) -> object:
        await asyncio.sleep(delay)

        return value

    return handler


async def fails(state: State) -> object:
    raise RuntimeError("model unavailable")


def load(name: str) -> BpmnGraph:
    return from_bpmn(parse_bpmn((DATA / f"{name}.bpmn").read_bytes()))


async def run(
    name: str,
    ctx: FakeContext,
    state: State | None = None,
    *,
    functions: Mapping[str, Callable[..., object]] | None = None,
    outputs: Mapping[str, str] | None = None,
    error_field: str | None = None,
) -> State:
    graph = load(name)
    process = BpmnProcess[State](
        graph=graph,
        agents={agent: ctx.agent(agent) for agent in ctx.agents},
        functions=functions or {},
        channel=ctx,
        error_field=error_field,
        outputs=outputs or {},
    )

    return await process(state or State())


@pytest.mark.parametrize(
    ("score", "calls", "verdict"),
    [(0.9, ["scorer", "publisher"], None), (0.2, ["scorer", "rejecter"], "no")],
)
async def test_exclusive(score: float, calls: list[str], verdict: str | None) -> None:
    ctx = FakeContext(
        agents={
            "scorer": returns({"score": score}),
            "publisher": returns({"published": True}),
            "rejecter": returns("no"),
        }
    )
    state = await run(
        "exclusive", ctx, State(task="Review the paper"), outputs={"reject": "verdict"}
    )

    assert ctx.calls == calls
    assert state.task == "Review the paper"
    assert state.verdict == verdict


async def test_parallel() -> None:
    async def combine(state: State) -> object:
        return state.web + state.papers

    ctx = FakeContext(
        agents={
            "web": returns({"web": 1}, delay=0.02),
            "papers": returns({"papers": 2}),
            "combine": combine,
        }
    )
    state = await run("parallel", ctx, outputs={"combine": "summary"})

    assert state.summary == 3
    assert ctx.calls.count("combine") == 1


@pytest.mark.parametrize(
    ("content", "called"),
    [
        (State(wants_mail=True, wants_chat=True), {"mail", "chat"}),
        (State(wants_chat=True), {"chat"}),
        (State(), {"log"}),
    ],
)
async def test_inclusive(content: State, called: set[str]) -> None:
    ctx = FakeContext(
        agents={
            "mail": returns({"mail": True}, delay=0.02),
            "chat": returns({"chat": True}),
            "log": returns({"log": True}),
            "done": returns({"done": True}),
        }
    )
    state = await run("inclusive", ctx, content)

    assert set(ctx.calls[:-1]) == called
    assert ctx.calls[-1] == "done"
    assert all(getattr(state, name) for name in called)


async def test_loop() -> None:
    async def draft(state: State) -> object:
        revisions = state.revisions + 1

        return {"revisions": revisions, "needs_revision": revisions < 3}

    ctx = FakeContext(agents={"drafter": draft})
    state = await run("loop", ctx)

    assert state.revisions == 3


async def test_boundary_events() -> None:
    ctx = FakeContext(
        agents={
            "slow": returns({"slow": True}, delay=1),
            "risky": fails,
            "recover": returns({"recovered": True}),
        }
    )
    state = await run("boundary", ctx, error_field="error")

    assert not state.slow
    assert state.error == "model unavailable"
    assert state.recovered


@pytest.mark.parametrize(
    ("replies", "confirmed"), [({"Customer": {"ok": True}}, True), ({}, False)]
)
async def test_messages(replies: dict[str, object], confirmed: bool) -> None:
    ctx = FakeContext(agents={"confirm": returns({"confirmed": True})}, replies=replies)
    state = await run("messages", ctx, State(order=7))

    assert ctx.sent == [("Customer", State(order=7))]
    assert state.confirmed == state.ok == confirmed


async def test_validate() -> None:
    graph = load("inclusive")

    assert validate(load("messages"), ["confirm"], {}) == []
    assert validate(graph, ["mail", "chat", "log"], {}) == [
        "Task 'done' calls the unknown agent 'done'"
    ]
    assert validate(graph, None, {"wants_chat": print}) == []

    with pytest.raises(ValueError, match="calls the unknown agent"):
        BpmnProcess(graph=graph)

    assert validate(load("boundary"), None, {}, outputs={"end": "x"}) == [
        "Outputs name 'end', which is no task or catch event",
        "Error boundary event 'failed' needs an error_field",
    ]


@pytest.mark.parametrize(
    ("outputs", "problem"),
    [({}, "outputs` names no field"), ({"combine": "total"}, "no fields total")],
)
async def test_outputs(outputs: dict[str, str], problem: str) -> None:
    ctx = FakeContext(
        agents={"web": returns({}), "papers": returns({}), "combine": returns(3)}
    )

    with pytest.raises(ValueError, match=problem):
        await run("parallel", ctx, outputs=outputs)


def flow(key: str, source: str, target: str, condition: str = "") -> str:
    expression = (
        f"<conditionExpression>{condition}</conditionExpression>" if condition else ""
    )

    return f'<sequenceFlow id="{key}" sourceRef="{source}" targetRef="{target}">{expression}</sequenceFlow>'


@pytest.mark.parametrize(
    ("elements", "problem"),
    [
        (
            '<startEvent id="s"/><exclusiveGateway id="a" default="out"/><exclusiveGateway id="b"/><endEvent id="e"/>'
            + flow("f1", "s", "a")
            + flow("loop", "a", "b", "done")
            + flow("out", "a", "e")
            + flow("back", "b", "a"),
            "A token loops through 'a' without reaching an activity",
        ),
        (
            '<startEvent id="s"/><task id="t" name="done"/><task id="u" name="done"/><endEvent id="e"/>'
            + flow("f1", "s", "t")
            + flow("f2", "t", "e")
            + flow("f3", "u", "e"),
            "Unreachable elements: 'u'",
        ),
        (
            '<startEvent id="s"/><parallelGateway id="j"/><task id="t" name="done"/>'
            '<exclusiveGateway id="x" default="out"/><endEvent id="e"/>'
            + flow("f1", "s", "j")
            + flow("f2", "j", "t")
            + flow("f3", "t", "x")
            + flow("back", "x", "j", "done")
            + flow("out", "x", "e"),
            "missing a dominating fork",
        ),
        (
            '<startEvent id="s"/><task id="t" name="done"/><task id="u" name="done"/>'
            + flow("f1", "s", "t")
            + flow("f2", "t", "u")
            + flow("f3", "u", "t"),
            "The process reaches no end event",
        ),
        (
            '<task id="t" name="done"/><endEvent id="e"/>' + flow("f1", "t", "e"),
            "The process has no start event",
        ),
    ],
    ids=[
        "loop without activity",
        "unreachable",
        "join without fork",
        "no end",
        "no start",
    ],
)
async def test_degenerate(elements: str, problem: str) -> None:
    graph = from_bpmn(
        parse_bpmn(
            f'<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"><process id="p">{elements}</process></definitions>'
        )
    )
    agents = {"done": FunctionAgent(func=lambda state: {"done": True})}

    [reported] = validate(graph, agents, {})

    assert problem in reported

    with pytest.raises(ValueError, match="Invalid process"):
        BpmnProcess[State](graph=graph, agents=agents)


GATE = from_bpmn(
    parse_bpmn("""
    <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
      <process id="gate">
        <startEvent id="start"/>
        <exclusiveGateway id="check" default="skip"/>
        <task id="publish" name="publisher"/>
        <endEvent id="end"/>
        <sequenceFlow id="f1" sourceRef="start" targetRef="check"/>
        <sequenceFlow id="long" sourceRef="check" targetRef="publish">
          <conditionExpression>words(task) &gt; 2</conditionExpression>
        </sequenceFlow>
        <sequenceFlow id="skip" sourceRef="check" targetRef="end"/>
        <sequenceFlow id="f2" sourceRef="publish" targetRef="end"/>
      </process>
    </definitions>
""")
)


@pytest.mark.parametrize(
    ("task", "published"), [("a long task", True), ("short", False)]
)
async def test_functions(task: str, published: bool) -> None:
    process = BpmnProcess[State](
        graph=GATE,
        agents={"publisher": FunctionAgent(func=lambda state: {"published": True})},
        functions={"words": lambda text: len(text.split())},
    )

    assert "state check <<choice>>" in process.render()
    assert (await process(State(task=task))).published == published
