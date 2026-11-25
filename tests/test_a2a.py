import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

import httpx
import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from pydantic import TypeAdapter
from starlette.requests import Request

from exar import otel
from exar.a2a import A2aAgent, app
from exar.agents import FunctionAgent, RunConfig, call, run_async, spend
from exar.errors import AgentError
from exar.otel import Usage
from exar.policy import Rules, allow_all
from exar.text import TEXT
from exar.typing import Action, Agent, Policy, Trace

URL = "http://remote"


def double(data: dict[str, float]) -> float:
    return 2 * data["value"]


async def failing(task: str, /) -> str:
    raise ValueError("broken")


async def spending(task: str, /) -> str:
    spend(Usage(input_tokens=3, requests=1))

    return task


async def guarded(task: str, /) -> str:
    delete = FunctionAgent[str, str](func=lambda path: f"deleted {path}")

    return await call(delete, task, name="delete")


def card(url: str = URL) -> AgentCard:
    return AgentCard(
        name="served",
        description="Serves an agent.",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0", url=url)
        ],
    )


def principal(request: Request) -> str | None:
    return request.headers.get("x-principal")


async def serve(
    stack: AsyncExitStack,
    agent: Agent[Any, Any],
    policy: Policy = allow_all,
    caller: str = "ada",
    served: AgentCard | None = None,
    wait: float = 600,
    approvals: int = 100,
    task_type: type[Any] = str,
    output: TypeAdapter[Any] = TEXT,
    on_trace: Callable[[Trace[Any, Any]], object] | None = None,
) -> A2aAgent[Any, Any]:
    """Serves an agent and returns a client agent for it, calling as *caller*."""
    asgi = app(
        agent,
        RunConfig(policy=policy),
        task_type=task_type,
        card=served or card(),
        authenticate=principal,
        wait=wait,
        on_trace=on_trace,
    )

    http = await stack.enter_async_context(
        httpx.AsyncClient(
            transport=httpx.ASGITransport(asgi),
            base_url=URL,
            headers={"x-principal": caller} if caller else {},
        )
    )

    return A2aAgent(
        name="remote", url=URL, http=http, approvals=approvals, output=output
    )


async def test_remote_agent(caplog: pytest.LogCaptureFixture) -> None:
    served: list[Trace[Any, Any]] = []

    async with AsyncExitStack() as stack:
        remote = await serve(
            stack,
            FunctionAgent(func=double),
            task_type=dict[str, float],
            output=TypeAdapter(float),
            on_trace=served.append,
        )
        trace = await run_async(remote, {"value": 21}, RunConfig(policy=allow_all))
        spent = await run_async(
            await serve(stack, spending), "go", RunConfig(policy=allow_all)
        )

        with pytest.raises(AgentError, match="Invalid task"):
            await run_async(remote, {"value": "many"}, RunConfig(policy=allow_all))

        with pytest.raises(AgentError, match="The run failed$"):
            await run_async(
                await serve(stack, failing), "go", RunConfig(policy=allow_all)
            )

    assert trace.output == 42
    assert [t.output for t in served] == [42]
    assert (otel.usage(spent.root).input_tokens, otel.usage(spent.root).requests) == (3, 1)
    assert "broken" in caplog.text


async def test_a_card_cannot_redirect_the_requests() -> None:
    async with AsyncExitStack() as stack:
        remote = await serve(stack, failing, served=card("http://elsewhere"))

        with pytest.raises(AgentError, match="no interface"):
            await run_async(remote, "go", RunConfig(policy=allow_all))


async def test_a_task_waits_for_its_answer_for_a_bounded_time() -> None:
    async def late(action: Action) -> bool:
        await asyncio.sleep(0.1)

        return True

    async with AsyncExitStack() as stack:
        remote = await serve(
            stack, guarded, Rules(by_name={"delete": "ask"}), wait=0.01
        )

        with pytest.raises(AgentError, match="waited too long"):
            await run_async(remote, "a.md", RunConfig(policy=allow_all, approve=late))


async def test_the_approvals_of_an_invocation_are_bounded() -> None:
    async with AsyncExitStack() as stack:
        remote = await serve(
            stack, guarded, Rules(by_name={"delete": "ask"}), approvals=0
        )

        with pytest.raises(AgentError, match="more than 0 approvals"):
            await run_async(remote, "a.md", RunConfig(policy=allow_all, approve=allow))


@pytest.mark.parametrize("approved", [True, False])
async def test_remote_ask_reaches_local_approver(approved: bool) -> None:
    asked: list[Action] = []

    async def approve(action: Action) -> bool:
        asked.append(action)

        return approved

    async with AsyncExitStack() as stack:
        remote = await serve(stack, guarded, Rules(by_name={"delete": "ask"}))
        trace = run_async(remote, "a.md", RunConfig(policy=allow_all, approve=approve))

        if approved:
            assert (await trace).output == "deleted a.md"
        else:
            with pytest.raises(AgentError, match="Refused"):
                await trace

    assert [(a.kind, a.name, a.effect, a.details) for a in asked] == [
        ("agent", "delete", "execute", "a.md")
    ]


async def test_an_unauthenticated_request_runs_nothing() -> None:
    ran: list[str] = []

    async def record(task: str, /) -> str:
        ran.append(task)

        return task

    async with AsyncExitStack() as stack:
        remote = await serve(stack, record, caller="")

        with pytest.raises(Exception, match="authenticates no principal"):
            await run_async(remote, "go", RunConfig(policy=allow_all))

    assert ran == []


async def allow(action: Action) -> bool:
    return True


async def test_json_tasks_arrive_as_sent_and_empty_ones_are_refused() -> None:
    async with AsyncExitStack() as stack:
        remote = await serve(
            stack,
            FunctionAgent(func=lambda task: task),
            task_type=object,
            output=TypeAdapter(object),
        )
        task = {"n": 5, "big": 2**60, "xs": [1.5, None]}

        assert (
            await run_async(remote, task, RunConfig(policy=allow_all))
        ).output == task

        with pytest.raises(AgentError, match="empty text"):
            await run_async(remote, "", RunConfig(policy=allow_all))
