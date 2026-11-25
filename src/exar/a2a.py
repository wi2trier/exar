"""Connects exar agents over the Agent2Agent protocol (A2A).

`app` serves an exar agent to A2A clients as an ASGI app, each A2A task one run of `exar.agents.run_async`
owned by the principal its `Authenticate` hook names, which every request needs,
and `A2aAgent` invokes a remote A2A agent like a local one.
Text parts map to text content, a single part of media type `application/json` to its value,
which travels as JSON text rather than as the data part of the protocol, whose numbers are all floats.
The served agent validates its tasks and the remote agent its outputs with a pydantic `TypeAdapter`.

Approvals cross the protocol as the `input-required` state of a task:
an action of a served run that the policy asks about becomes a status message
with the `exar.typing.Action` as data and the metadata key `exar/ask`,
and the client continues the task with `{"approved": true}` or `false` as data,
which counts only from the principal that owns the task.
`A2aAgent` asks the approver of its own run (`exar.agents.approve`) and answers such requests within its invocation.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from contextvars import Context
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, override

import httpx
from a2a.auth.user import UnauthenticatedUser, User
from a2a.client import A2ACardResolver, ClientCallContext, ClientConfig, create_client
from a2a.helpers import (
    get_data_parts,
    new_message,
    new_task_from_user_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.routes.common import DefaultServerCallContextBuilder
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCard,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    TaskState,
)
from google.protobuf.json_format import MessageToDict
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_json
from starlette.applications import Starlette
from starlette.requests import Request

from .agents import RunConfig, approve, run_async, spend
from .errors import AgentError, Refused
from .helpers import new_id
from .otel import Usage, usage
from .text import TEXT
from .typing import Action, Agent, Effect, Trace

__all__ = ["A2aAgent", "Authenticate", "app"]

_ASK: Final = "exar/ask"
"""Metadata key of the status message of an `input-required` task that asks to approve an action."""

_ACTION = TypeAdapter(Action)
_USAGE = TypeAdapter(Usage)
_PAUSED = (TaskState.TASK_STATE_INPUT_REQUIRED, TaskState.TASK_STATE_AUTH_REQUIRED)

_logger = logging.getLogger(__name__)

type Authenticate = Callable[[Request], str | None]
"""The principal a request authenticates, such as the subject of its verified token, `None` if it authenticates none."""


@dataclass(frozen=True, slots=True)
class _Principal(User):
    name: str

    @property
    @override
    def is_authenticated(self) -> bool:
        return True

    @property
    @override
    def user_name(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class _Authenticated(DefaultServerCallContextBuilder):
    authenticate: Authenticate

    @override
    def build(self, request: Request) -> ServerCallContext:
        principal = self.authenticate(request)
        user = UnauthenticatedUser() if not principal else _Principal(principal)

        return super().build(request).model_copy(update={"user": user})


def _owner(context: RequestContext) -> str:
    user = context.call_context.user

    if not user.is_authenticated:
        raise PermissionError("The request authenticates no principal")

    return user.user_name


_JSON: Final = "application/json"


def _parts(content: object) -> list[Part]:
    if isinstance(content, str):
        return [new_text_part(content)]

    return [new_text_part(to_json(content).decode(), _JSON)]


def _content(parts: Sequence[Part]) -> object:
    text_parts = [part for part in parts if part.HasField("text")]
    texts = [part.text for part in text_parts if part.media_type != _JSON]
    data = [
        *(json.loads(part.text) for part in text_parts if part.media_type == _JSON),
        *get_data_parts(parts),
    ]

    match texts, data:
        case [], [value]:
            return value
        case _, []:
            return "\n".join(texts)
        case _:
            return [*texts, *data]


type _Event = (
    tuple[Literal["ask"], Action, asyncio.Future[bool]]
    | tuple[Literal["done"], Trace[Any, Any]]
    | tuple[Literal["failed"], BaseException]
)


@dataclass(slots=True, kw_only=True)
class _Pending:
    """A run of a task that waits for the client to approve an action."""

    run: asyncio.Task[None]
    events: asyncio.Queue[_Event]
    answer: asyncio.Future[bool] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Executor[I, O](AgentExecutor):
    agent: Agent[I, O]
    config: RunConfig
    task: TypeAdapter[I]
    wait: float
    on_trace: Callable[[Trace[I, O]], object] | None
    pending: dict[tuple[str, str], _Pending] = field(
        default_factory=dict[tuple[str, str], _Pending]
    )

    def _start(self, key: tuple[str, str], task: I) -> _Pending:
        events: asyncio.Queue[_Event] = asyncio.Queue()

        async def ask(action: Action) -> bool:
            answer = asyncio.get_running_loop().create_future()
            events.put_nowait(("ask", action, answer))

            try:
                async with asyncio.timeout(self.wait):
                    return await answer
            except TimeoutError:
                self.pending.pop(key, None)
                raise

        async def solve() -> None:
            try:
                trace = await run_async(
                    self.agent, task, replace(self.config, approve=ask)
                )

                if self.on_trace is not None:
                    self.on_trace(trace)
            except Exception as error:  # noqa: BLE001
                events.put_nowait(("failed", error))
            else:
                events.put_nowait(("done", trace))

        # A served task is a run of its own, also when the server shares the context of a caller, such as in tests.
        return _Pending(
            run=asyncio.create_task(solve(), context=Context()), events=events
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        owner = _owner(context)
        message, task_id, context_id = (
            context.message,
            context.task_id,
            context.context_id,
        )

        if message is None or task_id is None or context_id is None:
            raise ValueError("The request has no message")

        if context.current_task is None:
            await event_queue.enqueue_event(new_task_from_user_message(message))

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        content = _content(message.parts)
        key = (owner, task_id)
        pending = self.pending.get(key)

        if pending is None and context.current_task is not None:
            await updater.failed(
                updater.new_agent_message(_parts("The task waited too long for input"))
            )

            return

        if pending is None:
            try:
                task = self.task.validate_python(content)
            except ValidationError as error:
                await updater.failed(
                    updater.new_agent_message(_parts(f"Invalid task: {error}"))
                )

                return

            pending = self.pending[key] = self._start(key, task)
        elif pending.answer is not None and not pending.answer.done():
            pending.answer.set_result(
                isinstance(content, dict) and content.get("approved") is True
            )

        match await pending.events.get():
            case ("ask", action, answer):
                pending.answer = answer
                await updater.requires_input(
                    updater.new_agent_message(
                        _parts(_ACTION.dump_python(action, mode="json")),
                        metadata={_ASK: True},
                    )
                )
            case ("done", trace):
                self.pending.pop(key, None)
                await updater.add_artifact(
                    _parts(trace.output),
                    name="result",
                    metadata={"usage": _USAGE.dump_python(usage(trace.root), mode="json")},
                )
                await updater.complete()
            case ("failed", error):
                self.pending.pop(key, None)
                await updater.failed(updater.new_agent_message(_parts(_failure(error))))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        owner = _owner(context)
        task_id, context_id = context.task_id, context.context_id

        if task_id is None or context_id is None:
            return

        if (pending := self.pending.pop((owner, task_id), None)) is not None:
            pending.run.cancel()

        await TaskUpdater(event_queue, task_id, context_id).cancel()


def _failure(error: BaseException, /) -> str:
    """What the client learns of a failed run, a refusal of the policy but no detail of other errors, which are logged."""
    if isinstance(error, Refused):
        return f"Refused: {error}"

    _logger.error("A run of an A2A task failed", exc_info=error)

    return "The run failed"


def app[I, O](
    agent: Agent[I, O],
    config: RunConfig,
    /,
    *,
    task_type: type[I],
    card: AgentCard,
    authenticate: Authenticate,
    wait: float = 600,
    rpc_url: str = "/",
    on_trace: Callable[[Trace[I, O]], object] | None = None,
) -> Starlette:
    """Serves an agent over A2A with JSON-RPC, each task one run of the agent owned by the principal that started it.

    The content of a new task is validated as a `task_type`, and a task that does not fit fails with the reason.
    The output of the run becomes the artifact of the task, and its usage the artifact's `usage` metadata.
    An action the policy asks about pauses the task as `input-required`, and the client's answer continues it.
    A task that waits longer than `wait` for its answer fails, and so does its run.
    Cancelling a task cancels its run.
    A failed run tells the client only that it failed or what the policy refused, and logs the error.
    Every request must authenticate a principal, and a task, its answers, and its cancellation are the owner's alone.

    Args:
        agent: The served agent.
        config: The policy, limits, and redaction of every run, whose approver is the client that owns the task.
        task_type: The type of the tasks of the agent.
        card: Describes the agent to clients, its JSON-RPC interface should point to `rpc_url`.
        authenticate: Names the principal of each request, such as from a token the host verified.
        wait: Seconds a task waits for the client to answer.
        rpc_url: Path of the JSON-RPC endpoint.
        on_trace: Receives the trace of every run that ended without an error, such as to assess and retain it.
    """
    handler = DefaultRequestHandler(
        agent_executor=_Executor(
            agent=agent,
            config=config,
            task=TypeAdapter(task_type),
            wait=wait,
            on_trace=on_trace,
        ),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(
                handler, rpc_url, context_builder=_Authenticated(authenticate)
            ),
        ]
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class A2aAgent[I = str, O = str]:
    """A remote A2A agent, each invocation a task in a new A2A context, whose result is validated as its output.

    A request of the remote agent to approve an action goes to the approver of the run (`exar.agents.approve`),
    and the answer continues the remote task within the invocation.
    A task that needs other input fails the invocation with the remote agent's question.
    The requests go only to the interfaces of the agent card at the origin of `url`,
    so a card cannot redirect them and the credentials of `http` to another host.
    The usage of the remote run, the `usage` metadata of its artifact, counts toward the local run (`exar.agents.spend`),
    while its spans stay in the trace of the remote run.

    Args:
        name: Name of the agent in traces.
        effect: What the remote agent may change, which policies check when a workflow invokes it.
        url: Base URL of the remote agent, where its agent card is served.
        output: Validates the result of the remote agent, as text by default.
        timeout: Seconds to wait for the remote agent to finish a task.
        http: Client for the requests, such as one with authentication headers.
        approvals: Approvals the remote agent may ask for within one invocation.
    """

    name: str
    effect: Effect = "execute"
    url: str
    output: TypeAdapter[O] = TEXT
    timeout: float = 600
    http: httpx.AsyncClient | None = None
    approvals: int = 100

    async def _card(self) -> AgentCard:
        """The agent card with only the interfaces at the origin of `url`.

        Raises:
            exar.errors.AgentError: If the card has no interface at the origin.
        """
        async with (
            nullcontext(self.http) if self.http is not None else httpx.AsyncClient()
        ) as http:
            card = await A2ACardResolver(http, self.url).get_agent_card()

        origin = _origin(self.url)
        interfaces = [
            interface
            for interface in card.supported_interfaces
            if _origin(interface.url) == origin
        ]

        if not interfaces:
            raise AgentError(f"The agent card has no interface at {self.url}")

        del card.supported_interfaces[:]
        card.supported_interfaces.extend(interfaces)

        return card

    async def __call__(self, task: I, /) -> O:
        if task == "":
            raise AgentError("An A2A task cannot be an empty text")

        context_id = new_id()
        config = ClientConfig(streaming=False, httpx_client=self.http)
        card = await self._card()

        async with await create_client(card, client_config=config) as client:

            async def send(content: object, task_id: str | None) -> StreamResponse:
                request = SendMessageRequest(
                    message=new_message(
                        _parts(content),
                        context_id=context_id,
                        task_id=task_id,
                        role=Role.ROLE_USER,
                    )
                )
                [response] = [
                    response
                    async for response in client.send_message(
                        request, context=ClientCallContext(timeout=self.timeout)
                    )
                ]

                return response

            response = await send(task, None)
            asked = 0

            while MessageToDict(response.task.status.message.metadata).get(_ASK):
                if (asked := asked + 1) > self.approvals:
                    raise AgentError(
                        f"The remote agent asked for more than {self.approvals} approvals"
                    )

                action = _ACTION.validate_python(
                    _content(response.task.status.message.parts)
                )
                # Asking pauses the remote task, which the answer continues.
                response = await send(
                    {"approved": await approve(action)}, response.task.id
                )

        if response.HasField("message"):
            return self.output.validate_python(_content(response.message.parts))

        state, status = response.task.status.state, response.task.status.message

        match state:
            case TaskState.TASK_STATE_COMPLETED:
                artifacts = response.task.artifacts

                for artifact in artifacts:
                    if (
                        usage := MessageToDict(artifact.metadata).get("usage")
                    ) is not None:
                        spend(_USAGE.validate_python(usage))

                return self.output.validate_python(
                    _content(
                        [part for artifact in artifacts for part in artifact.parts]
                    )
                )
            case state if state in _PAUSED:
                raise AgentError(
                    f"The remote agent needs input: {_content(status.parts)}"
                )
            case _:
                raise AgentError(
                    f"The remote task ended with {TaskState.Name(state)}: {_content(status.parts)}"
                )


def _origin(url: str, /) -> tuple[str, str, int | None]:
    parsed = httpx.URL(url)

    return parsed.scheme, parsed.host, parsed.port
