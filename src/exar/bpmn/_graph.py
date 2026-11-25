"""The compilation of a BPMN process into a pydantic-graph graph."""

from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, partial
from operator import eq
from typing import Any

import anyio
from cbrkit.model.graph import BpmnEdge, BpmnGraph, BpmnNode, bpmn
from pydantic import TypeAdapter
from pydantic_graph import (
    Decision,
    Graph,
    GraphBuilder,
    Join,
    Step,
    StepContext,
    reduce_null,
)
from pydantic_graph.node_types import DestinationNode, SourceNode
from pydantic_graph.paths import EdgePathBuilder
from pydantic_graph.step import StepFunction

from ..agents import call
from ..errors import AgentError
from ..typing import Agent, Channel
from ._conditions import condition, fields
from ._elements import (
    Index,
    agent_of,
    calls_agent,
    defines,
    kind_of,
    seconds_of,
)

__all__ = ["Deps", "State", "compile_process"]


@dataclass(slots=True, kw_only=True)
class State[S]:
    """The state of a run, the graph state of pydantic-graph, which the results of its steps update.

    Args:
        value: The current state.
        adapter: Validates the updated states.
        fields: The fields of the state.
    """

    value: S
    adapter: TypeAdapter[S]
    fields: tuple[str, ...]

    @classmethod
    def of(cls, value: S) -> State[S]:
        """The state of a run that starts with a value.

        Raises:
            TypeError: If the value is neither a pydantic model nor a dataclass.
        """
        adapter, names = _schema(type(value))

        return cls(value=value, adapter=adapter, fields=names)

    def update(self, key: str, result: object, output: str | None) -> None:
        """Updates the state by the result of a node, see `exar.bpmn.BpmnProcess`.

        Args:
            key: The id of the node.
            result: Replaces the state if it is one, updates the fields it names if it is a mapping,
                and else the field `output`.
            output: The field that a plain result updates, None if the node has none.
        """
        if isinstance(result, type(self.value)):
            self.value = result

            return

        if isinstance(result, Mapping):
            changes: Mapping[Any, object] = result
        elif output is None:
            raise ValueError(
                f"{key!r} returned a plain {type(result).__name__}, but `outputs` names no field for it"
            )
        else:
            changes = {output: result}

        if unknown := [name for name in changes if name not in self.fields]:
            raise ValueError(
                f"{key!r} updates unknown fields of the state: {', '.join(map(str, unknown))}"
            )

        current = {name: getattr(self.value, name) for name in self.fields}
        self.value = self.adapter.validate_python({**current, **changes})


@cache
def _schema(kind: type, /) -> tuple[TypeAdapter[Any], tuple[str, ...]]:
    """The adapter and fields of a state type, built once per type.

    Raises:
        TypeError: If the type is neither a pydantic model nor a dataclass.
    """
    if (names := fields(kind)) is None:
        raise TypeError(
            f"The state type {kind.__name__} is neither a pydantic model nor a dataclass"
        )

    return TypeAdapter(kind), names


@dataclass(frozen=True, slots=True, kw_only=True)
class Deps[S]:
    """The dependencies of a run, the agents that tasks invoke and the channel of messages."""

    agents: Mapping[str, Agent[S, Any]]
    channel: Channel[S] | None

    def messages(self) -> Channel[S]:
        """The channel.

        Raises:
            ValueError: If there is none.
        """
        if self.channel is None:
            raise ValueError("The process exchanges messages but has no channel")

        return self.channel


@dataclass(frozen=True, slots=True)
class _Event:
    """The boundary or catch event a step is left by."""

    key: str


type _Context[S] = StepContext[State[S], Deps[S], object]
type _Destination[S] = DestinationNode[State[S], Deps[S], Any]
type _Source[S] = SourceNode[State[S], Deps[S], Any]
type _Step[S] = Step[State[S], Deps[S], object, _Event | None]
type _Action[S] = Callable[[_Context[S]], Awaitable[None]]


def _current[S](ctx: _Context[S]) -> S:
    """The current state, the input of every node after a step."""
    return ctx.state.value


def _peer(node: BpmnNode, flows: Collection[BpmnEdge]) -> str | None:
    """The agent at the other end of a node's message flows.

    It is the node's `agent` attribute, or else the pool, agent, or name of the first
    message flow's other end.
    """
    if agent := node.value.get("agent"):
        return agent

    if not flows:
        return None

    flow = next(iter(flows))
    other = flow.target if flow.source.key == node.key else flow.source

    return other.value.get("pool") or agent_of(other)


def _earliest(nodes: Iterable[BpmnNode]) -> tuple[BpmnNode | None, float | None]:
    """The timer event among the nodes that expires first and its seconds, None for both without one."""
    timer = min(
        (node for node in nodes if defines(node, bpmn.TTimerEventDefinition)),
        key=seconds_of,
        default=None,
    )

    return timer, None if timer is None else seconds_of(timer)


@dataclass(slots=True, kw_only=True)
class _Compiler[S]:
    """Builds the graph of a process or embedded subprocess.

    Activities and events that do something become steps, which return the event they are left by, if any.
    Joining parallel and inclusive gateways become joins, other gateways become decisions over the state.
    Steps and joins are connected to what follows them only after the nodes before them,
    so the decision of a gateway is complete before a loop leads back to it.

    Raises:
        ValueError: If the scope has no start event, a token loops without reaching a step or join,
            no end event can be reached, or pydantic-graph rejects the graph, such as a join that no fork dominates.
    """

    graph: BpmnGraph
    index: Index
    functions: Mapping[str, Callable[..., object]]
    outputs: Mapping[str, str]
    error_field: str | None
    parent: str | None
    builder: GraphBuilder[State[S], Deps[S], None, None] = field(init=False)
    entries: dict[str, Sequence[_Destination[S]]] = field(default_factory=dict)
    end: Join[State[S], Deps[S], object, None] | None = None
    pending: list[Callable[[], None]] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)
    routing: set[str] = field(default_factory=set)
    ended: bool = False

    def __post_init__(self) -> None:
        self.builder = GraphBuilder(
            name=self.parent or "process", state_type=State[S], deps_type=Deps[S]
        )

    def build(self) -> Graph[State[S], Deps[S], None, None]:
        scope = (
            "The process" if self.parent is None else f"The subprocess {self.parent!r}"
        )

        if not (starts := self.index.starts.get(self.parent)):
            raise ValueError(f"{scope} has no start event")

        for start in starts:
            self.connect(self.builder.start_node, self.entry(start))

        while self.pending:
            self.pending.pop()()

        if not self.ended:
            raise ValueError(f"{scope} reaches no end event")

        return self.builder.build()

    def edge(self, source: _Source[S]) -> EdgePathBuilder[State[S], Deps[S], Any]:
        # ty does not resolve the `Source` alias of `GraphBuilder` yet.
        return self.builder.edge_from(source)  # ty: ignore[invalid-argument-type]

    def connect(
        self, source: _Source[S], destinations: Sequence[_Destination[S]]
    ) -> None:
        self.builder.add(self.edge(source).transform(_current).to(*destinations))

    def ends(self) -> _Destination[S]:
        """The join of the tokens that reach plain end events, after which the run ends."""
        self.ended = True

        if self.end is None:
            self.end = self.builder.join(reduce_null, initial=None)
            self.builder.add(self.edge(self.end).to(self.builder.end_node))

        return self.end

    def entry(self, node: BpmnNode) -> Sequence[_Destination[S]]:
        """The nodes a token that reaches a BPMN node goes to."""
        if (entry := self.entries.get(node.key)) is not None:
            return entry

        self.visited.add(node.key)
        kind = kind_of(node.value)

        if (
            issubclass(kind, bpmn.TParallelGateway | bpmn.TInclusiveGateway)
            and len(self.index.incoming[node.key]) > 1
        ):
            join = self.builder.join(
                reduce_null,
                initial=None,
                node_id=node.key,
                preferred_parent_fork="closest",
            )
            self.entries[node.key] = [join]
            self.pending.append(lambda: self.connect(join, self.route(node)))
        elif (step := self.step(node)) is not None:
            self.entries[node.key] = [step]
            self.pending.append(partial(self.leave, node, step))
        elif issubclass(kind, bpmn.TEndEvent) and defines(
            node, bpmn.TTerminateEventDefinition
        ):
            self.ended = True
            self.entries[node.key] = [self.builder.end_node]
        elif issubclass(kind, bpmn.TEndEvent):
            self.entries[node.key] = [self.ends()]
        elif node.key in self.routing:
            raise ValueError(
                f"A token loops through {node.key!r} without reaching an activity"
            )
        else:
            self.routing.add(node.key)
            self.entries[node.key] = self.route(node)
            self.routing.discard(node.key)

        return self.entries[node.key]

    def leave(self, node: BpmnNode, step: _Source[S]) -> None:
        """Connects a step to its boundary events or the events after an event-based gateway, and its flows."""
        gateway = issubclass(kind_of(node.value), bpmn.TEventBasedGateway)
        events = (
            [flow.target for flow in self.index.outgoing[node.key]]
            if gateway
            else self.index.boundaries[node.key]
        )

        if not events:
            self.connect(step, self.route(node))

            return

        decision = self.builder.decision()

        for event in events:
            self.visited.add(event.key)
            decision = decision.branch(
                self.builder.match(_Event, matches=partial(eq, _Event(event.key)))
                .transform(_current)
                .to(*self.route(event))
            )

        if not gateway:
            decision = decision.branch(
                self.builder.match(object).transform(_current).to(*self.route(node))
            )

        self.builder.add(self.edge(step).to(decision))

    def route(self, node: BpmnNode) -> Sequence[_Destination[S]]:
        """The nodes a token that leaves a BPMN node by its outgoing flows goes to.

        Parallel gateways take all flows, exclusive gateways the first whose condition holds,
        and all other nodes every flow whose condition holds or that has none.
        The default flow is taken if no other one is.
        """
        flows = self.index.outgoing[node.key]

        if not flows:
            return [self.ends()]

        kind = kind_of(node.value)
        default = node.value.get("default")
        conditional = default is not None or any(
            flow.value.get("condition") is not None for flow in flows
        )

        if issubclass(kind, bpmn.TParallelGateway) or (
            not conditional and (not issubclass(kind, bpmn.TGateway) or len(flows) == 1)
        ):
            return [entry for flow in flows for entry in self.entry(flow.target)]

        # Steps and joins hold the id of their node already.
        node_id = None if node.key in self.entries else node.key

        if not issubclass(kind, bpmn.TExclusiveGateway) and len(flows) > 1:
            return [self.inclusive(node, flows, node_id)]

        decision = self.builder.decision(node_id=node_id)

        for flow in sorted(flows, key=lambda flow: flow.key == default):
            decision = decision.branch(
                self.builder.match(
                    object,
                    matches=None if flow.key == default else self.condition(flow),
                ).to(*self.entry(flow.target))
            )

        return [decision]

    def inclusive(
        self, node: BpmnNode, flows: Sequence[BpmnEdge], node_id: str | None
    ) -> _Destination[S]:
        """Maps the state to the indices of the flows taken and routes each to its flow."""
        default = node.value.get("default")
        fallback = [index for index, flow in enumerate(flows) if flow.key == default]
        conditions = [
            (index, self.condition(flow))
            for index, flow in enumerate(flows)
            if flow.key != default
        ]

        def taken(ctx: _Context[S]) -> list[int]:
            state = _current(ctx)
            indices = [
                index for index, holds in conditions if holds is None or holds(state)
            ] or fallback

            if not indices:
                raise ValueError(f"No outgoing sequence flow of {node.key!r} applies")

            return indices

        targets: Decision[State[S], Deps[S], Any] = self.builder.decision()

        for index, flow in enumerate(flows):
            targets = targets.branch(
                self.builder.match(int, matches=partial(eq, index))
                .transform(_current)
                .to(*self.entry(flow.target))
            )

        return self.builder.decision(node_id=node_id).branch(
            self.builder.match(object).transform(taken).map().to(targets)
        )

    def condition(self, flow: BpmnEdge) -> Callable[[object], bool] | None:
        text = flow.value.get("condition")

        return None if text is None else condition(text, self.functions)

    def step(self, node: BpmnNode) -> _Step[S] | None:
        """The step of a node that does something, None for other nodes."""
        if issubclass(kind_of(node.value), bpmn.TEventBasedGateway):
            return self.builder.step(self.trigger(node), node_id=node.key)

        if (action := self.action(node)) is None:
            return None

        boundaries = self.index.boundaries[node.key]
        boundary = next(
            (b.key for b in boundaries if defines(b, bpmn.TErrorEventDefinition)),
            None,
        )
        timer, seconds = _earliest(boundaries)
        error_field = self.error_field

        async def perform(ctx: _Context[S]) -> _Event | None:
            with anyio.move_on_after(seconds) as scope:
                try:
                    await action(ctx)
                except Exception as error:
                    # Validation rejects error boundaries without a field, so this only guards.
                    if boundary is None or error_field is None:
                        raise

                    ctx.state.update(node.key, {error_field: str(error)}, None)

                    return _Event(boundary)

            if timer is not None and scope.cancelled_caught:
                return _Event(timer.key)

            return None

        return self.builder.step(perform, node_id=node.key)

    def action(self, node: BpmnNode) -> _Action[S] | None:
        """What a node does, None for nodes that do nothing."""
        value = node.value
        kind = kind_of(value)
        flows = self.index.messages[node.key]
        output = self.outputs.get(node.key)
        message = defines(node, bpmn.TMessageEventDefinition)

        if issubclass(kind, bpmn.TSubProcess):
            graph = _Compiler[S](
                graph=self.graph,
                index=self.index,
                functions=self.functions,
                outputs=self.outputs,
                error_field=self.error_field,
                parent=node.key,
                visited=self.visited,
            ).build()

            async def action(ctx: _Context[S]) -> None:
                await graph.run(state=ctx.state, deps=ctx.deps)

        elif issubclass(kind, bpmn.TSendTask) or (
            message and issubclass(kind, bpmn.TIntermediateThrowEvent | bpmn.TEndEvent)
        ):
            peer = _peer(node, flows)

            async def action(ctx: _Context[S]) -> None:
                await ctx.deps.messages().send(peer, ctx.state.value)

        elif issubclass(kind, bpmn.TReceiveTask) or (
            message and issubclass(kind, bpmn.TIntermediateCatchEvent)
        ):

            async def action(ctx: _Context[S]) -> None:
                _, content = await ctx.deps.messages().receive()
                ctx.state.update(node.key, content, output)

        elif issubclass(kind, bpmn.TIntermediateCatchEvent) and defines(
            node, bpmn.TTimerEventDefinition
        ):
            seconds = seconds_of(node)

            async def action(ctx: _Context[S]) -> None:
                await anyio.sleep(seconds)

        elif issubclass(kind, bpmn.TEndEvent) and defines(
            node, bpmn.TErrorEventDefinition
        ):
            error = value.get("errorRef") or value.get("name") or node.key

            async def action(ctx: _Context[S]) -> None:
                raise AgentError(error)

        elif calls_agent(node):
            name = agent_of(node) or node.key

            async def action(ctx: _Context[S]) -> None:
                agent = ctx.deps.agents[name]
                result = await call(agent, ctx.state.value, name=name)
                ctx.state.update(node.key, result, output)

        else:
            return None

        return action

    def trigger(
        self, gateway: BpmnNode
    ) -> StepFunction[State[S], Deps[S], object, _Event]:
        """Waits for the first event after an event-based gateway and returns it.

        The first message goes to the catch event whose peer is its sender, or else to the first one,
        unless the earliest timer expires first.
        """
        targets = [flow.target for flow in self.index.outgoing[gateway.key]]
        timer, seconds = _earliest(targets)
        # The peers are resolved here, so the step keeps no reference to the compiler.
        catches = [
            (target, _peer(target, self.index.messages[target.key]))
            for target in targets
            if not defines(target, bpmn.TTimerEventDefinition)
        ]
        outputs = self.outputs

        async def trigger(ctx: _Context[S]) -> _Event:
            if timer is not None and seconds is not None and not catches:
                await anyio.sleep(seconds)

                return _Event(timer.key)

            try:
                sender, content = await ctx.deps.messages().receive(timeout=seconds)
            except TimeoutError:
                if timer is None:
                    raise

                return _Event(timer.key)

            target = next(
                (catch for catch, peer in catches if peer == sender), catches[0][0]
            )
            ctx.state.update(target.key, content, outputs.get(target.key))

            return _Event(target.key)

        return trigger


def compile_process[S](
    graph: BpmnGraph,
    functions: Mapping[str, Callable[..., object]],
    outputs: Mapping[str, str],
    error_field: str | None,
    /,
) -> Graph[State[S], Deps[S], None, None]:
    """The graph of a process, whose runs end when no token is left or a terminate end event is reached.

    The results of nodes update the state, plain ones the field `outputs` names by node id,
    and error boundary events store the message of an error in `error_field`.

    Raises:
        ValueError: If the process or a subprocess is degenerate, see `_Compiler`,
            or a flow node cannot be reached from a start event.
    """
    compiler = _Compiler[S](
        graph=graph,
        index=Index.build(graph),
        functions=functions,
        outputs=outputs,
        error_field=error_field,
        parent=None,
    )
    flow = compiler.build()

    # Boundary events are attached to activities rather than reached by flows.
    if unreachable := [
        key
        for key, node in graph.nodes.items()
        if issubclass(kind := kind_of(node.value), bpmn.TFlowNode)
        and not issubclass(kind, bpmn.TBoundaryEvent)
        and key not in compiler.visited
    ]:
        raise ValueError(f"Unreachable elements: {', '.join(map(repr, unreachable))}")

    return flow
