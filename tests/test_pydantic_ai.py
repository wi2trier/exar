import asyncio
import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from cbrkit.indexable import InMemory
from cbrkit.sim.strings import WordCoverage
from opentelemetry.util.genai.types import TextPart as GenAIText
from pydantic_ai import (
    Agent,
    BinaryContent,
    DeferredToolRequests,
    FunctionToolset,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import (
    ModelContextWindowExceededError,
    OutputTokenLimitExceeded,
)
from pydantic_ai.messages import CompactionPart, ModelResponsePart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from exar import otel
from exar.agents import FunctionAgent, RunConfig, call, decide, run_async, usage
from exar.experience import CHANGES, FileChanges, Files, Retained, StorageExperience
from exar.otel import Usage
from exar.policy import EffectPolicy, allow_all
from exar.pydantic_ai import (
    CountUsage,
    DeclaredEffects,
    EffectGate,
    PydanticAiAgent,
    RecordWorkspace,
    RetrieveExperience,
    instrumentation,
    prompt_text,
    toolset,
    traced_events,
)
from exar.pydantic_ai.compaction import (
    COMPACT_PROMPT,
    CompactionWindow,
    CompactOnOverflow,
    run_summary,
    summarize,
)
from exar.text import as_text
from exar.tools import Retry, effect
from exar.typing import Action, Decision, Policy, Trace, Verdict
from exar.workflows import Chain

Agent.instrument_all(instrumentation())


def prompts(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def model(*calls: ToolCallPart) -> FunctionModel:
    """A model that makes the calls one per request and then answers with the last tool result."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if (count := sum(isinstance(m, ModelResponse) for m in messages)) < len(calls):
            return ModelResponse(parts=[ThinkingPart("plan"), calls[count]])

        last = messages[-1].parts[-1]

        return ModelResponse(
            parts=[
                TextPart(
                    str(last.content) if isinstance(last, ToolReturnPart) else "done"
                )
            ]
        )

    return FunctionModel(respond, model_name="m")


@effect("read")
async def search(query: str) -> str:
    """Searches."""
    if query == "bad":
        raise Retry("try good")

    return query


@effect("write")
async def save(text: str) -> str:
    """Saves."""
    return f"saved {text}"


async def mail(to: str) -> str:
    """Mails, without a declared effect."""
    return to


async def test_traces_the_run_in_a_workflow() -> None:
    agent = Agent(
        model(
            ToolCallPart("search", {"query": "bad"}, tool_call_id="b"),
            ToolCallPart("search", {"query": "cbr"}, tool_call_id="s"),
        ),
        name="researcher",
        instructions="Research.",
        toolsets=[toolset([search])],
    )
    researcher = PydanticAiAgent(run=agent.run, name="researcher", effect="read")
    trace = await run_async(
        Chain[str, str](agents=[researcher]), "go", RunConfig(policy=allow_all)
    )
    tools = [span for span in trace.walk() if otel.operation(span) == "execute_tool"]
    chats = [span for span in trace.walk() if otel.operation(span) == "chat"]

    assert trace.output == "cbr"
    assert otel.agents(trace.root) == ("Chain", "researcher")
    assert [
        (otel.tool_name(span), otel.tool_call_arguments(span)) for span in tools
    ] == [
        ("search", {"query": "bad"}),
        ("search", {"query": "cbr"}),
    ]
    assert otel.system_instructions(chats[-1]) == [GenAIText(content="Research.")]
    assert [m.role for m in otel.input_messages(chats[-1])] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert (
        otel.usage(trace.root).input_tokens
        == sum(otel.usage(chat).input_tokens for chat in chats)
        > 0
    )


async def test_records_only_the_share_of_a_run_on_a_shared_usage() -> None:
    shared = RunUsage(input_tokens=1000, requests=3)
    looker = Agent(TestModel(custom_output_text="seen"), name="looker")
    agent = PydanticAiAgent[list[str], str](
        run=lambda task: looker.run(
            task, instructions="Look.", metadata={"tenant": "a"}, usage=shared
        ),
        name="looker",
    )
    trace = await run_async(agent, ["Look at", "this"], RunConfig(policy=allow_all))
    [chat] = [span for span in trace.walk() if otel.operation(span) == "chat"]
    [invocation] = [span for span in trace.walk() if otel.attribute(span, "metadata")]

    assert trace.output == "seen"
    assert otel.system_instructions(chat) == [GenAIText(content="Look.")]
    assert otel.attribute(invocation, "metadata") == {"tenant": "a"}
    assert shared.input_tokens > 1000
    assert (
        otel.usage(trace.root).input_tokens
        == otel.usage(chat).input_tokens
        == shared.input_tokens - 1000
    )


def failing() -> Agent[None, str]:
    agent = Agent(TestModel(), name="failing", capabilities=[CountUsage()])

    @agent.tool_plain
    def boom() -> str:
        """Fails."""
        raise ValueError("broken")

    return agent


async def test_usage_counts_every_request_of_nested_and_failed_runs() -> None:
    helper = Agent(
        TestModel(custom_output_text="help"), name="helper", capabilities=[CountUsage()]
    )
    lead = Agent(TestModel(), name="lead", capabilities=[CountUsage()])

    @lead.tool
    async def delegate(ctx: RunContext[object]) -> str:
        """Delegates."""
        return (await helper.run("help", usage=ctx.usage)).output

    async def host(task: str) -> Usage:
        await call(PydanticAiAgent(run=lead.run, name="lead"), task)

        with pytest.raises(ValueError, match="broken"):
            await call(PydanticAiAgent(run=failing().run, name="failing"), task)

        return usage()

    trace = await run_async(host, "go", RunConfig(policy=allow_all))
    [led, failed] = trace.root.children
    chats = [span for span in led.walk() if otel.operation(span) == "chat"]

    assert len(chats) == 3
    assert otel.usage(led) == sum((otel.usage(chat) for chat in chats), Usage())
    assert failed.status == "error" and otel.usage(failed).input_tokens > 0
    assert (
        trace.output == otel.usage(trace.root) == otel.usage(led) + otel.usage(failed)
    )
    assert (
        otel.attribute(trace.root, "exar.usage.input_tokens")
        == otel.usage(trace.root).input_tokens
    )


async def test_traced_events_nests_the_run_and_passes_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    greeter = Agent(TestModel(custom_output_text="Hi"), name="greeter")

    async def host(agent: Agent[None, str]) -> str:
        try:
            async for _ in traced_events(
                agent.run_stream_events("go"),
                "go",
                RunConfig(policy=allow_all),
                name="chat",
            ):
                pass
        except ValueError as error:
            return str(error)

        return "done"

    greeted, failed = (
        await run_async(host, greeter, RunConfig(policy=allow_all)),
        await run_async(host, failing(), RunConfig(policy=allow_all)),
    )
    [chat] = greeted.root.children
    [broken] = failed.root.children

    assert (greeted.output, failed.output) == ("done", "broken")
    assert [otel.operation(span) for span in chat.walk()] == [
        "invoke_agent",
        "invoke_agent",
        "chat",
    ]
    assert otel.output_messages(chat)[0].parts == [GenAIText(content="Hi")]
    assert otel.usage(chat).input_tokens > 0
    assert (
        broken.status,
        otel.error_type(broken),
        otel.usage(broken) == otel.usage(failed.root),
    ) == (
        "error",
        "ValueError",
        True,
    )

    traces: list[Trace[str, str]] = []
    run = greeter.run_stream_events("go")

    async for _ in traced_events(
        run, "go", RunConfig(policy=allow_all), name="chat", on_trace=traces.append
    ):
        pass

    assert [otel.operation(span) for span in traces[0].walk()] == [
        "invoke_agent",
        "invoke_agent",
        "chat",
    ]


async def test_traced_events_keeps_the_run_out_of_the_consumer() -> None:
    errors: list[dict[str, Any]] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _, context: errors.append(context)
    )
    events = traced_events(
        Agent(TestModel()).run_stream_events("go"),
        "go",
        RunConfig(policy=allow_all),
        name="chat",
    )

    # Outside the run there is no policy, and the consumer fails between events.
    with pytest.raises(RuntimeError, match="No policy"):
        async for _ in events:
            decide(Action(kind="tool", name="probe", effect="read"))

    del events
    gc.collect()
    await asyncio.sleep(0.1)

    assert not errors


async def test_effect_gate_hides_and_pauses() -> None:
    agent = Agent(
        model(ToolCallPart("save", {"text": "a"}, tool_call_id="w")),
        output_type=[str, DeferredToolRequests],
        toolsets=[
            toolset([search, save, mail]),
            FunctionToolset([Tool[object](mail)])
            .prefixed("third")
            .with_metadata(effect="read"),
        ],
        capabilities=[EffectGate()],
    )
    trace = await run_async(
        PydanticAiAgent(run=agent.run, name="agent"),
        "go",
        RunConfig(policy=EffectPolicy(write="ask", execute="deny")),
    )
    [chat] = [span for span in trace.walk() if otel.operation(span) == "chat"]
    offered = otel.attribute(chat, "gen_ai.tool.definitions")

    assert isinstance(trace.output, DeferredToolRequests)
    assert [call.tool_name for call in trace.output.approvals] == ["save"]
    assert isinstance(offered, list)
    assert {str(tool["name"]) for tool in offered if isinstance(tool, dict)} == {
        "save",
        "search",
        "third_mail",
    }


async def test_declared_effects_reach_the_gate() -> None:
    offered: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.extend(tool.name for tool in info.function_tools)

        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(respond),
        toolsets=[FunctionToolset([Tool[object](mail)])],
        capabilities=[
            EffectGate(lambda _: EffectPolicy(execute="deny")),
            DeclaredEffects({"mail": "read"}),
        ],
    )
    await agent.run("go")

    assert offered == ["mail"]


async def test_effect_gate_decides_by_the_run_and_the_tool_metadata() -> None:
    offered: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.extend(tool.name for tool in info.function_tools)

        return ModelResponse(parts=[TextPart("done")])

    def policy(ctx: RunContext[bool]) -> Policy:
        return lambda action: (
            "allow" if ctx.deps or action.metadata.get("trusted") else "deny"
        )

    agent = Agent(
        FunctionModel(respond),
        deps_type=bool,
        toolsets=[
            toolset([mail]),
            FunctionToolset([Tool[bool](mail)])
            .prefixed("trusted")
            .with_metadata(trusted=True),
        ],
        capabilities=[EffectGate(policy)],
    )
    await agent.run("go", deps=False)
    await agent.run("go", deps=True)

    assert offered == ["trusted_mail", "mail", "trusted_mail"]


async def test_effect_gate_denies_a_call_by_its_arguments() -> None:
    saved: list[str] = []

    @effect("write")
    async def keep(text: str) -> str:
        """Keeps."""
        saved.append(text)

        return text

    def policy(action: Action) -> Decision:
        return "deny" if action.details == {"text": "secret"} else "allow"

    agent = Agent(
        model(ToolCallPart("keep", {"text": "secret"})),
        toolsets=[toolset([keep])],
        capabilities=[EffectGate(lambda _: policy)],
    )
    result = await agent.run("go")

    assert saved == []
    assert "denied" in result.output


async def test_compacts_on_overflow() -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        asked = prompts(messages)

        if asked[-1] == COMPACT_PROMPT:
            return ModelResponse(parts=[TextPart("summary")])

        if any(isinstance(p, CompactionPart) for m in messages for p in m.parts):
            assert asked == ["second"]
            return ModelResponse(parts=[TextPart("compacted")])

        if len(asked) > 1:
            raise ModelContextWindowExceededError("function", "too long")

        return ModelResponse(parts=[TextPart("answer")])

    agent = Agent(
        FunctionModel(model), capabilities=[CompactOnOverflow(), CompactionWindow()]
    )
    first = await agent.run("first")
    second = await agent.run("second", message_history=first.all_messages())
    compaction = second.all_messages()[2]

    assert second.output == "compacted"
    assert isinstance(compaction, ModelResponse)
    assert compaction.parts == [CompactionPart(content="summary")]
    assert second.usage.requests == 2


async def test_overflow_without_cut_raises() -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelContextWindowExceededError("function", "too long")

    agent = Agent(FunctionModel(model), capabilities=[CompactOnOverflow()])

    with pytest.raises(ModelContextWindowExceededError):
        await agent.run("first")


@pytest.mark.parametrize(
    ("parts", "attempts"), [([], 2), ([ThinkingPart("long thoughts")], 1)]
)
async def test_summary_sheds_messages_only_when_no_room_was_left(
    parts: list[ModelResponsePart], attempts: int
) -> None:
    asked: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        asked.append(len(messages))

        # Reported usage, since `FunctionModel` estimates output tokens otherwise.
        return ModelResponse(
            parts=parts, finish_reason="length", usage=RequestUsage(input_tokens=1)
        )

    agent = Agent(FunctionModel(model))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("a")]),
        ModelResponse(parts=[TextPart("x")], usage=RequestUsage(input_tokens=5000)),
        ModelRequest(parts=[UserPromptPart("b")]),
        ModelResponse(parts=[TextPart("y")], usage=RequestUsage(input_tokens=20000)),
    ]

    async def ask(messages: Sequence[ModelMessage]) -> str:
        result = await run_summary(
            lambda capabilities: agent.iter(
                COMPACT_PROMPT, message_history=messages, capabilities=capabilities
            )
        )

        return result.output

    # A reply that spent its cap thinking is no overflow, so nothing is shed for a retry.
    with pytest.raises(OutputTokenLimitExceeded):
        await summarize(history, ask, max_tokens=100)

    assert asked == [5, 3][:attempts]


async def test_summary_called_a_tool_is_asked_again_without_tools() -> None:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart("note", {}, tool_call_id="n")])

        return ModelResponse(parts=[TextPart("summary")])

    agent = Agent(FunctionModel(model))

    @agent.tool_plain
    def note() -> str:
        return "noted"

    result = await run_summary(
        lambda capabilities: agent.iter(
            COMPACT_PROMPT,
            usage_limits=UsageLimits(request_limit=1, tool_calls_limit=0),
            capabilities=capabilities,
        )
    )

    assert result.output == "summary"


@pytest.mark.parametrize(("text", "hinted"), [(["Add 2 and 3"], True), ([], False)])
async def test_retrieve_experience_multimodal(text: list[str], hinted: bool) -> None:
    cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
    trace = await run_async(
        FunctionAgent(func=lambda task: "4"), "Add 2 and 2", RunConfig(policy=allow_all)
    )
    await cases.retain([Retained(trace=trace, verdict=Verdict(score=1.0, passed=True))])
    agent = Agent(
        TestModel(),
        capabilities=[RetrieveExperience(experience=cases, task=prompt_text)],
    )
    result = await agent.run([*text, ImageUrl("https://example.com/sum.png")])
    [request, *_] = result.all_messages()

    assert isinstance(request, ModelRequest)
    assert (request.instructions is not None) is hinted


def test_prompts_render_media_as_placeholders() -> None:
    prompt = [
        "Compare",
        BinaryContent(b"\x89PNG", media_type="image/png"),
        ImageUrl("https://example.com/b.png"),
    ]

    assert (
        as_text(prompt)
        == "Compare\n\n[image: image/png, 4 bytes]\n\n[image: image/png, https://example.com/b.png]"
    )


async def test_retrieve_experience_tool() -> None:
    cases = StorageExperience[int, str, str](storage=InMemory(), measure=WordCoverage())
    trace = await run_async(
        FunctionAgent(func=lambda task: "4"), "Add 2 and 2", RunConfig(policy=allow_all)
    )
    await cases.retain([Retained(trace=trace, verdict=Verdict(score=1.0, passed=True))])
    agent = Agent(
        TestModel(call_tools=["retrieve_experience"]),
        capabilities=[
            RetrieveExperience(experience=cases, task=prompt_text, mode="tool")
        ],
    )
    messages = (await agent.run("Add 2 and 3")).all_messages()
    [result] = [
        part.model_response_str()
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]

    assert "follow their approach" in result
    assert isinstance(messages[0], ModelRequest) and messages[0].instructions is None


async def _history(tools: FunctionToolset[Any], name: str) -> list[object]:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(name, {}, "call")])

        return ModelResponse(parts=[TextPart("done")])

    result = await Agent(FunctionModel(model), toolsets=[tools]).run("go")

    return [
        (part.content, part.metadata, part.model_response_str(), part.files)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


@pytest.mark.parametrize(
    "value",
    ["plain", {"a": 1}, ["text", BinaryContent(b"png", media_type="image/png")]],
    ids=["str", "dict", "multimodal"],
)
async def test_toolset_returns_what_pydantic_ai_makes_of_a_plain_value(
    value: object,
) -> None:
    async def tool() -> object:
        return value

    bare = await _history(FunctionToolset([Tool[None](tool, name="tool")]), "tool")

    assert await _history(toolset([tool]), "tool") == bare


async def test_record_workspace(tmp_path: Path) -> None:
    (tmp_path / "input.csv").write_text("a\n")
    (tmp_path / "old.txt").write_text("old")

    def work() -> str:
        (tmp_path / "solve.py").write_text("print(4)\n")
        (tmp_path / "old.txt").unlink()
        return "done"

    agent = Agent(
        TestModel(),
        name="coder",
        tools=[work],
        capabilities=[LocalWorkspace(tmp_path), RecordWorkspace()],
    )

    async def solve(task: str) -> str:
        return (await agent.run(task)).output

    trace = await run_async(solve, "Sum", RunConfig(policy=allow_all))
    changes = FileChanges.of(trace)

    assert [span.name for span in trace.walk() if CHANGES in span.attributes] == [
        "invoke_agent coder"
    ]
    assert changes is not None
    assert [(f.path, f.change, f.text) for f in changes.files] == [
        ("old.txt", "removed", None),
        ("solve.py", "added", "print(4)\n"),
    ]
    assert '<file path="solve.py" change="added">' in await Files()(
        [Retained(trace=trace, verdict=Verdict(score=1.0, passed=True))], "Sum"
    )
