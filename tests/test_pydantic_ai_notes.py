import asyncio
from collections.abc import Sequence

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from exar.pydantic_ai import Read, RunNote, ToolNote, latest_read
from exar.pydantic_ai.compaction import compaction_window, is_compaction


def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("ok")])


def _turns(
    note: ToolNote[str], deps: Sequence[str], compact: Sequence[bool]
) -> list[ModelMessage]:
    agent = Agent(FunctionModel(_reply), deps_type=str, capabilities=[note])
    history: list[ModelMessage] = []

    for dep, compacted in zip(deps, compact, strict=True):
        if compacted:
            history.append(ModelResponse(parts=[CompactionPart(content="gist")]))

        history = asyncio.run(
            agent.run("hi", deps=dep, message_history=history)
        ).all_messages()

    return history


def _reads(history: Sequence[ModelMessage]) -> list[str]:
    return [
        part.model_response_str()
        for message in history
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def test_a_note_once_per_window_is_reported_again_after_a_compaction() -> None:
    async def memory(
        ctx: RunContext[str], window: Sequence[ModelMessage]
    ) -> list[Read]:
        if latest_read(window, "read_memory") is not None:
            return []

        return [Read(tool="read_memory", content=ctx.deps)]

    history = _turns(
        ToolNote(reads=memory, metadata={"hidden": True}),
        ["a", "b", "c"],
        [False, False, True],
    )

    assert _reads(history) == ["a", "c"]
    assert all(
        m.metadata == {"hidden": True}
        for m in history
        if any(isinstance(p, ToolReturnPart) for p in m.parts)
    )


def test_a_note_reports_a_changed_result_and_skips_an_unchanged_one() -> None:
    async def assistant(
        ctx: RunContext[str], window: Sequence[ModelMessage]
    ) -> list[Read]:
        return [Read(tool="read_assistant", content=ctx.deps, args={"scope": "user"})]

    history = _turns(ToolNote(reads=assistant), ["a", "a", "b"], [False, False, False])

    assert _reads(history) == ["a", "b"]


def test_stored_histories_keep_their_markers() -> None:
    """The shapes exar reads back from stored histories, see `exar.pydantic_ai`."""
    stored = ModelMessagesTypeAdapter.validate_json("""[
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "old"}]},
        {"kind": "response", "parts": [{"part_kind": "compaction", "content": "sum"}]},
        {"kind": "response", "parts": [
            {"part_kind": "tool-call", "tool_name": "read_memory", "args": {}, "tool_call_id": "1"}
        ]},
        {"kind": "request", "parts": [
            {"part_kind": "tool-return", "tool_name": "read_memory", "content": "- likes tea", "tool_call_id": "1"}
        ]}
    ]""")
    window = compaction_window(stored)

    assert is_compaction(stored[1])
    assert isinstance(window[0], ModelRequest) and window[0].parts == stored[1].parts
    assert latest_read(window, "read_memory") == "- likes tea"


def test_a_rewriting_run_note_is_fixed_text_ahead_of_instruction_callables() -> None:
    sent: list[list[tuple[str, bool]]] = []

    def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = info.model_request_parameters.instruction_parts or []
        sent.append([(part.content, part.dynamic) for part in parts])
        return _reply(messages, info)

    agent = Agent(
        FunctionModel(reply),
        deps_type=str,
        instructions=lambda ctx: "callable",
        capabilities=[
            RunNote[str](render=lambda ctx: f"tools {ctx.deps}", on_change="rewrite")
        ],
    )
    first = asyncio.run(agent.run("hi", deps="a")).all_messages()
    asyncio.run(agent.run("hi", deps="b", message_history=first))

    assert sent == [
        [("tools a", False), ("callable", True)],
        [("tools b", False), ("callable", True)],
    ]
