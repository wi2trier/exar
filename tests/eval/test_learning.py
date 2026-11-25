import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.util.genai.types import BlobPart, MessagePart, UriPart
from pydantic import JsonValue
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, Evaluator, EvaluatorContext

from exar.agents import FunctionAgent, RunConfig, run_async
from exar.eval import evaluate_sequence, export, optimize
from exar.eval.export import Format, chat_messages
from exar.experience import Retained
from exar.otel import handler
from exar.policy import allow_all
from exar.pydantic_ai import PydanticAiAgent, instrumentation
from exar.typing import Trace, Verdict

Agent.instrument_all(instrumentation())


async def solve(task: str, instructions: str = "") -> Trace[str, str]:
    async def answer(text: str, /) -> str:
        with handler().inference("test", request_model="m") as chat:
            chat.attributes["operation.cost"] = 0.5

        return text.upper() if "uppercase" in instructions else text

    return await run_async(answer, task, RunConfig(policy=allow_all), name="qa")


def cases(*tasks: str) -> Dataset[str, str, None]:
    return Dataset(
        name="tasks",
        cases=[
            Case(
                inputs=task,
                expected_output=task.upper(),
                evaluators=(EqualsExpected(),),
            )
            for task in tasks
        ],
    )


@dataclass
class Half(Evaluator[str, str, None]):
    def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> float:
        return 0.5


async def test_evaluate_sequence() -> None:
    first = await evaluate_sequence(cases("a"), solve)
    curve = await evaluate_sequence(cases("a", "B"), solve, done=first.assessments)

    assert curve.assessments[0] is first.assessments[0]
    assert curve.pass_rates == [0.0, 0.5]
    assert curve.costs == [0.5, 1.0]

    with pytest.raises(ValueError, match="not of the test case"):
        await evaluate_sequence(cases("B"), solve, done=first.assessments)

    strict = await evaluate_sequence(
        Dataset[str, str, None](
            name="half", cases=[Case(inputs="a")], evaluators=[Half()]
        ),
        solve,
        threshold=0.75,
    )

    assert strict.pass_rates == [0.0]


async def test_optimize() -> None:
    proposer = TestModel(custom_output_text="Answer in uppercase.")
    candidates = await optimize(
        "Answer.",
        cases("a", "b"),
        lambda prompt, task: solve(task, prompt),
        proposer,
        budget=3,
        seed=0,
    )

    assert [(c.prompt, c.score, c.parent) for c in candidates] == [
        ("Answer in uppercase.", 1.0, "Answer."),
        ("Answer.", 0.0, None),
    ]


async def test_export_is_lossless(tmp_path: Path) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ThinkingPart("look up"),
                    ToolCallPart("lookup", {"q": "x"}, tool_call_id="1"),
                ]
            )

        return ModelResponse(parts=[TextPart("Paris")])

    agent = Agent(FunctionModel(respond), instructions="Answer briefly.")

    @agent.tool_plain
    def lookup(q: str) -> str:
        """Looks up."""
        return "Paris"

    trace = await run_async(
        PydanticAiAgent(run=agent.run, name="agent"),
        "Capital?",
        RunConfig(policy=allow_all),
    )
    lyon = await run_async(
        FunctionAgent(func=str), "Capital?", RunConfig(policy=allow_all)
    )

    assert chat_messages(trace) == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Capital?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "look up",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "Paris"},
        {"role": "assistant", "content": "Paris"},
    ]

    assessments = [
        Retained(trace=trace, verdict=Verdict(score=1.0, passed=True)),
        Retained(trace=lyon, verdict=Verdict(score=0.0, passed=False)),
    ]
    path = tmp_path / "data.jsonl"

    def rows(format: Format) -> list[dict[str, Any]]:
        export(assessments, path, format=format)

        return [json.loads(line) for line in path.read_text().splitlines()]

    [sft] = rows("sft")
    [preference] = rows("preferences")

    assert sft["messages"] == chat_messages(trace)
    assert sft["tools"][0]["function"]["name"] == "lookup"
    assert [row["reward"] for row in rows("rewards")] == [1.0, 0.0]
    assert preference["rejected"] == [{"role": "assistant", "content": "Capital?"}]


@pytest.mark.parametrize(
    ("part", "content"),
    [
        (
            BlobPart(mime_type="audio/wav", modality="audio", content=b"RIFF"),
            {
                "type": "input_audio",
                "input_audio": {"data": "UklGRg==", "format": "wav"},
            },
        ),
        (
            UriPart(
                mime_type=None, modality="audio", uri="data:audio/mpeg;base64,SUQz"
            ),
            {"type": "input_audio", "input_audio": {"data": "SUQz", "format": "mp3"}},
        ),
        (
            BlobPart(mime_type="audio/ogg", modality="audio", content=b"OggS"),
            {"type": "file", "file": {"file_data": "data:audio/ogg;base64,T2dnUw=="}},
        ),
        (
            BlobPart(mime_type="video/mp4", modality="video", content=b"ftyp"),
            {"type": "file", "file": {"file_data": "data:video/mp4;base64,ZnR5cA=="}},
        ),
        (
            UriPart(
                mime_type="video/mp4", modality="video", uri="https://example.com/a.mp4"
            ),
            {"type": "text", "text": "https://example.com/a.mp4"},
        ),
    ],
)
async def test_chat_messages_export_media(
    part: MessagePart, content: JsonValue
) -> None:
    trace = await run_async(
        FunctionAgent(func=lambda task: "ok"),
        ["Listen", part],
        RunConfig(policy=allow_all),
    )

    assert chat_messages(trace)[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Listen"}, content],
    }
