"""Export of assessed runs as JSONL datasets in the OpenAI chat format for fine-tuning and reinforcement learning.

The formats follow the conversational datasets of TRL, which verl and OpenRLHF read as well:
`sft` rows hold the `messages` of passing runs for supervised fine-tuning,
`rewards` rows hold the `prompt` and `completion` messages of every assessed run with its score as `reward`,
and `preferences` rows pair the `chosen` completion of a passing run
with the `rejected` completion of a failing run of the same task.
The prompt is every message before the first assistant message,
and every row holds the `tools` the model was offered, if any.

`chat_messages` reads the conversation of the root agent from the GenAI messages of its model calls,
losslessly with system instructions, media, reasoning (`reasoning_content`), tool calls, and tool results.
Media becomes the content part of the chat format that carries it:
images an `image_url`, WAV and MP3 audio an `input_audio`, and other binary data or provider files a `file`.
Data URIs count as binary data, while the chat format references no other media by URI than images,
so the URI of such media, such as a video URL, is left as text.
"""

import base64
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict
from urllib.parse import urlsplit
from urllib.request import urlopen

from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    UriPart,
)
from pydantic_core import to_json

from ..helpers import atomic_write
from ..otel import (
    attribute,
    input_messages,
    operation,
    output_messages,
    system_instructions,
)
from ..text import as_text, media_parts
from ..typing import Assessed, JsonEntry, Span, Trace

__all__ = [
    "ChatMessage",
    "Format",
    "PreferenceRow",
    "RewardRow",
    "Row",
    "SftRow",
    "chat_messages",
    "chat_tools",
    "export",
    "rows",
    "task_key",
]

type ChatMessage = dict[str, JsonEntry]
"""A message in the OpenAI chat format, such as `{"role": "user", "content": "Hi"}`."""

type Format = Literal["sft", "rewards", "preferences"]
"""The kind of dataset, see the module."""


class SftRow(TypedDict):
    """A row of the `sft` format."""

    messages: Sequence[ChatMessage]
    tools: NotRequired[Sequence[ChatMessage]]


class RewardRow(TypedDict):
    """A row of the `rewards` format."""

    prompt: Sequence[ChatMessage]
    completion: Sequence[ChatMessage]
    reward: float
    tools: NotRequired[Sequence[ChatMessage]]


class PreferenceRow(TypedDict):
    """A row of the `preferences` format."""

    prompt: Sequence[ChatMessage]
    chosen: Sequence[ChatMessage]
    rejected: Sequence[ChatMessage]
    tools: NotRequired[Sequence[ChatMessage]]


type Row = SftRow | RewardRow | PreferenceRow
"""A row of any format."""

# The audio formats of `input_audio` by media type.
_AUDIO = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


def _chats(trace: Trace[Any, Any]) -> list[Span]:
    """The model calls of the outermost agent that calls models in order, skipping those of subagents."""
    chats: list[tuple[int, Span]] = []

    def visit(span: Span, agents: int) -> None:
        if operation(span) == "chat":
            chats.append((agents, span))

        for child in span.children:
            visit(child, agents + (operation(span) == "invoke_agent"))

    visit(trace.root, 0)
    outermost = min((agents for agents, _ in chats), default=0)

    return [span for agents, span in chats if agents == outermost]


def _url(part: BlobPart | UriPart) -> str:
    if isinstance(part, UriPart):
        return part.uri

    return f"data:{part.mime_type or 'application/octet-stream'};base64,{base64.b64encode(part.content).decode()}"


def _blob(part: UriPart) -> BlobPart:
    """The binary data of a data URI."""
    with urlopen(part.uri) as response:
        return BlobPart(
            mime_type=part.mime_type or response.headers.get_content_type(),
            modality=part.modality,
            content=response.read(),
        )


def _content(part: MessagePart) -> JsonEntry:
    if (
        isinstance(part, UriPart)
        and str(part.modality) != "image"
        and urlsplit(part.uri).scheme == "data"
    ):
        part = _blob(part)

    match part:
        case TextPart(content=text):
            return {"type": "text", "text": text}
        case BlobPart() | UriPart() if str(part.modality) == "image":
            return {"type": "image_url", "image_url": {"url": _url(part)}}
        case BlobPart(content=data) if audio := _AUDIO.get(part.mime_type or ""):
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(data).decode(),
                    "format": audio,
                },
            }
        case BlobPart():
            return {"type": "file", "file": {"file_data": _url(part)}}
        case FilePart(file_id=file_id):
            return {"type": "file", "file": {"file_id": file_id}}
        case UriPart(uri=uri):
            return {"type": "text", "text": uri}
        case _:
            return None


def _text(
    parts: Sequence[MessagePart], kind: type[TextPart | ReasoningPart]
) -> str | None:
    texts = [part.content for part in parts if isinstance(part, kind)]

    return "\n".join(texts) if texts else None


def _chat(message: InputMessage | OutputMessage) -> list[ChatMessage]:
    parts = message.parts

    match message.role:
        case "assistant":
            chat: dict[str, JsonEntry] = {
                "role": "assistant",
                "content": _text(parts, TextPart),
            }

            if (reasoning := _text(parts, ReasoningPart)) is not None:
                chat["reasoning_content"] = reasoning

            if calls := [
                part for part in parts if isinstance(part, ToolCallRequestPart)
            ]:
                chat["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": to_json(call.arguments).decode(),
                        },
                    }
                    for call in calls
                ]

            # Null content is for a message of tool calls alone.
            if chat["content"] is None and "tool_calls" not in chat:
                chat["content"] = ""

            return [chat]
        case "tool":
            return [
                {
                    "role": "tool",
                    "tool_call_id": part.id,
                    "content": as_text(part.response),
                }
                for part in parts
                if isinstance(part, ToolCallResponsePart)
            ]
        case role:
            if all(isinstance(part, TextPart) for part in parts):
                return [{"role": role, "content": _text(parts, TextPart) or ""}]

            content: list[JsonEntry] = [
                item for part in parts if (item := _content(part)) is not None
            ]

            return [{"role": role, "content": content}]


def chat_messages(trace: Trace[Any, Any], /) -> list[ChatMessage]:
    """The conversation of the root agent of a run in the OpenAI chat format.

    It is the system instructions and the input messages of the first model call
    of the outermost agent that calls models, followed by the output messages of each model call
    and the messages sent with the next one, so the reasoning that providers drop from later requests is kept.
    Without model calls, it is the task, with its media if it is multimodal content (`exar.text.genai_parts`),
    and the output.

    Examples:
        >>> from exar.agents import FunctionAgent, RunConfig, run
        >>> from exar.policy import allow_all
        >>> chat_messages(run(FunctionAgent(func=lambda question: "Paris"), "Capital of France?", RunConfig(policy=allow_all)))
        [{'role': 'user', 'content': 'Capital of France?'}, {'role': 'assistant', 'content': 'Paris'}]
    """
    if not (chats := _chats(trace)):
        parts = media_parts(trace.task) or [TextPart(as_text(trace.task))]

        return [
            *_chat(InputMessage(role="user", parts=parts)),
            {"role": "assistant", "content": as_text(trace.output)},
        ]

    system = _text(system_instructions(chats[0]), TextPart)
    head: list[ChatMessage] = [{"role": "system", "content": system}] if system else []
    messages: list[InputMessage | OutputMessage] = []
    sent = 0

    for chat in chats:
        inputs = input_messages(chat)
        messages.extend(inputs[sent:])
        outputs = output_messages(chat)
        messages.extend(outputs)
        sent = len(inputs) + len(outputs)

    return [
        *head,
        *(chat for message in messages for chat in _chat(message)),
    ]


def chat_tools(trace: Trace[Any, Any], /) -> list[ChatMessage]:
    """The function tools offered in the last model call of the root agent in the OpenAI format, see `chat_messages`."""
    chats = _chats(trace)
    definitions = attribute(chats[-1], "gen_ai.tool.definitions") if chats else None

    if not isinstance(definitions, list):
        return []

    return [
        {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters"),
            },
        }
        for tool in definitions
        if isinstance(tool, dict) and tool.get("type") == "function"
    ]


def task_key(trace: Trace[Any, Any], /) -> str:
    """The task of a run as JSON with its binary data in base64, which pairs runs of exactly the same task.

    Unlike `as_text`, it keeps the data of media, so tasks with different images of the same size differ.
    """
    return to_json(trace.task, fallback=str, bytes_mode="base64").decode()


def _split(
    messages: Sequence[ChatMessage],
) -> tuple[Sequence[ChatMessage], Sequence[ChatMessage]]:
    index = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages),
    )

    return messages[:index], messages[index:]


def rows(
    assessments: Iterable[Assessed[Any, Any]],
    /,
    *,
    format: Format = "sft",
    key: Callable[[Trace[Any, Any]], Hashable] = task_key,
) -> Iterator[Row]:
    """The rows of a dataset of the assessed runs, see `export` and the module.

    Args:
        assessments: The assessed runs.
        format: The kind of dataset.
        key: Groups the runs of the same or a similar task for preference pairs.
    """
    evaluated = [
        (a.trace, a.verdict, chat_messages(a.trace), chat_tools(a.trace))
        for a in assessments
    ]

    match format:
        case "sft":
            for _, verdict, messages, tools in evaluated:
                if verdict.passed:
                    yield _offered({"messages": messages}, tools)
        case "rewards":
            for _, verdict, messages, tools in evaluated:
                prompt, completion = _split(messages)
                yield _offered(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "reward": verdict.score,
                    },
                    tools,
                )
        case "preferences":
            groups: defaultdict[
                Hashable, list[tuple[bool, Sequence[ChatMessage], list[ChatMessage]]]
            ] = defaultdict(list)

            for trace, verdict, messages, tools in evaluated:
                groups[key(trace)].append((verdict.passed, messages, tools))

            for group in groups.values():
                for passed, messages, tools in group:
                    if not passed:
                        continue

                    prompt, chosen = _split(messages)

                    for rejected_passed, rejected, _ in group:
                        if not rejected_passed:
                            yield _offered(
                                {
                                    "prompt": prompt,
                                    "chosen": chosen,
                                    "rejected": _split(rejected)[1],
                                },
                                tools,
                            )


def _offered[R: Row](row: R, tools: Sequence[ChatMessage]) -> R:
    """The row with the tools offered in the run, if any."""
    if tools:
        row["tools"] = tools

    return row


def export(
    assessments: Iterable[Assessed[Any, Any]],
    path: Path,
    /,
    *,
    format: Format = "sft",
    key: Callable[[Trace[Any, Any]], Hashable] = task_key,
) -> int:
    """Writes the assessed runs as a JSONL dataset atomically and returns the number of rows.

    The rows go to a sibling that replaces the file once complete, so a failed export leaves the file as it was.

    Args:
        assessments: The assessed runs.
        path: The JSONL file.
        format: The kind of dataset, see the module.
        key: Groups the runs of the same or a similar task for preference pairs,
            such as by a cluster of the tasks by a cbrkit measure.
    """
    lines = [to_json(row) + b"\n" for row in rows(assessments, format=format, key=key)]
    atomic_write(path, lines)

    return len(lines)
