import ast
import sys
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry.util.genai.types import BlobPart, GenericPart, ToolCallResponsePart
from pydantic import TypeAdapter

import exar
from exar import otel
from exar.typing import Span, Trace

NOW = datetime.now(UTC)


def test_span_reads_genai_payload_and_round_trips() -> None:
    chat = Span(
        name="chat m",
        span_id="02",
        parent_id="01",
        start=NOW,
        end=NOW,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": '[{"role": "user", "parts": ['
            '{"type": "blob", "modality": "image", "mime_type": "image/png", "content": "iVBORw=="},'
            '{"type": "x.custom", "value": 1}]},'
            '{"role": "tool", "parts": [{"type": "tool_call_response", "id": "1", "result": 3}]}]',
            "gen_ai.usage.input_tokens": 3,
            "operation.cost": 0.5,
        },
    )
    root = Span(
        name="invoke_agent a",
        span_id="01",
        start=NOW,
        end=NOW,
        children=(chat,),
    )
    trace = Trace[str, str](task="Describe", output="A logo", root=root, id="ab")
    [user, tool] = otel.input_messages(chat)

    assert user.parts[0] == BlobPart(
        mime_type="image/png", modality="image", content=b"\x89PNG"
    )
    assert isinstance(user.parts[1], GenericPart)
    assert tool.parts == [ToolCallResponsePart(response=3, id="1")]
    assert (
        otel.usage(chat).input_tokens,
        otel.usage(chat).cost,
        otel.usage(trace.root).cost,
    ) == (3, 0.5, 0.5)

    adapter = TypeAdapter(Trace[str, str])

    assert adapter.validate_json(adapter.dump_json(trace)) == trace


def _imports(path: Path) -> set[str]:
    nodes = list(ast.walk(ast.parse(path.read_text())))

    return {
        "." * node.level + (node.module or "")
        for node in nodes
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in nodes
        if isinstance(node, ast.Import)
        for alias in node.names
    }


def test_core_depends_on_the_standard_library_only() -> None:
    root = Path(exar.__file__).parent

    for name in ("typing.py", "errors.py"):
        foreign = {
            module
            for module in _imports(root / name) - {".typing"}
            if module.split(".")[0] not in sys.stdlib_module_names
        }

        assert foreign == set(), name
