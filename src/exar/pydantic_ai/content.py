"""Registers the user content of pydantic-ai with `exar.text.genai_parts`,
so prompts render as text wherever exar renders a task, such as in hints and traces.
Content is converted by pydantic-ai as for its spans and decoded as exar decodes spans.
"""

from opentelemetry.util.genai.types import MessagePart
from pydantic import TypeAdapter
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    TextContent,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_core import to_json

from exar.otel import _BYTES
from exar.text import genai_parts

__all__: list[str] = []

_SETTINGS = InstrumentationSettings(version=6, include_binary_content=True)
_PARTS = TypeAdapter(list[MessagePart], config=_BYTES)


@genai_parts.register
def _(
    value: TextContent
    | BinaryContent
    | ImageUrl
    | AudioUrl
    | VideoUrl
    | DocumentUrl
    | UploadedFile
    | CachePoint,
    /,
) -> list[MessagePart]:
    return _PARTS.validate_json(
        to_json(UserPromptPart([value]).otel_message_parts(_SETTINGS))
    )
