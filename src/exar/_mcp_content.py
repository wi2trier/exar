"""The results of tools as MCP content, which `exar.mcp` and `exar.claude_agent_sdk` share, importing only the MCP types."""

import base64
from urllib.parse import quote

from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from pydantic_ai import BinaryContent

from .tools.batch import ToolResult

__all__ = ["tool_content"]


def _content_block(file: BinaryContent, /) -> ContentBlock:
    """A file of a tool result as an MCP content block, an image or else an embedded resource.

    Examples:
        >>> _content_block(BinaryContent(b"png", media_type="image/png", identifier="a")).type
        'image'
    """
    encoded = base64.b64encode(file.data).decode("ascii")

    if file.is_image:
        return ImageContent(type="image", data=encoded, mime_type=file.media_type)

    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=f"exar://file/{quote(file.identifier, safe='')}",  # pyright: ignore[reportArgumentType]
            mime_type=file.media_type,
            blob=encoded,
        ),
    )


def tool_content(result: ToolResult, /) -> list[ContentBlock]:
    """The text and the files of an unpacked tool result (`exar.tools.batch.unpack`) as MCP content blocks.

    Examples:
        >>> from exar.tools.batch import unpack
        >>> [block.type for block in tool_content(unpack("x"))]
        ['text']
    """
    return [
        TextContent(type="text", text=result.text),
        *(
            _content_block(file)
            for file in result.files
            if isinstance(file, BinaryContent)
        ),
    ]
