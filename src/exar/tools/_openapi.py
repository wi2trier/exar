"""The configuration of `OpenApi`, its OpenAPI documents, and the input schemas of their operations."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ._host import HostClient

__all__ = [
    "OpenApiDocument",
    "OpenApiFields",
    "OpenApiOperation",
    "OpenApiParameter",
    "operation_schema",
]

type _Json = dict[str, JsonValue]

_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")


def _inline(node: JsonValue, root: _Json, seen: tuple[str, ...] = ()) -> JsonValue:
    """Replaces every local `$ref` of the document by what it points to.

    Raises:
        ValueError: If a reference is cyclic or points nowhere.
    """
    if isinstance(node, list):
        return [_inline(item, root, seen) for item in node]

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")

    if not isinstance(ref, str):
        return {key: _inline(value, root, seen) for key, value in node.items()}

    if ref in seen:
        raise ValueError(f"The reference {ref} is cyclic")

    target: JsonValue = root

    for part in ref.removeprefix("#/").split("/"):
        key = part.replace("~1", "/").replace("~0", "~")

        if (
            not ref.startswith("#/")
            or not isinstance(target, dict)
            or key not in target
        ):
            raise ValueError(f"The reference {ref} points nowhere")

        target = target[key]

    return _inline(target, root, (*seen, ref))


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class OpenApiParameter(_Model):
    """A parameter of an operation."""

    name: str
    location: Literal["path", "query", "header", "cookie"] = Field(alias="in")
    required: bool = False
    description: str = ""
    schema_: _Json = Field(default_factory=dict, alias="schema")


class _Content(_Model):
    schema_: _Json = Field(default_factory=dict, alias="schema")


class _Body(_Model):
    required: bool = False
    content: dict[str, _Content] = {}


class OpenApiOperation(_Model):
    """An operation of a path, from which a tool is made."""

    operation_id: str | None = Field(default=None, alias="operationId")
    summary: str = ""
    description: str = ""
    parameters: list[OpenApiParameter] = []
    request_body: _Body | None = Field(default=None, alias="requestBody")


class _PathItem(_Model):
    parameters: list[OpenApiParameter] = []
    get: OpenApiOperation | None = None
    put: OpenApiOperation | None = None
    post: OpenApiOperation | None = None
    delete: OpenApiOperation | None = None
    patch: OpenApiOperation | None = None
    head: OpenApiOperation | None = None
    options: OpenApiOperation | None = None
    trace: OpenApiOperation | None = None

    def operations(self) -> dict[str, OpenApiOperation]:
        """The operations by their upper-case method."""
        return {
            method.upper(): operation
            for method in _METHODS
            if isinstance(operation := getattr(self, method), OpenApiOperation)
        }


class OpenApiDocument(_Model):
    """The paths of an OpenAPI 3 document, with its references inlined."""

    paths: dict[str, _PathItem]

    @model_validator(mode="before")
    @classmethod
    def _inline_refs(cls, data: JsonValue) -> JsonValue:
        return _inline(data, data) if isinstance(data, dict) else data

    def operations(self) -> dict[str, tuple[str, str, OpenApiOperation]]:
        """Every operation with an id by its id, with its method and path."""
        return {
            operation.operation_id: (method, path, operation)
            for path, item in self.paths.items()
            for method, operation in item.operations().items()
            if operation.operation_id
        }


def operation_schema(parameters: list[OpenApiParameter], body: _Body | None) -> _Json:
    """The input schema of an operation, its parameters and its JSON body, every path parameter required."""
    properties: _Json = {
        parameter.name: {**parameter.schema_, "description": parameter.description}
        for parameter in parameters
    }
    required: list[JsonValue] = [
        parameter.name
        for parameter in parameters
        if parameter.required or parameter.location == "path"
    ]

    if body and (media := body.content.get("application/json")):
        properties["body"] = media.schema_

        if body.required:
            required.append("body")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


@dataclass(slots=True, frozen=True, kw_only=True)
class OpenApiFields:
    """The configuration of `exar.tools.http.OpenApi`, which its requests read."""

    client: HostClient
    base_url: str
    document: OpenApiDocument
    operations: tuple[str, ...]
    prefix: str = ""
    max_response_bytes: int = 1_000_000
    timeout_seconds: float = 20.0
