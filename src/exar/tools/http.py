"""Tools that call the operations of an HTTP API, one tool per operation of its OpenAPI document.

`OpenApi` is a keyword-only dataclass like the tools of `exar.tools.files`, whose `tools` returns one `SchemaTool` per allowed operation.
The name of a tool is the `operationId` and its input schema the one of the operation, its path, query, header, and cookie parameters and its JSON body.
A `GET` or `HEAD` has the effect `"read"` and every other method `"write"`, so policies ask before a call changes anything.
Requests go through the client of the host (`HostClient`), where it enforces its origin, authentication, and egress.
The document is data, so `$ref`s to its own components are inlined while it is validated, and what it does not describe is ignored.
It needs the `pydantic-ai` extra, and `httpx2` of the `http` extra only when the tools are built or called.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue

from .._lazy import extra_hint
from ..schema import serializable
from ..text import first_line
from . import SchemaTool
from ._openapi import (
    OpenApiDocument,
    OpenApiFields,
    OpenApiOperation,
    operation_schema,
)

__all__ = ["OpenApi", "OpenApiDocument", "OpenApiOperation"]

_IGNORED_HEADERS = frozenset({"accept", "content-type", "authorization"})
"""The header parameters OpenAPI says to ignore, which the client and the body set."""


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class OpenApi(OpenApiFields):
    """The allowed operations of an HTTP API described by an OpenAPI 3 document.

    Args:
        client: The client that sends, bound by the host, which owns its origin, authentication, and egress.
        base_url: What the paths of the document are appended to.
        document: The OpenAPI document.
        operations: The `operationId`s of the operations to offer, which must exist in the document.
        prefix: What to put before the names of the tools, such as the API they serve.
        max_response_bytes: Bytes of a response that are read, the rest is reported as cut.
        timeout_seconds: Bound of a request as a whole.
    """

    kind: Literal["openapi"] = field(default="openapi", repr=False)

    def instructions(self) -> str:
        """What to tell the model about the tools, naming each with the first line of its description.

        Raises:
            ValueError: If an allowed operation is not in the document.
        """
        operations = "\n".join(
            f"- {tool.name}: {first_line(tool.description)}" for tool in self.tools()
        )

        return f"The API offers the operations\n{operations}\n"

    def tools(self) -> list[SchemaTool]:
        """One tool per allowed operation.

        Raises:
            exar.errors.MissingExtra: If the `http` extra is missing.
            ValueError: If an allowed operation is not in the document.
        """
        known = self.document.operations()

        if unknown := sorted(set(self.operations) - known.keys()):
            raise ValueError(f"The document has no operations {', '.join(unknown)}")

        return [self._tool(name, *known[name]) for name in self.operations]

    def _tool(
        self, name: str, method: str, path: str, operation: OpenApiOperation
    ) -> SchemaTool:
        with extra_hint("OpenApi", "http"):
            from ._http import send

        # An operation's own parameter replaces one of its path with the same name and location.
        declared = {
            (parameter.name, parameter.location): parameter
            for parameter in [
                *self.document.paths[path].parameters,
                *operation.parameters,
            ]
            if not (
                parameter.location == "header"
                and parameter.name.lower() in _IGNORED_HEADERS
            )
        }
        parameters = list(declared.values())
        names = [parameter.name for parameter in parameters] + (
            ["body"] if operation.request_body else []
        )

        if clashing := sorted(
            {argument for argument in names if names.count(argument) > 1}
        ):
            raise ValueError(
                f"The operation {name} names several arguments {', '.join(clashing)}"
            )

        text = "\n\n".join(filter(None, (operation.summary, operation.description)))

        async def call(**arguments: JsonValue) -> object:
            return await send(self, method, path, parameters, arguments)

        return SchemaTool(
            name=f"{self.prefix}{name}",
            description=f"{method} {path}\n\n{text}".strip(),
            schema=operation_schema(parameters, operation.request_body),
            effect="read" if method in ("GET", "HEAD") else "write",
            call=call,
        )
