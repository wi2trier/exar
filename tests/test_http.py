import json
from collections.abc import AsyncIterator

import httpx2
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from exar.pydantic_ai import EFFECT, toolset
from exar.tools import Retry
from exar.tools.http import OpenApi, OpenApiDocument
from exar.tools.web import HostClient

DOCUMENT = {
    "paths": {
        "/pets/{id}": {
            "parameters": [
                {
                    "name": "id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                }
            ],
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet",
                "parameters": [
                    {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                    {"name": "X-Version", "in": "header", "schema": {"type": "string"}},
                    {
                        "name": "Authorization",
                        "in": "header",
                        "schema": {"type": "string"},
                    },
                ],
            },
            "delete": {"operationId": "deletePet"},
        },
        "/pets": {
            "post": {
                "operationId": "addPet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Pet"}
                        }
                    },
                },
            }
        },
    },
    "components": {
        "schemas": {
            "Pet": {"type": "object", "properties": {"name": {"type": "string"}}}
        }
    },
}


class Streamed(httpx2.AsyncByteStream):
    """A body that is only read by streaming it, as one from the network is."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body


def api(
    *operations: str,
    max_response_bytes: int = 1000,
    headers: dict[str, str] | None = None,
) -> tuple[OpenApi, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)

        if request.method == "DELETE":
            return httpx2.Response(404, text="gone")

        return httpx2.Response(
            200,
            headers={"content-type": "application/json"},
            stream=Streamed(b'{"ok": true}'),
        )

    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond), headers=headers
    )

    return OpenApi(
        client=HostClient("api", lambda: client),
        base_url="https://api.example/v1/",
        document=OpenApiDocument.model_validate(DOCUMENT),
        operations=operations,
        max_response_bytes=max_response_bytes,
    ), seen


async def test_operations_become_tools() -> None:
    openapi, seen = api("getPet", "addPet", "deletePet")
    tools = {tool.name: tool for tool in openapi.tools()}

    assert {name: tool.effect for name, tool in tools.items()} == {
        "getPet": "read",
        "addPet": "write",
        "deletePet": "write",
    }
    assert tools["getPet"].schema["required"] == ["id"]
    assert tools["addPet"].schema["properties"] == {
        "body": {"type": "object", "properties": {"name": {"type": "string"}}}
    }

    assert await tools["getPet"].call(id=7, verbose=True) == {"ok": True}
    assert str(seen[0].url) == "https://api.example/v1/pets/7?verbose=true"

    assert await tools["addPet"].call(body={"name": "a"}) == {"ok": True}
    assert json.loads(seen[1].content) == {"name": "a"}

    with pytest.raises(Retry, match="missing required arguments id"):
        await tools["getPet"].call()

    with pytest.raises(Retry, match="unknown arguments x"):
        await tools["getPet"].call(id=1, x=2)

    with pytest.raises(Retry, match="HTTP 404: gone"):
        await tools["deletePet"].call(id=1)


async def test_sends_headers_and_keeps_path_segments() -> None:
    openapi, seen = api("getPet")
    [tool] = openapi.tools()

    assert tool.schema["required"] == ["id"]
    assert tool.schema["properties"] == {
        "id": {"type": "integer", "description": ""},
        "verbose": {"type": "boolean", "description": ""},
        "X-Version": {"type": "string", "description": ""},
    }
    assert await tool.call(id="a/b", **{"X-Version": "2"}) == {"ok": True}
    assert seen[0].url.raw_path == b"/v1/pets/a%2Fb"
    assert seen[0].headers["X-Version"] == "2"

    for climb in ("..", "."):
        with pytest.raises(Retry, match="path segment"):
            await tool.call(id=climb)


async def test_host_headers_win_over_arguments() -> None:
    openapi, seen = api("getPet", headers={"x-version": "host"})
    [tool] = openapi.tools()

    assert await tool.call(id=1, **{"X-Version": "model"}) == {"ok": True}
    assert seen[0].headers.get_list("X-Version") == ["host"]


async def test_cuts_response_and_refuses_unknown_operation() -> None:
    openapi, _ = api("getPet", max_response_bytes=5)

    assert await openapi.tools()[0].call(id=1) == '{"ok"\n\n(response cut at 5 bytes)'

    with pytest.raises(ValueError, match="nope"):
        api("nope")[0].tools()


async def test_toolset_carries_effects() -> None:
    served = toolset(api("getPet", "addPet")[0].tools())
    agent = Agent(TestModel(call_tools=["getPet"]), toolsets=[served])
    result = await agent.run("go")

    assert {
        name: tool.metadata[EFFECT]
        for name, tool in served.tools.items()
        if tool.metadata
    } == {"getPet": "read", "addPet": "write"}
    assert "ok" in result.output
