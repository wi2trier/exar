"""The requests of `OpenApi` over HTTPX, which is imported only when the tools are built or called."""

import json
from typing import NamedTuple
from urllib.parse import quote

from pydantic import JsonValue

from . import Retry
from ._openapi import OpenApiFields, OpenApiParameter
from .transport import Response, client_of, media_type, request


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value)


class Request(NamedTuple):
    """What `_build` makes of the arguments of an operation."""

    url: str
    query: dict[str, str | list[str]]
    headers: dict[str, str]


def _segment(parameter: OpenApiParameter, value: JsonValue) -> str:
    """The value of a path parameter as one encoded segment, which may not climb to another path."""
    segment = quote(_text(value), safe="")

    if segment in ("", ".", ".."):
        raise Retry(
            f"argument {parameter.name} must name a path segment, not {segment!r}."
        )

    return segment


def _build(
    config: OpenApiFields,
    path: str,
    parameters: list[OpenApiParameter],
    arguments: dict[str, JsonValue],
) -> Request:
    """The request of the arguments, which fails with `Retry` if they do not fit the parameters.

    A path parameter is required whatever the document says, as OpenAPI demands, and is a single segment.
    """
    missing = [
        parameter.name
        for parameter in parameters
        if (parameter.required or parameter.location == "path")
        and arguments.get(parameter.name) is None
    ]

    if missing:
        raise Retry(f"missing required arguments {', '.join(missing)}.")

    if unknown := sorted(
        arguments.keys() - {parameter.name for parameter in parameters} - {"body"}
    ):
        raise Retry(f"unknown arguments {', '.join(unknown)}.")

    given = [
        (parameter, value)
        for parameter in parameters
        if (value := arguments.get(parameter.name)) is not None
    ]

    for parameter, value in given:
        if parameter.location == "path":
            path = path.replace(f"{{{parameter.name}}}", _segment(parameter, value))

    query = {
        parameter.name: [_text(item) for item in value]
        if isinstance(value, list)
        else _text(value)
        for parameter, value in given
        if parameter.location == "query"
    }
    headers = {
        parameter.name: _text(value)
        for parameter, value in given
        if parameter.location == "header"
    }

    if cookies := [
        f"{parameter.name}={quote(_text(value), safe='')}"
        for parameter, value in given
        if parameter.location == "cookie"
    ]:
        headers["Cookie"] = "; ".join(cookies)

    return Request(config.base_url.rstrip("/") + path, query, headers)


def _decode(config: OpenApiFields, method: str, response: Response) -> object:
    """The response as JSON, text, or the headers of a `HEAD`, which fails with `Retry` on an error status.

    The body was streamed, so it is parsed from the bytes read, and a JSON body that does not parse is returned as text.
    """
    text = response.text

    if response.is_error:
        raise Retry(f"HTTP {response.status}: {text}")

    if method == "HEAD":
        return dict(response.headers)

    if response.truncated:
        return f"{text}\n\n(response cut at {config.max_response_bytes} bytes)"

    if not (media_type(response.headers).endswith("json") and text):
        return text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def send(
    config: OpenApiFields,
    method: str,
    path: str,
    parameters: list[OpenApiParameter],
    arguments: dict[str, JsonValue],
) -> object:
    """Sends a request built from the arguments and returns the response as JSON, text, or the headers of a `HEAD`.

    A header the client of the host sets, such as its credentials, is never replaced by an argument of the model.
    """
    built = _build(config, path, parameters, arguments)
    client = client_of(config.client)
    response = await request(
        config.client,
        method,
        built.url,
        timeout_seconds=config.timeout_seconds,
        max_bytes=config.max_response_bytes,
        params=built.query,
        headers={
            name: value
            for name, value in built.headers.items()
            if name not in client.headers
        },
        json=arguments.get("body"),
    )

    return _decode(config, method, response)
