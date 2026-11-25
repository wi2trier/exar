from dataclasses import replace

import httpx2
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Tool

from exar.helpers import get_name
from exar.tools import ItemFailure, Retry, effect_of
from exar.tools.extract import Extract
from exar.tools.web import (
    HostClient,
    HostSession,
    Openserp,
    Searxng,
    WebFetch,
    WebQuery,
    WebRead,
    WebSearch,
    WebSession,
)


def bind(handler: httpx2.MockTransport) -> tuple[HostClient, HostSession, WebSession]:
    client = httpx2.AsyncClient(transport=handler)
    session = WebSession()

    return (
        HostClient("web", lambda: client),
        HostSession("web", lambda: session),
        session,
    )


async def test_search_then_fetch() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/search":
            hits = [
                {"url": "https://a.example/", "title": " A ", "content": "x"},
                {"url": "javascript:alert(1)", "title": "Unsafe"},
            ]

            return httpx2.Response(200, json={"results": hits})

        return httpx2.Response(
            200,
            text="<p>Hello</p>\nsee https://b.example/more.",
            headers={"content-type": "text/html"},
        )

    client, session, state = bind(httpx2.MockTransport(respond))
    search = WebSearch(client=client, session=session, backend=Searxng(url="http://s"))
    fetch = WebFetch(
        client=client,
        session=session,
        html=lambda body, url: Extract(
            "Title", "Hello", "2026-01-02", ("https://b.example/more",)
        ),
    )

    assert (get_name(search.tool()), effect_of(search.tool())) == ("web_search", "read")

    with pytest.raises(Retry, match="has not appeared"):
        await fetch.tool()([WebRead("https://a.example/")])

    found = await search([WebQuery("a")])

    assert found.return_value == "==> a <==\n[1] A (https://a.example/)\n    x"

    with pytest.raises(Retry, match="more than 10 bytes"):
        await replace(search, max_response_bytes=10)([WebQuery("a")])
    assert "https://a.example/" in state.known

    page = await fetch([WebRead("https://a.example/")])

    assert (
        page.return_value
        == "==> https://a.example/ <==\nTitle (https://a.example/), published 2026-01-02\n1: Hello"
    )
    assert "https://b.example/more" in state.known


async def test_fetch_failures_and_paging() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/gone":
            return httpx2.Response(404)

        return httpx2.Response(
            200, text="one\ntwo\nthree", headers={"content-type": "text/plain"}
        )

    client, session, state = bind(httpx2.MockTransport(respond))
    fetch = WebFetch(client=client, session=session, max_chars=26)
    state.known.add("https://a.example/", "https://a.example/gone")

    result = await fetch(
        [WebRead("https://a.example/"), WebRead("https://a.example/gone")]
    )
    page, gone = result.metadata

    assert not isinstance(page, ItemFailure)
    assert page.content == "one\ntwo"
    assert gone == ItemFailure(item="https://a.example/gone", reason="HTTP 404.")
    assert "call again with offset=3" in str(result.return_value)


def test_backends_are_serialized_by_kind() -> None:
    client, session, _ = bind(httpx2.MockTransport(lambda _: httpx2.Response(200)))
    tool = WebSearch(
        client=client, session=session, backend=Openserp(url="http://s/base")
    )
    adapter = TypeAdapter(WebSearch)
    data = adapter.dump_python(tool, mode="json")

    assert data["backend"] == {"kind": "openserp", "url": "http://s/base"}
    assert adapter.validate_python(data).backend == tool.backend
    assert tool.backend.endpoint == "http://s/base/mega/search"


async def test_default_extractor_reads_main_content() -> None:
    pytest.importorskip("trafilatura")
    page = (
        "<html><head><title>T</title></head><body><article><p>"
        + "Words of the article. " * 20
        + '<a href="/next">more</a></p></article></body></html>'
    )
    client, session, state = bind(
        httpx2.MockTransport(
            lambda _: httpx2.Response(
                200, text=page, headers={"content-type": "text/html"}
            )
        )
    )
    state.known.add("https://a.example/")

    result = await WebFetch(client=client, session=session)(
        [WebRead("https://a.example/")]
    )
    fetched = result.metadata[0]

    assert not isinstance(fetched, ItemFailure)
    assert "Words of the article." in fetched.content
    assert "https://a.example/next" in fetched.links


def test_notes_extend_the_description() -> None:
    client, session, _ = bind(httpx2.MockTransport(lambda _: httpx2.Response(200)))
    description = Tool(
        WebFetch(client=client, session=session, notes="Behind a proxy.").tool()
    ).description

    assert description is not None
    assert description.endswith("\n\nBehind a proxy.")
