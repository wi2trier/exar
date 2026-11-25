"""The downloads of `WebFetch` over HTTPX, which is imported only when the tool is built or called."""

import asyncio
import logging
from collections.abc import Sequence
from functools import partial

import httpx2
from pydantic_ai import ToolReturn

from ...text import LineClip, fit_window, hint_suffix
from .. import BatchData, Retry
from ..batch import BatchShare, serve_batch, tool_return
from ..common import line_clip
from ..extract import Extract
from ..transport import Response, media_type, request
from ._fields import WebFetchFields
from .models import WebPage, WebRead
from .session import Page, spelled_urls

logger = logging.getLogger(__name__)


def _is_html(mime: str) -> bool:
    return mime in ("text/html", "application/xhtml+xml")


def _is_textual(mime: str) -> bool:
    return (
        mime.startswith("text/")
        or mime in ("application/json", "application/xml")
        or mime.endswith(("+json", "+xml"))
    )


def _extract(config: WebFetchFields, response: Response) -> Page:
    if _is_html(media_type(response.headers)):
        html = config.html

        if html is None:
            from .._html import extract_html

            html = partial(extract_html, fast=config.fast)

        extract = html(response.body, response.url)
    else:
        text = response.text.strip()
        extract = Extract("", text, links=spelled_urls(text, "://"))

    return Page(
        url=response.url,
        title=extract.title,
        lines=tuple(extract.text.splitlines()),
        published=extract.published,
        links=extract.links,
        truncated=response.truncated,
    )


def _check(response: httpx2.Response, /) -> None:
    """Refuses an error status or a content type that is no text before the body is downloaded."""
    if not response.is_success:
        raise Retry(f"HTTP {response.status_code}.")

    mime = media_type(response.headers)

    if not _is_textual(mime):
        raise Retry(f"unsupported content type '{mime}'.")


async def _download(config: WebFetchFields, url: str) -> Page:
    response = await request(
        config.client,
        "GET",
        url,
        timeout_seconds=config.timeout_seconds,
        max_bytes=config.max_response_bytes,
        headers=None
        if config.user_agent is None
        else {"User-Agent": config.user_agent},
        check=_check,
    )

    # Parsing a multi-megabyte page is CPU-bound, so it must not stall the
    # event loop every other request of the server shares.
    return await asyncio.to_thread(_extract, config, response)


def _window(
    config: WebFetchFields, read: WebRead, fetched: Page, clip: LineClip, max_chars: int
) -> ToolReturn[WebPage]:
    """The window of a fetched page that a read asks for."""
    window = fit_window(fetched.lines, read.offset, max_chars, clip)
    page = WebPage(
        requested_url=read.url,
        url=fetched.url,
        title=fetched.title,
        start_line=window.start,
        end_line=window.end,
        total_lines=window.total,
        content=window.content,
        truncated=fetched.truncated,
        published=fetched.published,
        links=fetched.links,
    )

    if not window.total:
        return tool_return(page, "(no readable text on this page)")

    if past_end := window.past_end("page"):
        return tool_return(page, past_end)

    header = f"{page.title} ({page.url})" if page.title else page.url

    if page.published:
        header += f", published {page.published}"

    hints = clip.hints() + window.hints()

    if page.truncated:
        cap = config.size(config.max_response_bytes)
        hints.append(f"page cut at the {cap} download cap")

    return tool_return(page, f"{header}\n{window.rendered}{hint_suffix(hints)}")


async def _page(
    config: WebFetchFields, read: WebRead, full_lines: bool, share: BatchShare
) -> ToolReturn[WebPage]:
    session = config.session()

    if read.url not in session.known:
        raise Retry(
            "this URL has not appeared in the conversation; fetch only URLs "
            "the user gave, search results, or links on fetched pages."
        )

    page = await session.pages.get(read.url, lambda: _download(config, read.url))
    clip = line_clip(config.max_line_chars, full_lines)

    return _window(config, read, page, clip, share.of(config.max_chars))


async def _page_or_retry(
    config: WebFetchFields, read: WebRead, full_lines: bool, share: BatchShare
) -> ToolReturn[WebPage]:
    """Fetches one page, turning every failure into a correctable refusal."""
    try:
        return await _page(config, read, full_lines, share)
    except Retry:
        raise
    except Exception as exc:
        logger.exception("Web fetch failed for URL %r", read.url)
        raise Retry("failed to fetch URL.") from exc


async def fetch(
    config: WebFetchFields, pages: Sequence[WebRead], full_lines: bool
) -> ToolReturn[BatchData[WebPage]]:
    """The windows of the pages, fetched together."""

    async def one(read: WebRead, share: BatchShare) -> ToolReturn[WebPage]:
        return await _page_or_retry(config, read, full_lines, share)

    output = await serve_batch(
        pages,
        one,
        key=lambda read: read.key,
        concurrency=config.concurrency,
        identity=lambda page: (page.url, page.start_line),
    )
    config.session().known.add_results(output.metadata)

    return output
