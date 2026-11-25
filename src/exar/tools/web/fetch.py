"""`WebFetch`, which fetches pages as readable text numbered by line."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic_ai import ToolReturn

from ..._lazy import extra_hint
from ...schema import serializable
from .. import BatchData
from ..common import FullLinesArg
from ._fields import WebFetchFields
from .models import WebPage, WebReadsArg

__all__ = ["WebFetch"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class WebFetch(WebFetchFields):
    """Fetches web pages and returns their readable content, a tool of effect `"read"`.

    Only a URL in the known URLs of the session is fetched, and every page a fetched page links to joins it.
    The session also holds the pages of the run, so reading on from an offset neither downloads nor extracts a page again.
    A call keeps at most `concurrency` requests in flight.

    Args:
        client: The client that fetches, bound by the host.
            Its request hook should validate the URL and every redirect hop, which HTTPX configures at the client with the hop limit.
            `exar.tools.transport.public_client` reaches public addresses alone, without which a fetched page can link to internal hosts.
        session: The session of the run, bound by the host.
        timeout_seconds: Bound of each page download as a whole.
        max_response_bytes: Raw bytes downloaded per page.
        max_chars: Characters of the window of lines shown, which a call fetching several pages splits between them.
        max_line_chars: Characters per line before it is clipped, which keeps a data URI or a minified line from flooding the context.
        user_agent: What to send as the User-Agent, that of the client if None.
        fast: Whether the default extractor skips the backup extractors that `trafilatura` compares its own result with.
        notes: What to add to the description after it, such as which proxy the client connects through.
        html: Reduces HTML to its main content, `trafilatura` with `fast` if None.
        size: Writes the byte count of the download cap in a hint.
    """

    extra: ClassVar[str | None] = "web"
    kind: Literal["web_fetch"] = field(default="web_fetch", repr=False)
    name: str = "web_fetch"

    async def __call__(
        self, pages: WebReadsArg, full_lines: FullLinesArg = False
    ) -> ToolReturn[BatchData[WebPage]]:
        """Fetch one or more web pages as readable text.

        Only a URL that appeared in the conversation can be fetched: one the
        user wrote, a search result, or a link on a fetched page.
        HTML is reduced to its main content as markdown, without navigation,
        ads, or comments.  Plain-text and JSON responses pass through
        unchanged.  Each content line is numbered
        so it can be cited like a document line, and a long page is shown
        from ``offset`` as far as the budget shared by the pages allows, with
        the offset to read on from.  Redirects are followed
        by the client, whose request hook checks every hop against the URL
        host policy.  Each page is shown
        under the URL it was asked for, and a URL redirecting to a page
        already fetched is not shown twice.
        """
        with extra_hint("WebFetch", "web"):
            from ._fetch import fetch

        return await fetch(self, pages, full_lines)
