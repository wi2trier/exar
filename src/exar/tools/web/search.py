"""`WebSearch`, which searches the web through a metasearch service."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from pydantic_ai import ToolReturn

from ..._lazy import extra_hint
from ...schema import serializable
from .. import BatchData
from ._fields import WebSearchFields
from .models import WebMaxResultsArg, WebSearchesArg, WebSearchResults

__all__ = ["WebSearch"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class WebSearch(WebSearchFields):
    """Searches the web through a metasearch service, a tool of effect `"read"`.

    The backend sends each query to several search engines and merges their rankings, so no API key is needed here.
    Its API is an endpoint of the operator, reached with the client of the host,
    and the model only ever chooses the query, never a host the service connects to.
    The results link anywhere, and the URL policy of the fetching client decides which of them `WebFetch` may open.

    Args:
        client: The client that reaches the service, bound by the host.
        session: The session that learns the URLs of the results, bound by the host.
        backend: The service and how to talk to it, a `SearchBackendKind` of the spec or any `SearchBackend` of Python code.
        timeout_seconds: Bound of each search as a whole.
        max_response_bytes: Bytes of the answer of the service to a search that are read, a larger answer is refused.
    """

    extra: ClassVar[str | None] = "web"
    kind: Literal["web_search"] = field(default="web_search", repr=False)
    name: str = "web_search"

    async def __call__(
        self,
        searches: WebSearchesArg,
        max_results: WebMaxResultsArg = 5,
    ) -> ToolReturn[BatchData[WebSearchResults]]:
        """Search the web for up-to-date information.

        Every search runs through several search engines at once, so it
        finds news, documentation, papers, forums, and any other public
        page.  Pick the category, language, and time range that fit the
        question, e.g. the `news` category and a recent time range for
        current events.

        Returns each search's results with ``title``, ``href``, ``body``,
        and ``published`` fields, a page already found by an earlier search
        of the call left out of the later ones. ``body`` is only a short
        snippet of the page.
        """
        with extra_hint("WebSearch", "web"):
            from ._search import run_searches

        return await run_searches(self, searches, max_results)
