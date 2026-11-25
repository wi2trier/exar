"""Tools that search the web and fetch its pages, configured as serializable classes.

Like the tools of `exar.tools.files`, each is a keyword-only dataclass whose `__call__` is the tool,
and `tool` turns it into the plain function that agent frameworks register, with an effect of `"read"`.
What depends on the host is not serialized but bound by name: the HTTP client (`HostClient`),
which is where a host enforces its URL policy, proxies, and redirect limit,
and what a run has seen of the web (`HostSession`).
Results are pydantic-ai `ToolReturn`s, the text for the model and the typed data for programs and clients.
The tools need the `pydantic-ai` extra and, only once they are built or called, the `web` extra.

- `WebSearch`: Searches the web through a metasearch service (`Searxng`, `Openserp`).
- `WebFetch`: Fetches pages as readable text numbered by line, which the model reads on from an offset.

A fetch opens only a URL in the `KnownUrls` of the session, which holds what the user wrote and what the web tools returned,
so an instruction planted on a page cannot make the model spell a URL that carries data out.
HTML is reduced to its main content by `trafilatura` and rendered as markdown, or by the `html` extractor of the host.
"""

from .._host import Client, HostClient
from .backends import (
    AnySearchBackend,
    Openserp,
    SearchBackend,
    SearchBackendKind,
    SearchPage,
    Searxng,
)
from .fetch import WebFetch
from .models import (
    WebCategoryArg,
    WebLanguageArg,
    WebMaxResultsArg,
    WebPage,
    WebQuery,
    WebQueryArg,
    WebRead,
    WebReadsArg,
    WebSearchesArg,
    WebSearchHit,
    WebSearchResults,
    WebTimeRange,
    WebTimeRangeArg,
    WebUrlArg,
)
from .search import WebSearch
from .session import (
    FetchedPages,
    HostSession,
    KnownUrls,
    Page,
    WebSession,
)

__all__ = [
    "AnySearchBackend",
    "Client",
    "FetchedPages",
    "HostClient",
    "HostSession",
    "KnownUrls",
    "Openserp",
    "Page",
    "SearchBackend",
    "SearchBackendKind",
    "SearchPage",
    "Searxng",
    "WebCategoryArg",
    "WebFetch",
    "WebLanguageArg",
    "WebMaxResultsArg",
    "WebPage",
    "WebQuery",
    "WebQueryArg",
    "WebRead",
    "WebReadsArg",
    "WebSearch",
    "WebSearchHit",
    "WebSearchResults",
    "WebSearchesArg",
    "WebSession",
    "WebTimeRange",
    "WebTimeRangeArg",
    "WebUrlArg",
]
