"""What the web tools bind to the host, its HTTP client and what a run has seen of the web."""

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

from ..._lazy import extra_hint
from .. import ItemFailure
from .._host import HostFactory
from .models import WebPage, WebSearchResults

__all__ = [
    "FetchedPages",
    "HostSession",
    "KnownUrls",
    "Page",
    "WebSession",
    "spelled_urls",
]

type _WebResult = WebSearchResults | WebPage | ItemFailure
"""One entry of a `WebSearch` or `WebFetch` batch."""

_PUNCTUATION = "<>[]{}\"'`*.,;:!?"
"""What may enclose or follow a URL in free text."""


def _url_key(url: str) -> str | None:
    """Normalizes a URL so two spellings of one page compare equal.

    A left out scheme means `https`, while a given one is kept, so a page known over `https` is never fetched over `http`.
    The fragment is ignored, the host is lowercased and non-ASCII characters are percent-encoded.
    `None` stands for a text that is no URL.

    Examples:
        >>> _url_key("Example.org/a#b")
        'https://example.org/a'
        >>> _url_key("http://example.org/a")
        'http://example.org/a'
    """
    with extra_hint("The web tools", "web"):
        import httpx2

    try:
        parsed = httpx2.URL(url)

        if not parsed.host:
            parsed = httpx2.URL(f"https://{url}")
    except httpx2.InvalidURL:
        return None

    return str(parsed.copy_with(fragment=None))


def spelled_urls(text: str, marker: str) -> tuple[str, ...]:
    """The words of a text holding the marker that may be URLs.

    Each word counts without the punctuation around it, once also without parentheses and once keeping them, as some URLs end with one.
    """
    return tuple(
        {
            variant
            for word in text.split()
            if marker in word
            for variant in (word.strip(_PUNCTUATION), word.strip(_PUNCTUATION + "()"))
        }
    )


@dataclass(slots=True)
class KnownUrls:
    """The URLs a run has seen from sources the model cannot write, which `WebFetch` may open.

    Examples:
        >>> known = KnownUrls()
        >>> known.add_text("Summarize de.wikipedia.org/wiki/Müller, please.")
        >>> "https://de.wikipedia.org/wiki/M%C3%BCller#intro" in known
        True
        >>> "https://de.wikipedia.org/wiki/M%C3%BCller?leak=1" in known
        False
        >>> "https://de.wikipedia.org/wiki/M" in known
        False
    """

    urls: set[str] = field(default_factory=set)

    def add(self, *urls: str) -> None:
        """Knows urls, skipping a text that is no URL."""
        self.urls.update(key for url in urls if (key := _url_key(url)))

    def add_text(self, text: str) -> None:
        """Knows every URL a free text such as a user prompt names, even without a scheme."""
        self.add(*spelled_urls(text, "."))

    def add_results(self, results: Iterable[_WebResult]) -> None:
        """Knows every URL the web tools returned in results."""
        for result in results:
            if not isinstance(result, ItemFailure):
                self.add(*result.urls)

    def __contains__(self, url: str) -> bool:
        return _url_key(url) in self.urls


@dataclass(slots=True, frozen=True)
class Page:
    """A fetched page's readable text, which every window of it is cut from."""

    url: str
    title: str
    lines: tuple[str, ...]
    published: str | None
    links: tuple[str, ...]
    truncated: bool


@dataclass(slots=True)
class FetchedPages:
    """The pages a run fetched, so reading on from an offset downloads none again.

    Each page is held as the task fetching it, so the reads of one page in a batch share a single download,
    and a failed or cancelled one is forgotten, so a later call tries again.
    At most `max_pages` are kept, the least recently read going first.
    """

    max_pages: int = 8
    _pages: OrderedDict[str, asyncio.Task[Page]] = field(
        default_factory=OrderedDict, init=False
    )

    async def get(self, url: str, fetch: Callable[[], Awaitable[Page]]) -> Page:
        """The page at the URL, fetched with `fetch` unless it already was."""
        key = _url_key(url) or url
        task = self._pages.get(key)

        if task is None or (
            task.done() and (task.cancelled() or task.exception() is not None)
        ):
            task = self._pages[key] = asyncio.ensure_future(fetch())

            if len(self._pages) > self.max_pages:
                self._pages.popitem(last=False)
        else:
            self._pages.move_to_end(key)

        # Shielded, so one cancelled read does not cancel the download the other reads of the page wait on.
        try:
            return await asyncio.shield(task)

        except BaseException:
            # A done task failed or was cancelled itself, while one still running outlives the cancel of this read.
            if task.done() and self._pages.get(key) is task:
                del self._pages[key]

            raise


@dataclass(slots=True)
class WebSession:
    """What a run has seen of the web, shared by its web tools, its subagents, and its programs."""

    known: KnownUrls = field(default_factory=KnownUrls)
    pages: FetchedPages = field(default_factory=FetchedPages)


@dataclass(slots=True, frozen=True)
class HostSession(HostFactory[WebSession]):
    """The session of the host, serialized by its name only.

    The bound function must return the same session throughout a run,
    so the URLs a search found are known to the fetch that follows.
    """
