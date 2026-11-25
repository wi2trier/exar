"""The arguments and results of the web tools."""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from ..batch import batch_field
from ..common import OffsetArg

__all__ = [
    "WebCategoryArg",
    "WebLanguageArg",
    "WebMaxResultsArg",
    "WebPage",
    "WebQuery",
    "WebQueryArg",
    "WebRead",
    "WebReadsArg",
    "WebSearchHit",
    "WebSearchResults",
    "WebSearchesArg",
    "WebTimeRange",
    "WebTimeRangeArg",
    "WebUrlArg",
]

WebTimeRange = Literal["day", "week", "month", "year"]

WebQueryArg = Annotated[
    str,
    Field(description="Search query string, phrased in the language of the results."),
]

WebCategoryArg = Annotated[
    Literal["general", "news", "science", "it"],
    Field(
        description=(
            "Kind of source to search: `news` for current events, `science` for "
            "papers, `it` for programming, and `general` for everything else."
        )
    ),
]

WebLanguageArg = Annotated[
    str | None,
    Field(
        description=(
            "Language of the results as a code like `en`, `de`, or `en-US`. "
            "Search another one (e.g. English or the user's language) when "
            "results are poor. Omit it for the search service's default."
        ),
        pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$",
    ),
]

WebTimeRangeArg = Annotated[
    WebTimeRange | None,
    Field(
        description=(
            "Only find pages from the last day, week, month, or year, for recent "
            "events. Omit it to search all time."
        )
    ),
]

WebMaxResultsArg = Annotated[
    int,
    Field(description="Maximum number of results to return per search.", ge=1, le=20),
]

WebUrlArg = Annotated[str, Field(description="HTTP or HTTPS URL to fetch.")]


@dataclass(slots=True, frozen=True)
class WebRead:
    """One page to fetch, and the line of its text to read on from."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    url: WebUrlArg
    offset: OffsetArg = 1

    @property
    def key(self) -> str:
        """The URL, plus the offset when it is not the first line.

        Examples:
            >>> WebRead("https://a.example", offset=40).key
            'https://a.example (offset=40)'
        """
        return self.url if self.offset == 1 else f"{self.url} (offset={self.offset})"


WebReadsArg = Annotated[
    list[WebRead],
    batch_field(
        "Pages to fetch, each with an optional offset. Fetch every page you "
        "need in one call, since the output budget is shared between them, "
        "and a page that fails is reported without failing the others."
    ),
]


@dataclass(slots=True, frozen=True)
class WebQuery:
    """One search: the query, and the filters to run it with."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    query: WebQueryArg
    category: WebCategoryArg = "general"
    language: WebLanguageArg = None
    time_range: WebTimeRangeArg = None


WebSearchesArg = Annotated[
    list[WebQuery],
    batch_field(
        "Searches to run. Search several phrasings, languages, or categories in "
        "one call rather than one call each, and a page several of them find is "
        "listed once."
    ),
]


@dataclass(slots=True, frozen=True)
class WebSearchHit:
    """One page a search found, with its snippet and publication date."""

    title: str
    href: str
    body: str
    published: str | None = None


@dataclass(slots=True, frozen=True)
class WebSearchResults:
    """The pages one search found that no earlier search of the call had."""

    query: str
    hits: tuple[WebSearchHit, ...]

    @property
    def urls(self) -> tuple[str, ...]:
        """The pages the model may fetch next."""
        return tuple(hit.href for hit in self.hits)


@dataclass(slots=True, frozen=True)
class WebPage:
    """The window of readable text a call shows of a fetched web page.

    `requested_url` is the URL asked for and `url` the final one after redirects.
    The page's text is its main content as markdown (for HTML) or its raw text (for plain-text and JSON responses),
    and `content` holds its lines `start_line` to `end_line` of `total_lines`.
    `published` is the date the page states it was published, if any,
    and `links` the absolute targets of all its links or the URLs its text spells out.
    """

    requested_url: str
    url: str
    title: str
    start_line: int
    end_line: int
    total_lines: int
    content: str
    truncated: bool
    published: str | None = None
    links: tuple[str, ...] = ()

    @property
    def urls(self) -> tuple[str, ...]:
        """The pages the model may fetch next."""
        return (self.url, *self.links)
