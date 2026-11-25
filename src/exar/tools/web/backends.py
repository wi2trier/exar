"""The metasearch services behind `WebSearch`."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Annotated, ClassVar, Literal, Protocol, override
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ...schema import Open, serializable
from .models import WebQuery, WebSearchHit, WebTimeRange

__all__ = [
    "AnySearchBackend",
    "Openserp",
    "SearchBackend",
    "SearchBackendKind",
    "SearchPage",
    "Searxng",
]


def _hit(
    url: str | None, title: str, body: str, published: str | None = None
) -> WebSearchHit | None:
    """Shapes one result, `None` for one without a link the model could fetch."""
    if url is None or not url.startswith(("https://", "http://")):
        return None

    return WebSearchHit(
        title=" ".join(title.split()),
        href=url,
        body=" ".join(body.split()),
        published=published[:10] if published else None,
    )


@dataclass(slots=True, frozen=True)
class SearchPage:
    """What a backend answered one search with.

    `failed` describes the engines that did not answer, which explains a search that found nothing.
    """

    hits: tuple[WebSearchHit, ...]
    failed: tuple[str, ...] = ()


class SearchBackend(Protocol):
    """A metasearch service `WebSearch` queries, which translates a search into a request and its response into a `SearchPage`.

    The `Searxng` and `Openserp` backends are the ones a spec names, and Python code passes any other.
    """

    @property
    def endpoint(self) -> str:
        """The URL a search is sent to."""
        ...

    def params(self, search: WebQuery, max_results: int) -> dict[str, str | None]:
        """The query parameters that run a search, `None` for one left out."""
        ...

    def parse(self, content: bytes) -> SearchPage:
        """Reads a response body, raising `ValidationError` on a foreign one."""
        ...


@dataclass(slots=True, frozen=True, kw_only=True)
class _Backend(ABC):
    """A metasearch service `WebSearch` queries over its HTTP JSON API.

    A backend only translates: a search into a request to its endpoint, and the response body into a `SearchPage`.
    The tool owns the request, its errors, and everything done with the results.

    Args:
        url: Base URL of the service, an endpoint of the operator, never chosen by the model.
    """

    path: ClassVar[str]
    url: str

    @property
    def endpoint(self) -> str:
        """The URL a search is sent to, `path` resolved below `url`."""
        base = urlsplit(self.url)

        return base._replace(path=str(PurePosixPath(base.path, self.path))).geturl()

    @abstractmethod
    def params(self, search: WebQuery, max_results: int) -> dict[str, str | None]:
        """The query parameters that run a search, `None` for one left out."""

    @abstractmethod
    def parse(self, content: bytes) -> SearchPage:
        """Reads a response body, raising `ValidationError` on a foreign one."""


class _SearxngResult(BaseModel):
    url: str | None = None
    title: str = ""
    content: str = ""
    published_date: str | None = Field(default=None, alias="publishedDate")


class _SearxngResponse(BaseModel):
    results: tuple[_SearxngResult, ...] = ()
    unresponsive_engines: tuple[tuple[str, str], ...] = ()


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Searxng(_Backend):
    """SearXNG, which must list `json` under its `search.formats`.

    It honours every filter, and a search naming no language takes the instance's `search.default_lang`.
    """

    path: ClassVar[str] = "search"
    kind: Literal["searxng"] = field(default="searxng", repr=False)

    @override
    def params(self, search: WebQuery, max_results: int) -> dict[str, str | None]:
        return {
            "q": search.query,
            "format": "json",
            "categories": search.category,
            "language": search.language,
            "time_range": search.time_range,
        }

    @override
    def parse(self, content: bytes) -> SearchPage:
        payload = _SearxngResponse.model_validate_json(content)
        hits = (
            _hit(result.url, result.title, result.content, result.published_date)
            for result in payload.results
        )

        return SearchPage(
            hits=tuple(hit for hit in hits if hit),
            failed=tuple(
                f"{engine}: {error}" for engine, error in payload.unresponsive_engines
            ),
        )


class _OpenserpResult(BaseModel):
    url: str | None = None
    title: str = ""
    snippet: str = ""


class _OpenserpMeta(BaseModel):
    engines_failed: tuple[str, ...] = ()


class _OpenserpResponse(BaseModel):
    results: tuple[_OpenserpResult, ...] = ()
    meta: _OpenserpMeta = _OpenserpMeta()


_TIME_RANGE_DAYS: dict[WebTimeRange, int] = {
    "day": 1,
    "week": 7,
    "month": 31,
    "year": 365,
}


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Openserp(_Backend):
    """OpenSERP's megasearch, which merges the results of all its engines.

    It knows no categories, so every search covers the general web, and its results carry no publication date.
    A language such as `en-US` becomes its `lang` and `region`, and a time range the dates up to today.
    """

    path: ClassVar[str] = "mega/search"
    kind: Literal["openserp"] = field(default="openserp", repr=False)

    @override
    def params(self, search: WebQuery, max_results: int) -> dict[str, str | None]:
        language, _, region = (search.language or "").partition("-")
        today = datetime.now(UTC).date()
        days = _TIME_RANGE_DAYS[search.time_range] if search.time_range else None
        since = (
            f"{today - timedelta(days=days):%Y%m%d}..{today:%Y%m%d}" if days else None
        )

        return {
            "text": search.query,
            "limit": str(max_results),
            "lang": language.upper() or None,
            "region": region or None,
            "date": since,
        }

    @override
    def parse(self, content: bytes) -> SearchPage:
        payload = _OpenserpResponse.model_validate_json(content)
        hits = (
            _hit(result.url, result.title, result.snippet) for result in payload.results
        )

        return SearchPage(
            hits=tuple(hit for hit in hits if hit),
            failed=payload.meta.engines_failed,
        )


type SearchBackendKind = Annotated[Searxng | Openserp, Field(discriminator="kind")]
"""The backends of `WebSearch` a spec names."""

type AnySearchBackend = Open[SearchBackendKind, SearchBackend]
"""A `SearchBackendKind` or any other `SearchBackend` of Python code, which serialized forms and JSON Schemas only know as `SearchBackendKind`."""
