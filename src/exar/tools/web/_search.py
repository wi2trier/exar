"""The searches of `WebSearch` over HTTPX, which is imported only when the tool is built or called."""

import logging
from collections.abc import Sequence

from pydantic import ValidationError
from pydantic_ai import ToolReturn

from ...helpers import bounded_gather
from ...text import hint_suffix, plural
from .. import BatchData, Retry
from ..batch import BatchShare, serve_batch, tool_return
from ..files import BLOCK_SEP
from ..transport import request
from ._fields import WebSearchFields
from .models import WebQuery, WebSearchHit, WebSearchResults

logger = logging.getLogger(__name__)


def _label(search: WebQuery) -> str:
    """The query with the filters that tell it apart from another search."""
    category = search.category if search.category != "general" else None
    filters = ", ".join(
        value for value in (category, search.language, search.time_range) if value
    )

    return f"{search.query} [{filters}]" if filters else search.query


def _search_output(
    results: WebSearchResults, *, repeated: int
) -> ToolReturn[WebSearchResults]:
    """Renders one search's pages as numbered blocks.

    `repeated` counts the pages left out as already listed by an earlier search,
    which is said so an empty block is not read as an empty search.
    """
    blocks: list[str] = []

    for i, hit in enumerate(results.hits, 1):
        block = f"[{i}] {hit.title} ({hit.href})"

        if hit.published:
            block += f"\n    published {hit.published}"

        if hit.body:
            block += f"\n    {hit.body}"

        blocks.append(block)

    note = f"{repeated} {plural(repeated, 'page')} already listed above"

    if not blocks:
        formatted = f"(only {note})" if repeated else "(no results)"
    else:
        formatted = BLOCK_SEP.join(blocks) + hint_suffix([note] if repeated else [])

    return tool_return(results, formatted)


_FAILED = (
    "web search failed, the language code may be unknown or the "
    "search service unavailable; check the arguments and try again."
)


async def _hits(
    config: WebSearchFields, search: WebQuery, max_results: int
) -> tuple[WebSearchHit, ...]:
    """Runs one search against the backend, refusing an answer of more than `max_response_bytes`."""
    backend = config.backend
    params = backend.params(search, max_results)
    response = await request(
        config.client,
        "GET",
        backend.endpoint,
        timeout_seconds=config.timeout_seconds,
        max_bytes=config.max_response_bytes,
        params={key: value for key, value in params.items() if value is not None},
    )

    if response.is_error:
        logger.warning(
            "Web search failed for query %r: HTTP %s", search.query, response.status
        )
        raise Retry(_FAILED)

    if response.truncated:
        raise Retry(
            f"the search service answered with more than {config.max_response_bytes} bytes, "
            "ask for fewer results."
        )

    try:
        page = backend.parse(response.body)
    except ValidationError as exc:
        logger.warning("Web search failed for query %r: %s", search.query, exc)
        raise Retry(_FAILED) from exc

    if not page.hits and page.failed:
        raise Retry(
            f"no search engine answered ({', '.join(page.failed)}); try again "
            "later or rephrase the query."
        )

    return page.hits[:max_results]


async def run_searches(
    config: WebSearchFields, searches: Sequence[WebQuery], max_results: int
) -> ToolReturn[BatchData[WebSearchResults]]:
    """The results of the searches, run together."""
    unique = list(dict.fromkeys(searches))

    async def hits(search: WebQuery) -> tuple[WebSearchHit, ...] | Retry:
        try:
            return await _hits(config, search, max_results)
        except Retry as exc:
            return exc

    # Fetched together, then rendered in request order, so which search
    # a page is listed under does not depend on which answered first.
    fetched = await bounded_gather(unique, hits, limit=config.concurrency)
    found = dict(zip(unique, fetched, strict=True))
    seen: set[str] = set()

    async def render(
        search: WebQuery, _share: BatchShare
    ) -> ToolReturn[WebSearchResults]:
        outcome = found[search]

        if isinstance(outcome, Retry):
            raise outcome

        fresh = tuple(hit for hit in outcome if hit.href not in seen)
        seen.update(hit.href for hit in fresh)
        results = WebSearchResults(search.query, fresh)

        return _search_output(results, repeated=len(outcome) - len(fresh))

    output = await serve_batch(unique, render, key=_label)
    config.session().known.add_results(output.metadata)

    return output
