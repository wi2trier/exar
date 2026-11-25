"""Helpers shared by the memories: digests, the default measure, the ranking of searches, and the authorization of writes."""

import asyncio
import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from heapq import nlargest

from ..agents import approve
from ..policy import authorize
from ..tools import tool_action
from ..typing import Action, Changes, Decision, Hit, Measure

__all__ = ["digest", "top_hits", "word_coverage"]

_WORDS = re.compile(r"\w+")
_RRF_K = 60
"""Smoothing constant of reciprocal rank fusion, larger values narrow the gap between top ranks."""


def digest(text: str) -> str:
    """The SHA-256 hex digest of a text, which marks what an index holds.

    Examples:
        >>> digest("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    return hashlib.sha256(text.encode()).hexdigest()


def word_coverage(text: str, query: str, /) -> float:
    """Share of the distinct words of the query that the text contains, ignoring case.

    It is asymmetric: a long text that contains every word of a short query is fully similar to it.

    Examples:
        >>> word_coverage("The cat sat on the mat", "Cat dog")
        0.5
        >>> word_coverage("anything", "")
        0.0
    """
    words = set(_WORDS.findall(query.casefold()))

    if not words:
        return 0.0

    return len(words & set(_WORDS.findall(text.casefold()))) / len(words)


def top_hits[K, V](
    scores: Mapping[K, float], values: Mapping[K, V], limit: int | None
) -> list[Hit[K, V]]:
    """The entries of the highest scores as hits, ties in mapping order, all if `limit` is None."""
    ranked = (
        sorted(scores, key=scores.__getitem__, reverse=True)
        if limit is None
        else nlargest(limit, scores, key=scores.__getitem__)
    )

    return [Hit(key=key, value=values[key], score=scores[key]) for key in ranked]


def _fuse[K, V](
    rankings: Iterable[Sequence[Hit[K, V]]],
    /,
    *,
    weights: Sequence[float] | None = None,
    limit: int,
) -> list[Hit[K, V]]:
    """Fuses rankings by weighted reciprocal rank fusion, an entry scores the sum of `weight / (60 + rank)`.

    Ranks stay comparable when scores are not, such as across several queries or memories.

    Examples:
        >>> fused = _fuse(
        ...     [
        ...         [Hit(key="a", value=1, score=0.9), Hit(key="b", value=2, score=0.1)],
        ...         [Hit(key="b", value=2, score=0.8), Hit(key="c", value=3, score=0.5)],
        ...     ],
        ...     limit=2,
        ... )
        >>> [hit.key for hit in fused]
        ['b', 'a']
    """
    scores: dict[K, float] = {}
    values: dict[K, V] = {}
    weighted = (
        zip(rankings, weights, strict=True)
        if weights is not None
        else ((ranking, 1.0) for ranking in rankings)
    )

    for ranking, weight in weighted:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.key] = scores.get(hit.key, 0.0) + weight / (_RRF_K + rank)
            values.setdefault(hit.key, hit.value)

    return top_hits(scores, values, limit)


def _rank[K, V](
    cases: Mapping[K, V],
    queries: Sequence[str],
    text: Callable[[V], str],
    measure: Measure,
    limit: int,
) -> list[Hit[K, V]]:
    texts = {key: text(value) for key, value in cases.items()}
    rankings = [
        top_hits(
            {key: float(measure(value, query)) for key, value in texts.items()},
            cases,
            limit if len(queries) == 1 else None,
        )
        for query in queries
    ]

    return rankings[0] if len(rankings) == 1 else _fuse(rankings, limit=limit)


async def _search[K, V](
    cases: Mapping[K, V],
    queries: Sequence[str],
    text: Callable[[V], str],
    measure: Measure,
    limit: int,
) -> list[Hit[K, V]]:
    """The entries whose text the measure scores highest for the queries, in a worker thread.

    The rankings of several queries are fused by reciprocal rank fusion, see `_fuse`.
    """
    return await asyncio.to_thread(lambda: _rank(cases, queries, text, measure, limit))


async def _authorize[K, V](
    changes: Changes[K, V], decision_of: Callable[[Action], Decision], name: str
) -> None:
    """Lets every change pass if its decision allows it or asks and the approver of the run approves it.

    Each change is an action of kind `"tool"` named after the memory with effect `"write"` and its key as target,
    and outside a run nothing that is asked about passes, see `exar.agents.approve`.

    Raises:
        exar.errors.Refused: If a change does not pass.
    """
    for change in changes.items:
        action = tool_action(
            name, "write", target=None if change.key is None else str(change.key)
        )
        await authorize(decision_of(action), action, approve)
