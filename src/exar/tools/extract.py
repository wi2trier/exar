"""The readable content of an HTML page, which `WebFetch` windows by line, without the dependencies of the extractor."""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["Extract", "HtmlExtractor"]


@dataclass(slots=True, frozen=True)
class Extract:
    """The readable content of an HTML page, which `WebFetch` windows by line.

    Args:
        title: Title of the page, empty if it has none.
        text: Main content as markdown, links written as `[text](url)`.
        published: Date the page states it was published, if any.
        links: Absolute targets of the links of the page.
    """

    title: str
    text: str
    published: str | None = None
    links: tuple[str, ...] = ()


type HtmlExtractor = Callable[[bytes, str], Extract]
"""Reduces the bytes of an HTML page at a URL to its main content, which is CPU-bound and so runs in a thread."""
