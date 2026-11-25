"""The default HTML extractor of `WebFetch`, which needs the `web` extra and so is imported only when a page is extracted."""

import httpx2
from lxml.html import HtmlElement
from markdownify import MarkdownConverter
from trafilatura import bare_extraction, html2txt, load_html
from trafilatura.htmlprocessing import build_html_output
from trafilatura.settings import Document

from .extract import Extract

__all__ = ["extract_html"]

_MARKDOWN = MarkdownConverter(
    heading_style="ATX",
    bullets="-",
    escape_asterisks=False,
    escape_underscores=False,
    escape_misc=False,
    bs4_options="lxml",
)
"""Renders extracted HTML as markdown, leaving the URLs in link text intact."""


def _inline_links(tree: HtmlElement, url: str) -> tuple[str, ...]:
    """Write each link of *tree* as the text ``[text](url)`` and return the targets.

    Every extractor keeps text while some drop link elements, so this is what
    lets the model follow a link of any page.  Targets are made absolute
    against *url*, and links within the page or to other schemes are left out.
    """
    page = httpx2.URL(url).copy_with(fragment=None)
    links: dict[str, None] = {}

    for anchor in list(tree.iter("a")):
        try:
            target = page.join(anchor.get("href", ""))
        except httpx2.InvalidURL:
            continue

        if (
            target.scheme not in ("http", "https")
            or target.copy_with(fragment=None) == page
        ):
            continue

        link = str(target)
        links[link] = None
        text = " ".join("".join(anchor.itertext()).split())
        anchor.clear(keep_tail=True)
        anchor.text = f"[{text or link}]({link})"

    return tuple(links)


def extract_html(body: bytes, url: str, *, fast: bool) -> Extract:
    """Extract the title, markdown, publication date, and links from raw HTML bytes.

    `trafilatura` sniffs the encoding, keeps the main content as HTML, and drops
    navigation, boilerplate, and reader comments, so the model reads the
    article rather than the page around it, which is then rendered as
    markdown with the links `_inline_links` wrote.  A page without main
    content falls back to all of its text.
    *fast* skips the backup extractors that `trafilatura` compares its own result with.
    """
    tree = load_html(body)

    if tree is None:
        return Extract("", "")

    links = _inline_links(tree, url)

    document = bare_extraction(
        tree,
        url=url,
        fast=fast,
        include_comments=False,
        include_formatting=True,
        with_metadata=True,
    )

    if not isinstance(document, Document) or not document.text:
        return Extract("", html2txt(tree).strip(), links=links)

    markdown = _MARKDOWN.convert(build_html_output(document)).strip()

    return Extract(document.title or "", markdown, document.date, links)
