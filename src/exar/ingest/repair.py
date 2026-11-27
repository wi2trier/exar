"""Repairs of the Markdown a parser produced, which edit its prose only, never code, math, HTML, or link targets.

Each repair is a pure function of the Markdown, applied after parsing and before chunking,
so the chunks index the repaired text.
"""

from collections.abc import Iterable

from tree_sitter import Node, Parser, Query, QueryCursor

from ._markdown import _BLOCK, _INLINE, _inlines, _spans, _text, prose

__all__ = ["dehyphenate", "inline_footnotes"]

_SUSPENDED = frozenset({"and", "or", "nor", "und", "oder", "bzw", "sowie"})
"""The conjunctions after a hyphen that suspends a compound, such as `pre- and post-processing`, which stays."""

_CONTINUATION = frozenset(" \t>")
"""What may lie between a line break and the next word of a paragraph, its indentation and the markers of its quotes."""

_LINKS = Query(_INLINE, "(shortcut_link) @link")
"""The shortcut links of inline text, such as the reference `[^1]` to a footnote."""

_DEFINITIONS = Query(_BLOCK, "(link_reference_definition) @definition")
"""The link reference definitions, such as a footnote of one word, `[^1]: Word`."""


def _apply(text: str, edits: Iterable[tuple[int, int, str]]) -> str:
    """*text* with each span of *edits*, in order and disjoint, replaced by its own text."""
    parts: list[str] = []
    position = 0

    for start, end, replacement in edits:
        parts += [text[position:start], replacement]
        position = end

    parts.append(text[position:])

    return "".join(parts)


def dehyphenate(markdown: str, /) -> str:
    r"""Joins the words that a hyphen at the end of a line split, such as those of text extracted from a PDF.

    Only a word continued in lowercase is joined, so compounds such as `X-\nRay` stay,
    as does a hyphen that suspends a compound before a conjunction, such as `pre-\nand post-processing`.

    Examples:
        >>> dehyphenate("A Verbin-\ndung, `co-\nde`, pre-\nand post-\nprocessing.\n\n> quo-\n> ted")
        'A Verbindung, `co-\nde`, pre-\nand postprocessing.\n\n> quoted'
    """
    if "-\n" not in markdown:
        return markdown

    text = prose(markdown)
    edits: list[tuple[int, int, str]] = []
    hyphen = text.find("-\n")

    while hyphen >= 0:
        end = hyphen + 2

        while end < len(text) and text[end] in _CONTINUATION:
            end += 1

        stop = end

        while stop < len(text) and text[stop].isalpha():
            stop += 1

        word = text[end:stop]

        if (
            text[hyphen - 1 : hyphen].isalpha()
            and word[:1].islower()
            and word not in _SUSPENDED
        ):
            edits.append((hyphen, end, ""))

        hyphen = text.find("-\n", end)

    return _apply(markdown, edits)


def _label(node: Node | None) -> str | None:
    """The label of a footnote that *node* references, a shortcut link such as `[^1]`."""
    if node is None or node.type != "shortcut_link":
        return None

    label = _text(node.named_child(0))

    return label[1:] if label.startswith("^") and len(label) > 1 else None


def _words(data: bytes) -> str:
    return " ".join(data.decode().split()).removesuffix(".")


def inline_footnotes(markdown: str, /) -> str:
    r"""Replaces every reference to a footnote with its text in parentheses and drops the definitions.

    A footnote defined by a paragraph or a link reference definition, such as `[^1]: Text`, is inlined, others stay as they are.

    Examples:
        >>> inline_footnotes("A claim[^1]. Another [^x]\n\nPlain\n\n[^1]: The source\n    of it.\n\n[^2]: Unused.\n")
        'A claim (The source of it). Another [^x]\n\nPlain\n\n\n'
    """
    if "[^" not in markdown:
        return markdown

    data = markdown.encode()
    root = Parser(_BLOCK).parse(data).root_node
    inlines = _inlines(data, root)
    definitions: dict[str, str] = {}
    removed: list[tuple[int, int]] = []

    # A paragraph that starts with a reference and a colon, whose words are the footnote.
    for text, inline in inlines:
        if not inline.child_count:
            continue

        first = inline.child(0)
        colon = first.next_sibling if first is not None else None

        if (
            first is not None
            and (label := _label(first)) is not None
            and colon is not None
            and colon.type == ":"
            and colon.start_byte == first.end_byte
            and (block := text.parent) is not None
        ):
            definitions.setdefault(label, _words(data[colon.end_byte : text.end_byte]))
            removed.append((block.start_byte, block.end_byte))

    # A footnote of one word parses as a link reference definition, its word the destination.
    for node in QueryCursor(_DEFINITIONS).captures(root).get("definition", []):
        if (label := _text(node.named_child(0))).startswith("[^") and (
            target := node.named_child(1)
        ) is not None:
            definitions.setdefault(
                label[2:-1], _words(data[target.start_byte : node.end_byte])
            )
            removed.append((node.start_byte, node.end_byte))

    if not definitions:
        return markdown

    # The inline text of a definition paragraph starts where the paragraph does.
    defining = {start for start, _ in removed}

    def inlined(node: Node, words: str) -> tuple[int, int, str]:
        space = "" if data[node.start_byte - 1 : node.start_byte].isspace() else " "

        return node.start_byte, node.end_byte, f"{space}({words})"

    references = [
        inlined(node, words)
        for text, inline in inlines
        if text.start_byte not in defining
        for node in QueryCursor(_LINKS).captures(inline).get("link", [])
        if (words := definitions.get(_label(node) or "")) is not None
    ]
    edits = sorted([*references, *((start, end, "") for start, end in removed)])
    spans = _spans(data, ((start, end) for start, end, _ in edits))

    return _apply(
        markdown,
        (
            (start, end, text)
            for (_, _, text), (start, end) in zip(edits, spans, strict=True)
        ),
    )
