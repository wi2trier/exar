"""Markdown parsed with tree-sitter, its pieces to chunk and its images located by character offsets."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from types import CapsuleType

import tree_sitter_markdown
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from .model import Chunk

__all__ = ["Image", "Piece", "Split", "images", "pieces"]


def _language(capsule: CapsuleType) -> Language:
    # The stubs of tree-sitter-markdown return `CapsuleType` without importing it.
    return Language(capsule)


_BLOCK = _language(tree_sitter_markdown.language())
_INLINE = _language(tree_sitter_markdown.inline_language())

_TEXTS = Query(_BLOCK, "[(inline) (pipe_table_cell)] @text")
"""The inline text of the blocks, which the inline grammar parses."""

_IMAGES = Query(
    _INLINE, "(image (image_description)? @alt (link_destination) @path) @image"
)
"""The inline images, without the references, whose path lies elsewhere."""

_METADATA = frozenset({"minus_metadata", "plus_metadata"})
_CONTAINERS = frozenset({"section", "list", "list_item", "block_quote"})
_STICKY = frozenset(
    {
        "atx_heading",
        "setext_heading",
        "block_continuation",
        "block_quote_marker",
        "list_marker_dot",
        "list_marker_minus",
        "list_marker_parenthesis",
        "list_marker_plus",
        "list_marker_star",
    }
)
"""The nodes that belong to the block after them, such as a heading or the marker of a list item."""

_LEAD_RESERVE = 8
"""One in this many tokens of a split block is left for the heading or marker before it.

A fixed share rather than its exact length, so the splitter sees two sizes and caches two chunkers.
"""

type Split = Callable[[str, int, bool], Sequence[Chunk]]
"""Splits a block into parts of at most a number of tokens, or a table into rows under its header."""


@dataclass(frozen=True, slots=True)
class Piece:
    """A span of a document to chunk as one, with the headings above it as its hierarchy, and its tokens.

    The tokens of joined blocks are their sum, counting the whitespace between them as one per byte,
    which never falls short of counting their text at once.
    Its own *text*, such as rows of a table under its header, differs from its span.
    A piece *alone*, such as the front matter or the rows of a table, joins no other.
    """

    start: int
    end: int
    hierarchy: tuple[str, ...]
    tokens: int
    text: str | None = None
    alone: bool = False


@dataclass(frozen=True, slots=True)
class Image:
    """An inline image of a document, its character span, alternative text, and path."""

    start: int
    end: int
    alt: str
    path: str


def _measured[T: (str, bytes)](
    source: T, offsets: Iterable[int], length: Callable[[T], int]
) -> list[int]:
    """Increasing *offsets* into *source* as the summed *length* of what lies before each, measuring every slice once."""
    result: list[int] = []
    position = total = 0

    for offset in offsets:
        total += length(source[position:offset])
        position = offset
        result.append(total)

    return result


def _encoded(text: str) -> int:
    return len(text.encode())


def _decoded(data: bytes) -> int:
    return len(data.decode())


def _text(node: Node | None) -> str:
    return "" if node is None or node.text is None else node.text.decode()


def _title(section: Node) -> str:
    """The title of a section, empty for the text before the first heading."""
    heading = section.named_child(0) if section.named_child_count else None

    if (
        heading is None
        or heading.type != "atx_heading"
        or heading.named_child_count < 2
    ):
        return ""

    # Its marker comes first, then its text, which an empty heading lacks.
    return " ".join(_text(heading.named_child(1)).split())


@dataclass(frozen=True, slots=True)
class _Walk:
    """Splits the nodes of a document, its UTF-8 *data*, into pieces by byte offsets, counting each block once."""

    data: bytes
    size: int
    count: Callable[[str], int]
    split: Split

    def _piece(
        self, start: int, end: int, hierarchy: tuple[str, ...], *, alone: bool = False
    ) -> Piece:
        return Piece(
            start, end, hierarchy, self.count(self.data[start:end].decode()), alone=alone
        )

    def node(
        self, node: Node, hierarchy: tuple[str, ...], lead: int | None
    ) -> list[Piece]:
        """The pieces of *node*, the first one from *lead*, the start of a heading or marker before it, where they fit."""
        start = node.start_byte if lead is None else lead

        if node.type in _METADATA:
            return [self._piece(start, node.end_byte, hierarchy, alone=True)]

        if node.type == "section" and (title := _title(node)):
            hierarchy = (*hierarchy, title)

        # No tokenizer yields more tokens than bytes, so a node of at most `size` bytes fits without a descent.
        if node.end_byte - start <= self.size:
            return [self._piece(start, node.end_byte, hierarchy)]

        if node.type in _CONTAINERS:
            return self.children(node, hierarchy, lead)

        piece = self._piece(start, node.end_byte, hierarchy)

        return (
            [piece] if piece.tokens <= self.size else self._leaf(node, hierarchy, lead)
        )

    def children(
        self, node: Node, hierarchy: tuple[str, ...], lead: int | None
    ) -> list[Piece]:
        """The pieces of the children of *node*, joined while they fit, a heading or marker kept with the child after it.

        Joined bottom up, a container that fits ends as one piece.
        """
        result: list[Piece] = []

        for child in node.named_children:
            if child.type in _STICKY:
                lead = child.start_byte if lead is None else lead
                continue

            for piece in self.node(child, hierarchy, lead):
                self._append(result, piece)

            lead = None

        if lead is not None:
            self._append(result, self._piece(lead, node.end_byte, hierarchy))

        return result

    def _append(self, result: list[Piece], piece: Piece) -> None:
        if piece.text is None and self.data[piece.start : piece.end].isspace():
            return

        if result and not (result[-1].alone or piece.alone):
            last = result[-1]
            tokens = last.tokens + (piece.start - last.end) + piece.tokens

            if tokens <= self.size:
                # Built directly, since `replace` is slow on this hot path.
                result[-1] = Piece(last.start, piece.end, last.hierarchy, tokens)
                return

        result.append(piece)

    def _leaf(
        self, node: Node, hierarchy: tuple[str, ...], lead: int | None
    ) -> list[Piece]:
        """The parts of a block too long for one piece, the first one from *lead* where it still fits."""
        table = node.type == "pipe_table"
        block = _text(node)
        limit = self.size if lead is None else self.size - self.size // _LEAD_RESERVE
        parts = self.split(block, limit, table)
        bounds = iter(
            _measured(
                block, (o for part in parts for o in (part.start, part.end)), _encoded
            )
        )
        pieces = [
            Piece(
                node.start_byte + start,
                node.start_byte + end,
                hierarchy,
                part.tokens,
                part.text if table else None,
                table,
            )
            for part, start, end in zip(parts, bounds, bounds, strict=True)
        ]

        if lead is not None:
            first = pieces[0]
            prefix = self.data[lead : node.start_byte].decode()
            tokens = self.count(prefix) + first.tokens

            if tokens <= self.size:
                text = None if first.text is None else prefix + first.text
                pieces[0] = replace(first, start=lead, tokens=tokens, text=text)
            else:
                pieces.insert(0, self._piece(lead, node.start_byte, hierarchy))

        return pieces


def pieces(
    markdown: str, *, size: int, count: Callable[[str], int], split: Split
) -> list[Piece]:
    r"""Splits *markdown* into the pieces to chunk, in order, each of at most *size* tokens by *count* but the front matter.

    Every block is counted once.
    A section that fits is one piece, else its blocks are, and lists and quotes split at their items and blocks the same way,
    joined while they fit, with a heading or marker kept with the block after it.
    The document itself is never one piece, so its front matter stands alone.
    A block too long for a piece is split with *split*, a table into rows under its header.

    Examples:
        >>> text = "---\na: 1\n---\n\n# A\n\nIntro.\n\n## B\n\n- one\n- two\n"
        >>> whole = lambda block, size, table: [Chunk(text=block, start=0, end=len(block), tokens=len(block))]
        >>> [(text[p.start : p.end], p.hierarchy) for p in pieces(text, size=99, count=len, split=whole)]
        [('---\na: 1\n---\n', ()), ('# A\n\nIntro.\n\n## B\n\n- one\n- two\n', ('A',))]
        >>> [(text[p.start : p.end], p.hierarchy) for p in pieces(text, size=20, count=len, split=whole)]
        [('---\na: 1\n---\n', ()), ('# A\n\nIntro.\n', ('A',)), ('## B\n\n- one\n- two\n', ('A', 'B'))]
    """
    data = markdown.encode()
    found = _Walk(data, size, count, split).children(
        Parser(_BLOCK).parse(data).root_node, (), None
    )
    offsets = iter(
        _measured(
            data, (o for piece in found for o in (piece.start, piece.end)), _decoded
        )
    )

    return [
        replace(piece, start=start, end=end)
        for piece, start, end in zip(found, offsets, offsets, strict=True)
    ]


def images(markdown: str) -> list[Image]:
    """The inline images of *markdown*, in order, but none that code shows or a reference names.

    Examples:
        >>> [image.path for image in images("![a](a.png) `![b](b.png)` | ![c](<c d.png>) |")]
        ['a.png', 'c d.png']
    """
    data = markdown.encode()

    # Every image starts with `![`, so text without one is never parsed for images.
    if b"![" not in data:
        return []

    inline = Parser(_INLINE)
    texts = (
        QueryCursor(_TEXTS)
        .captures(Parser(_BLOCK).parse(data).root_node)
        .get("text", [])
    )
    found: list[tuple[Node, Node | None, Node]] = []

    for text in sorted(texts, key=lambda node: node.start_byte):
        if data.find(b"![", text.start_byte, text.end_byte) < 0:
            continue

        inline.included_ranges = [text.range]

        for _, match in QueryCursor(_IMAGES).matches(inline.parse(data).root_node):
            alt = match.get("alt")
            found.append((match["image"][0], alt[0] if alt else None, match["path"][0]))

    offsets = iter(
        _measured(
            data,
            (o for image, _, _ in found for o in (image.start_byte, image.end_byte)),
            _decoded,
        )
    )

    return [
        Image(
            start,
            end,
            _text(alt),
            _text(path).strip().removeprefix("<").removesuffix(">"),
        )
        for (_, alt, path), start, end in zip(found, offsets, offsets, strict=True)
    ]
