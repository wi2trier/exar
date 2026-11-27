"""Markdown parsed with tree-sitter, its pieces to chunk and its images located by character offsets."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from types import CapsuleType

import tree_sitter_markdown
from tree_sitter import Language, Node, Parser, Query, QueryCursor

from .model import Chunk

__all__ = ["Image", "Piece", "Split", "images", "pieces", "prose"]


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

_VERBATIM = Query(
    _INLINE,
    "[(code_span) (email_autolink) (html_tag) (latex_block) (link_destination) (uri_autolink)] @verbatim",
)
"""The inline nodes that are no prose, such as code, math, HTML, and the targets of links."""

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
class Heading:
    """The heading of a section, its title, the byte offset it starts at, and its line as written."""

    title: str
    start: int
    line: str


@dataclass(frozen=True, slots=True)
class Piece:
    """A span of a document to chunk as one, with the headings of the sections it lies in, and its tokens.

    The tokens of joined blocks are their sum, counting the whitespace between them as one per byte,
    which never falls short of counting their text at once.
    Its own *text*, such as rows of a table under its header, differs from its span.
    A piece *alone*, such as the front matter or the rows of a table, joins no other.
    Its *context* is the lines of the headings above it that it does not start with, outermost first.
    """

    start: int
    end: int
    headings: tuple[Heading, ...]
    tokens: int
    text: str | None = None
    alone: bool = False
    context: tuple[str, ...] = ()

    @property
    def hierarchy(self) -> tuple[str, ...]:
        """The titles of the sections it lies in, outermost first."""
        return tuple(heading.title for heading in self.headings)


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


def _spans(data: bytes, spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Increasing byte *spans* of the UTF-8 *data* as character spans, measuring every slice once."""
    offsets = iter(_measured(data, (o for span in spans for o in span), _decoded))

    return list(zip(offsets, offsets, strict=True))


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
        self,
        start: int,
        end: int,
        headings: tuple[Heading, ...],
        *,
        alone: bool = False,
    ) -> Piece:
        return Piece(
            start,
            end,
            headings,
            self.count(self.data[start:end].decode()),
            alone=alone,
        )

    def node(
        self, node: Node, headings: tuple[Heading, ...], lead: int | None
    ) -> list[Piece]:
        """The pieces of *node*, the first one from *lead*, the start of a heading or marker before it, where they fit."""
        start = node.start_byte if lead is None else lead

        if node.type in _METADATA:
            return [self._piece(start, node.end_byte, headings, alone=True)]

        if node.type == "section" and (title := _title(node)):
            heading = node.children[0]
            line = self.data[heading.start_byte : heading.end_byte].decode().strip()
            headings = (*headings, Heading(title, heading.start_byte, line))

        # No tokenizer yields more tokens than bytes, so a node of at most `size` bytes fits without a descent.
        if node.end_byte - start <= self.size:
            return [self._piece(start, node.end_byte, headings)]

        if node.type in _CONTAINERS:
            return self.children(node, headings, lead)

        piece = self._piece(start, node.end_byte, headings)

        return (
            [piece] if piece.tokens <= self.size else self._leaf(node, headings, lead)
        )

    def children(
        self, node: Node, headings: tuple[Heading, ...], lead: int | None
    ) -> list[Piece]:
        """The pieces of the children of *node*, joined while they fit, a heading or marker kept with the child after it.

        Joined bottom up, a container that fits ends as one piece.
        """
        result: list[Piece] = []

        for child in node.named_children:
            if child.type in _STICKY:
                lead = child.start_byte if lead is None else lead
                continue

            for piece in self.node(child, headings, lead):
                self._append(result, piece)

            lead = None

        if lead is not None:
            self._append(result, self._piece(lead, node.end_byte, headings))

        return result

    def _append(self, result: list[Piece], piece: Piece) -> None:
        if piece.text is None and self.data[piece.start : piece.end].isspace():
            return

        if result and not (result[-1].alone or piece.alone):
            last = result[-1]
            tokens = last.tokens + (piece.start - last.end) + piece.tokens

            if tokens <= self.size:
                # Built directly, since `replace` is slow on this hot path.
                result[-1] = Piece(last.start, piece.end, last.headings, tokens)
                return

        result.append(piece)

    def _leaf(
        self, node: Node, headings: tuple[Heading, ...], lead: int | None
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
                headings,
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
                pieces.insert(0, self._piece(lead, node.start_byte, headings))

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
    spans = _spans(data, ((piece.start, piece.end) for piece in found))

    return [
        replace(
            piece,
            start=start,
            end=end,
            # A heading before the piece is no part of it, one at its start or within is.
            context=tuple(
                heading.line
                for heading in piece.headings
                if heading.start < piece.start
            ),
        )
        for piece, (start, end) in zip(found, spans, strict=True)
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

    spans = _spans(data, ((image.start_byte, image.end_byte) for image, _, _ in found))

    return [
        Image(
            start,
            end,
            _text(alt),
            _text(path).strip().removeprefix("<").removesuffix(">"),
        )
        for (_, alt, path), (start, end) in zip(found, spans, strict=True)
    ]


def _inlines(data: bytes, root: Node) -> list[tuple[Node, Node]]:
    """The blocks of inline text below *root*, a block tree of the UTF-8 *data*, in order, each with its root of the inline grammar."""
    inline = Parser(_INLINE)
    texts = QueryCursor(_TEXTS).captures(root).get("text", [])
    result: list[tuple[Node, Node]] = []

    for text in sorted(texts, key=lambda node: node.start_byte):
        inline.included_ranges = [text.range]
        result.append((text, inline.parse(data).root_node))

    return result


def prose(markdown: str) -> str:
    r"""*markdown* with everything but its prose blanked to NUL character by character, so offsets into either agree.

    Prose is the inline text of the blocks, such as paragraphs, headings, and table cells,
    without code, math, HTML, and the targets of links, so a pattern matched on it never touches those.

    Examples:
        >>> prose("# A\n\nb `c` $d$ [e](f)\n\n```\ng\n```\n").replace("\0", ".")
        '..A..b ... ... [e](.)............'
    """
    data = markdown.encode()
    texts: list[tuple[int, int]] = []

    for _, root in _inlines(data, Parser(_BLOCK).parse(data).root_node):
        position = root.start_byte
        verbatim = QueryCursor(_VERBATIM).captures(root).get("verbatim", [])

        # Sorted by start, a node nested in one before it, such as a code span in HTML, starts before the position.
        for node in sorted(verbatim, key=lambda node: node.start_byte):
            if node.start_byte >= position:
                texts.append((position, node.start_byte))
                position = node.end_byte

        texts.append((position, root.end_byte))

    parts: list[str] = []
    position = 0

    for start, end in _spans(data, texts):
        parts += ["\0" * (start - position), markdown[start:end]]
        position = end

    parts.append("\0" * (len(markdown) - position))

    return "".join(parts)
