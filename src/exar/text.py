"""Text helpers for language models: decoding bytes, framing data in tags, and fitting text into a budget.

`decode` turns bytes into text, `as_text` renders any value as text, `TEXT` validates an output as text,
multimodal content such as a prompt with images as text with placeholders for its media (`genai_parts`),
`tagged` frames untrusted data in a tag it cannot break out of, and `fenced` in a Markdown code fence.
`render_example` frames a run as a few-shot example, and `replace_once` edits a text unambiguously.
The truncation helpers keep tool output within the budget of a context window,
and the line helpers number, clip, and window the lines of a tool output and name what they left out.
"""

import codecs
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import singledispatch
from itertools import islice
from typing import Any, Final
from urllib.parse import urlsplit
from xml.sax.saxutils import quoteattr

from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    MessagePart,
    TextPart,
    UriPart,
)
from pydantic import TypeAdapter
from pydantic_core import to_json

from .typing import Verdict

__all__ = [
    "MAX_BYTES_PER_CHAR",
    "NOT_TEXT_REASON",
    "TEXT",
    "Decoded",
    "LineClip",
    "LineWindow",
    "as_text",
    "decode",
    "fenced",
    "first_line",
    "fit_window",
    "format_bytes",
    "genai_parts",
    "hint_suffix",
    "leading_text",
    "limit_hints",
    "markdown_cell",
    "markdown_table",
    "media_parts",
    "nfc",
    "number_lines",
    "os_reason",
    "plural",
    "render_example",
    "replace_once",
    "tagged",
    "truncate_around",
    "truncate_line",
    "truncate_middle",
    "window_start",
]

NOT_TEXT_REASON = "is not text-like content"
"""Wording for a file that `decode` refuses, which a caller wraps in its own message."""

TEXT: Final[TypeAdapter[Any]] = TypeAdapter(str)
"""Validates an output as text, the default output of remote agents, which they tell from any other by identity."""

MAX_BYTES_PER_CHAR = 4
"""Widest encoding of one character among those `decode` accepts, UTF-32, which bounds the characters of a file by its size."""

# C0 and C1 control characters except the whitespace and escape characters of text.
_BINARY_CONTROL = re.compile(r"[\x00-\x06\x0e-\x1a\x1c-\x1f\x7f-\x9f]")
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Decoded:
    """Text decoded from bytes, with the encoding it had if that was not UTF-8."""

    text: str
    encoding: str | None = None


def decode(content: bytes, /) -> Decoded | None:
    """Decodes bytes to text, or returns None if they are not text.

    A byte order mark decides the Unicode encoding, then strict UTF-8 is tried,
    then CP1252, the common legacy encoding of Western Windows exports.
    Text with binary control characters is refused.

    Examples:
        >>> decode(b'{"a": 1}').text
        '{"a": 1}'
        >>> decode("Grüße".encode("utf-16")).encoding
        'utf-16'
        >>> decode(b"\\x89PNG\\r\\n\\x1a\\n") is None
        True
    """
    encoding = next((name for bom, name in _BOMS if content.startswith(bom)), None)

    try:
        text = content.decode(encoding or "utf-8")
    except UnicodeDecodeError:
        if encoding is not None:
            return None

        encoding = "cp1252"

        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            return None

    if _BINARY_CONTROL.search(text) is not None:
        return None

    return Decoded(text=text, encoding=encoding)


def fenced(text: str, language: str = "", /) -> str:
    """Wraps text in a Markdown code fence longer than any backtick run inside it.

    Examples:
        >>> fenced("print(1)", "py")
        '```py\\nprint(1)\\n```\\n'
        >>> fenced("a ```` b")
        '`````\\na ```` b\\n`````\\n'
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)

    return f"{fence}{language}\n{text}\n{fence}\n"


@singledispatch
def genai_parts(value: object, /) -> list[MessagePart] | None:
    """The GenAI message parts of content, such as a multimodal prompt, None if the value is not content.

    Content is a string or a GenAI part of text, binary data, a URI, or a file,
    or a sequence of them, whose parts are flattened in order.
    Frameworks register their content types, as `exar.pydantic_ai` does for the user content of pydantic-ai.

    Examples:
        >>> genai_parts(["Describe", FilePart(mime_type="image/png", modality="image", file_id="f1")])
        [TextPart(content='Describe', type='text'), FilePart(mime_type='image/png', modality='image', file_id='f1', type='file')]
        >>> genai_parts({"answer": 42}) is None
        True
    """
    return None


@genai_parts.register
def _(value: str, /) -> list[MessagePart]:
    return [TextPart(value)]


@genai_parts.register
def _(value: TextPart | BlobPart | UriPart | FilePart, /) -> list[MessagePart]:
    return [value]


@genai_parts.register(Sequence)
def _(value: Sequence[object], /) -> list[MessagePart] | None:
    parts: list[MessagePart] = []

    for item in value:
        if (item_parts := genai_parts(item)) is None:
            return None

        parts.extend(item_parts)

    return parts


def media_parts(value: object, /) -> list[MessagePart] | None:
    """The parts of multimodal content (`genai_parts`), `None` unless some part is not text.

    Examples:
        >>> media_parts("Paris") is None, media_parts([TextPart("a"), TextPart("b")]) is None
        (True, True)
        >>> media_parts([BlobPart(mime_type="image/png", modality="image", content=b"")]) is None
        False
    """
    parts = genai_parts(value)

    return (
        None
        if parts is None or all(isinstance(part, TextPart) for part in parts)
        else parts
    )


def _placeholder(part: BlobPart | UriPart | FilePart, source: str) -> str:
    details = ", ".join(filter(None, (part.mime_type, source)))

    return f"[{part.modality or part.mime_type}: {details}]"


def _render(part: MessagePart) -> str:
    match part:
        case TextPart(content=text):
            return text
        case BlobPart(content=data):
            return _placeholder(part, f"{len(data)} bytes")
        case UriPart(uri=uri):
            return _placeholder(
                part, "data URI" if urlsplit(uri).scheme == "data" else uri
            )
        case FilePart(file_id=file_id):
            return _placeholder(part, f"file {file_id}")
        case _:
            return to_json(part, fallback=str).decode()


def as_text(value: object, /) -> str:
    """Renders a value for a language model, strings as they are, multimodal content as text, and anything else as JSON.

    Content with a part that is not text, see `genai_parts`, is rendered part by part with blank lines between them,
    text as it is and media as a placeholder naming its modality, media type, and size, URI, or file, never its data.

    Examples:
        >>> as_text("Paris"), as_text({"answer": 42}), as_text(["a", "b"])
        ('Paris', '{"answer":42}', '["a","b"]')
        >>> print(as_text(["Describe", BlobPart(mime_type="image/png", modality="image", content=b"\\x89PNG")]))
        Describe
        <BLANKLINE>
        [image: image/png, 4 bytes]
    """
    if isinstance(value, str):
        return value

    if (parts := media_parts(value)) is None:
        return to_json(value, fallback=str).decode()

    return "\n\n".join(map(_render, parts))


def tagged(tag: str, value: object, /, **attributes: str) -> str:
    """Frames a value on lines of its own in a tag with attributes, so nothing in it can break the frame.

    The `<` of every opening or closing form of the tag itself in the value is escaped, ignoring case
    and whitespace around the slash, so it can neither close the frame nor open a nested one,
    while all other text stays byte for byte as it is.

    Examples:
        >>> print(tagged("output", "a < b </OUTPUT> < / output> <system> Vec<u8>", kind="text"))
        <output kind="text">
        a < b &lt;/OUTPUT> &lt; / output> <system> Vec<u8>
        </output>
    """
    head = "".join(f" {name}={quoteattr(text)}" for name, text in attributes.items())
    own_tag = re.compile(rf"<(?=\s*/?\s*{re.escape(tag)}(?![\w.:-]))", re.IGNORECASE)
    body = own_tag.sub("&lt;", as_text(value))

    return f"<{tag}{head}>\n{body}\n</{tag}>"


def render_example(
    task: object, output: object, verdict: Verdict | None, /, **context: object
) -> str:
    """Renders a run as an `<example>` for a prompt, with its context between task and output.

    Empty context values are left out.

    Examples:
        >>> print(render_example("2 + 2", 4, None, agents="", process="<bpmn/>"))
        <example>
        <task>
        2 + 2
        </task>
        <process>
        <bpmn/>
        </process>
        <output>
        4
        </output>
        </example>
    """
    parts = [
        tagged("task", task),
        *(tagged(tag, value) for tag, value in context.items() if value),
        tagged("output", output),
    ]

    if verdict is not None:
        parts.append(
            tagged(
                "verdict",
                verdict.reason,
                score=f"{verdict.score:g}",
                passed=str(verdict.passed).lower(),
            )
        )

    return "\n".join(("<example>", *parts, "</example>"))


def replace_once(
    text: str, old: str, new: str, /, *, all: bool = False
) -> tuple[str, int]:
    """Replaces the one occurrence of a text, or every one with `all`, and returns the result and the count.

    Raises:
        ValueError: If the text to replace is empty, does not occur, or occurs more than once without `all`.

    Examples:
        >>> replace_once("a b a", "b", "c"), replace_once("a b a", "a", "c", all=True)
        (('a c a', 1), ('c b c', 2))
        >>> replace_once("a b a", "a", "c")
        Traceback (most recent call last):
        ValueError: The text occurs 2 times, it must occur once
    """
    if not old:
        raise ValueError("The text to replace is empty")

    count = text.count(old)

    if count == 0 or (count > 1 and not all):
        raise ValueError(
            f"The text occurs {count} times, it must occur {'at least once' if all else 'once'}"
        )

    return text.replace(old, new), count


def truncate_line(text: str, max_chars: int | None = None, /) -> str:
    """Clips text to `max_chars`, marking a cut with an ellipsis.

    Examples:
        >>> truncate_line("hello world", 8), truncate_line("short", 100), truncate_line("whole")
        ('hello w…', 'short', 'whole')
    """
    if max_chars is None or len(text) <= max_chars:
        return text

    return text[: max_chars - 1] + "…" if max_chars > 0 else ""


def first_line(text: str, /) -> str:
    """The first line of a text, empty for an empty one, such as the summary of a description.

    Examples:
        >>> first_line("Lists rows.\\nMore."), first_line("")
        ('Lists rows.', '')
    """
    return next(iter(text.splitlines()), "")


def os_reason(error: OSError, /) -> str:
    """Why an operation of the OS failed, without the paths it names, so no host path reaches a model.

    Examples:
        >>> os_reason(FileNotFoundError(2, "No such file or directory", "/srv/a")), os_reason(OSError())
        ('No such file or directory', 'OSError')
    """
    return error.strerror or type(error).__name__


def nfc(value: str, /) -> str:
    r"""Normalizes text to the NFC form, so two spellings of one name compare equal.

    Examples:
        >>> nfc("e\u0301") == "\u00e9"
        True
    """
    return unicodedata.normalize("NFC", value)


def markdown_cell(value: object, /) -> str:
    r"""Renders a table cell on one line with its pipes escaped, None as empty.

    Examples:
        >>> markdown_cell("a|b\nc"), markdown_cell(None)
        ('a\\|b c', '')
    """
    if value is None:
        return ""

    return str(value).replace("|", r"\|").replace("\n", " ")


def markdown_table(
    columns: Iterable[str], rows: Iterable[Iterable[str]], /, *, compact: bool = False
) -> str:
    r"""Joins the header, its separator, and the rendered rows of a Markdown table.

    Examples:
        >>> markdown_table(["a", "b"], [["1", "2"]])
        '| a | b |\n| --- | --- |\n| 1 | 2 |'
        >>> markdown_table(["a"], [["1"]], compact=True)
        '| a |\n|---|\n| 1 |'
    """
    columns = list(columns)
    separator = (
        f"|{'---|' * len(columns)}"
        if compact
        else f"| {' | '.join('---' for _ in columns)} |"
    )

    return "\n".join(
        [
            f"| {' | '.join(columns)} |",
            separator,
            *(f"| {' | '.join(row)} |" for row in rows),
        ]
    )


def window_start(text: str, start: int, max_chars: int, /) -> int:
    """The offset at which `truncate_around` opens its window of `max_chars` onto text around `start`.

    It lets a caller point into the clipped text, such as with a caret under the character at `start`.

    Examples:
        >>> window_start("a" * 100, 50, 20), window_start("a" * 100, 99, 20), window_start("abc", 2, 20)
        (45, 80, 0)
    """
    if len(text) <= max_chars:
        return 0

    return max(min(start - max_chars // 4, len(text) - max_chars), 0)


def truncate_around(text: str, start: int, max_chars: int, /) -> str:
    """Clips text to `max_chars` around offset `start`, such as a search hit far into a long line.

    The window opens a quarter of the budget before `start` so the lead-in reads too,
    and each cut side is marked with an ellipsis.

    Examples:
        >>> truncate_around("a" * 20 + "hit" + "b" * 20, 20, 12)
        '…aahitbbbbb…'
        >>> truncate_around("hit and more", 0, 8)
        'hit and…'
    """
    begin = window_start(text, start, max_chars)

    if not begin:
        return truncate_line(text, max_chars)

    # The leading ellipsis replaces the first kept character,
    # one character past the window tells whether the tail is cut.
    return "…" + truncate_line(text[begin + 1 : begin + max_chars + 1], max_chars - 1)


def truncate_middle(
    text: str, max_chars: int, /, note: Callable[[str, int], str] | None = None
) -> str:
    """Keeps the head and tail of text within `max_chars`, around a note on the characters left out.

    The head gets three quarters of the room, both parts are cut at line ends where possible,
    and the tail keeps the hints that tools append to their output.
    The note is never cut, if it does not fit it is all that is returned, clipped.

    Args:
        text: Text to shorten.
        max_chars: Budget of the result.
        note: Renders the note from the kept head and the number of characters left out,
            such as the line a read of the whole text resumes at.
            It is sized for the whole text as head and its length as count, which it must not outgrow.

    Examples:
        >>> print(truncate_middle("\\n".join(map(str, range(100))), 60))
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        [265 characters left out here]
        98
        99
    """
    if len(text) <= max_chars:
        return text

    render: Callable[[str, int], str] = note or (
        lambda _head, left_out: f"[{left_out} characters left out here]"
    )
    # Sized for the widest note, two line breaks join the note to head and tail.
    widest = render(text, len(text))
    room = max_chars - len(widest) - 2

    if room < 0:
        return truncate_line(widest, max_chars)

    head_room = room * 3 // 4
    tail_start = len(text) - (room - head_room)
    end = text.rfind("\n", 0, head_room + 1)
    begin = text.find("\n", tail_start - 1)
    head = text[:end] if end > 0 else text[:head_room]
    tail = text[begin + 1 :] if begin >= 0 else text[tail_start:]

    return "\n".join(
        filter(None, (head, render(head, len(text) - len(head) - len(tail)), tail))
    )


def hint_suffix(hints: Iterable[str], /) -> str:
    """Renders hints on what a tool left out as the bracketed note that trails its output.

    Examples:
        >>> hint_suffix(["3 more lines", "pass offset=4"])
        '\\n\\n[3 more lines; pass offset=4]'
        >>> hint_suffix([])
        ''
    """
    text = "; ".join(hints)

    return f"\n\n[{text}]" if text else ""


def plural(count: int, noun: str, many: str | None = None, /) -> str:
    """The `noun` in the form that matches `count`, which is `many` or the noun with an `s` when it is not one.

    Examples:
        >>> plural(1, "row"), plural(2, "row"), plural(2, "entry", "entries")
        ('row', 'rows', 'entries')
    """
    return noun if count == 1 else many or f"{noun}s"


def format_bytes(size: int, /) -> str:
    """Renders a byte count in the largest unit that keeps it at or above one, with one decimal place from KB up.

    Examples:
        >>> format_bytes(512), format_bytes(1536), format_bytes(52428800)
        ('512 B', '1.5 KB', '50.0 MB')
    """
    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"

    return f"{size / (1024 * 1024):.1f} MB"


def limit_hints(limit: int, total: int, remedy: str, /) -> list[str]:
    """The hint that a result shows only `limit` of its `total` items, with what reveals the rest.

    Examples:
        >>> limit_hints(5, 9, "raise max_results to see more"), limit_hints(5, 5, "")
        (['showing 5 of 9, raise max_results to see more'], [])
    """
    return [f"showing {limit} of {total}, {remedy}"] if total > limit else []


def leading_text(value: object, /) -> tuple[str, list[object] | None] | None:
    """The text a tool return value leads with and the items after it, `None` without leading text.

    The text is a `str` return, which has no items after it,
    or the leading `str` of a content list such as `[text, BinaryContent]`.

    Examples:
        >>> leading_text("a"), leading_text(["a", 1]), leading_text([1])
        (('a', None), ('a', [1]), None)
    """
    match value:
        case str():
            return value, None
        case [str() as text, *rest] if isinstance(value, list):
            return text, rest
        case _:
            return None


@dataclass(slots=True)
class LineClip:
    """A budget per line that clips each line it is handed and remembers whether it cut one.

    A clipped line is the one cut a reader cannot tell from the output alone, so `hints` names it,
    with the `remedy` that shows the lines whole.
    A `None` budget keeps every line whole.

    Examples:
        >>> clip = LineClip(4, remedy="read the file to see them whole")
        >>> clip("ab"), clip("abcdef"), clip.hints()
        ('ab', 'abc…', ['lines clipped at 4 chars, read the file to see them whole'])
        >>> LineClip(None)("abcdef")
        'abcdef'
    """

    max_chars: int | None
    noun: str = "lines"
    remedy: str = ""
    clipped: bool = field(default=False, init=False)

    def __call__(self, text: str, /) -> str:
        """Clips text from its start."""
        return self.around(text, 0)

    def around(self, text: str, start: int, /) -> str:
        """Clips text around offset `start`, as `truncate_around` does."""
        if self.max_chars is None or len(text) <= self.max_chars:
            return text

        self.clipped = True

        return truncate_around(text, start, self.max_chars)

    def hints(self) -> list[str]:
        """The hint naming the cut, or none if no line was clipped."""
        if not self.clipped:
            return []

        remedy = f", {self.remedy}" if self.remedy else ""

        return [f"{self.noun} clipped at {self.max_chars} chars{remedy}"]


def _numbered(lines: Iterable[str], start: int, clip: LineClip | None) -> Iterator[str]:
    """Each line numbered from `start`, clipped by `clip`."""
    for number, line in enumerate(lines, start):
        yield f"{number}: {line if clip is None else clip(line)}"


def number_lines(
    lines: Iterable[str], /, start: int = 1, clip: LineClip | None = None
) -> str:
    """Numbers lines from `start`, each clipped by `clip`, so a model can cite the exact line of a sentence.

    Examples:
        >>> number_lines(["a", "b"], start=4)
        '4: a\\n5: b'
        >>> number_lines(["abcdef"], clip=LineClip(3))
        '1: ab…'
    """
    return "\n".join(_numbered(lines, start, clip))


@dataclass(frozen=True, slots=True)
class LineWindow:
    """The run of lines of a text that one call of a tool shows."""

    start: int
    end: int
    total: int
    content: str
    """The lines shown, unclipped, such as for a client."""
    rendered: str
    """The lines shown, numbered and clipped, for the model."""

    def past_end(self, noun: str, /) -> str | None:
        """The message standing in for a window that starts past the end of the text, which is a `noun`."""
        if self.start <= self.total:
            return None

        return f"(offset {self.start} is past the end of the {noun} with {self.total} {plural(self.total, 'line')})"

    def hints(self) -> list[str]:
        """The hint naming the `offset` to read on at, or none at the end of the text."""
        remaining = self.total - self.end

        if remaining <= 0:
            return []

        return [
            f"{remaining} more {plural(remaining, 'line')}, call again with offset={self.end + 1}"
        ]


def fit_window(
    lines: Sequence[str],
    start: int,
    max_chars: int,
    clip: LineClip,
    /,
    limit: int | None = None,
) -> LineWindow:
    """The lines from the 1-based `start` on, at most `limit`, that fit `max_chars` as rendered through `clip`.

    The lines are measured as rendered, numbered and clipped,
    since the numbers of many short lines would otherwise push the window past the budget.
    They are a contiguous prefix, so a caller resumes from exactly where the output stops.
    The first line counts however long it is, since a lone oversized line tells more than an empty result.

    Examples:
        >>> window = fit_window(["ab", "cd", "ef"], 2, 8, LineClip(None))
        >>> window.rendered, window.hints()
        ('2: cd', ['1 more line, call again with offset=3'])
    """
    stop = len(lines) if limit is None else min(len(lines), start + limit - 1)
    # A throwaway clip, since it also sees the line that does not fit.
    measure = LineClip(clip.max_chars)
    kept = 0
    # The first line takes no newline before it.
    size = -1

    for line in _numbered(islice(lines, start - 1, stop), start, measure):
        size += len(line) + 1

        if kept and size > max_chars:
            break

        kept += 1

    shown = lines[start - 1 : start - 1 + kept]

    return LineWindow(
        start=start,
        end=start + kept - 1,
        total=len(lines),
        content="\n".join(shown),
        rendered="\n".join(_numbered(shown, start, clip)),
    )
