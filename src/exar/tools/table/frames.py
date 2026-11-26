"""Loads files as typed frames, which types what a file stores as text and guesses the header rows of a sheet."""

import re
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from io import BytesIO
from itertools import chain
from pathlib import PurePosixPath

import polars as pl

from ...memory.mounts import Mount, entry_stat
from .. import Retry
from ..common import check_read_budget, read_bytes_or_retry, read_text_or_retry
from ..formats import DELIMITERS
from .result import HeaderGuess, QueriedTable, TextColumn

__all__ = ["Source", "open_source", "polars_message"]

_LEADING_ZERO = r"^[+-]?0\d"
"""Zero-padded digits are an identifier (a zip code, an EAN), not a number."""

_DATETIME = r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d"
"""The shape of a timestamp, which keeps looser text from parsing as one."""

_OFFSET = r"(?:[Zz]|[+-]\d{2}(?::?\d{2})?)$"
"""The end of a timestamp that names its UTC offset."""

_INTEGER = r"^-?\d+$"
_DECIMAL = r"^-?\d*\.?\d+$"
"""Plain digits only, since a sign, an exponent, `inf`, or `nan` marks an identifier or a code."""

_EXACT_DIGITS = 15
"""Digits a float holds exactly, past which a value would reach a query rounded."""

_UNPARSED_SAMPLE = 3
"""Distinct unparsable values named per column, enough to say what they are."""

_UNNAMED = re.compile(r"(?:_duplicated_|__UNNAMED__)\d+|")
"""How Polars names a header cell that is empty, in a spreadsheet or a CSV."""

_CACHED_SOURCES = 16
"""Loaded files kept for the next call, which asks about the same few files again and again."""

_HEADER_SCAN = 10
"""Leading rows a header guess may span, past which they are data gone odd."""


def polars_message(exc: pl.exceptions.PolarsError) -> str:
    """Polars' message as a sentence, without the plan dump nothing acts on.

    >>> polars_message(pl.exceptions.SQLInterfaceError(
    ...     "unsupported function 'instr'\\n\\nResolved plan until failure:\\n\\nDF"
    ... ))
    "unsupported function 'instr'."
    """
    text = str(exc).partition("Resolved plan until failure:")[0].rstrip()

    return text if text.endswith((".", "?", "!")) else f"{text}."


@dataclass(slots=True, frozen=True)
class _Coercion:
    """A dtype a text column takes on when every one of its values is one."""

    dtype: pl.DataType
    sql: str
    """The name of the dtype as DuckDB, which runs the queries, spells it."""

    convert: Callable[[pl.Expr], pl.Expr]
    numeric: bool = False
    """Whether one zero-padded value rules this dtype out for the column."""

    def matches(self, text: pl.Expr) -> pl.Expr:
        """Whether a value becomes this dtype, tested without ever raising."""
        return self.convert(text).is_not_null()


def _numeric(dtype: pl.DataType, sql: str, shape: str) -> _Coercion:
    """Build the coercion to *dtype*, which only plain digits of *shape* take."""

    def convert(text: pl.Expr) -> pl.Expr:
        exact = text.str.count_matches(r"\d") <= _EXACT_DIGITS

        return pl.when(
            text.str.contains(shape) & (exact if dtype.is_float() else True)
        ).then(text.cast(dtype, strict=False))

    return _Coercion(dtype, sql, convert, numeric=True)


def _datetime(*, aware: bool) -> _Coercion:
    """Build the coercion to a timestamp, in UTC when the values name an offset and naive when none does."""

    def convert(text: pl.Expr) -> pl.Expr:
        named = text.str.contains(_OFFSET)
        parsed = text.str.to_datetime(time_zone="UTC", strict=False)

        return pl.when(
            text.str.contains(_DATETIME) & (named if aware else ~named)
        ).then(parsed if aware else parsed.dt.replace_time_zone(None))

    return _Coercion(
        pl.Datetime(time_zone="UTC" if aware else None),
        "TIMESTAMP WITH TIME ZONE" if aware else "TIMESTAMP",
        convert,
    )


def _boolean(text: pl.Expr) -> pl.Expr:
    """Parse the two boolean literals Polars infers from delimited files."""
    return text.str.to_lowercase().replace_strict(
        {"true": True, "false": False}, default=None, return_dtype=pl.Boolean
    )


_COERCIONS = (
    # Integers wider than an Int64 stay text, an identifier rather than a number,
    # since the float that would hold them rounds them.
    _numeric(pl.Int64(), "BIGINT", _INTEGER),
    _numeric(pl.Float64(), "DOUBLE", _DECIMAL),
    # An explicit format rejects an impossible day that a shape test would
    # wave through, and still reads a single-digit month.
    _Coercion(
        pl.Date(), "DATE", lambda text: text.str.to_date("%Y-%m-%d", strict=False)
    ),
    _datetime(aware=False),
    _datetime(aware=True),
    _Coercion(pl.Boolean(), "BOOLEAN", _boolean),
)
"""Tried in order, so the narrowest dtype that fits every value wins."""


def _text(name: str) -> pl.Expr:
    """Address a text column with its surrounding whitespace dropped."""
    return pl.col(name).str.strip_chars()


def _filled(text: pl.Expr) -> pl.Expr:
    """Whether a text cell holds a value, as an empty one reads as missing."""
    return text.is_not_null() & text.ne("")


def _label(cell: str | None) -> str | None:
    """A cell as a column name, ``None`` for a blank one.

    >>> _label("  Menge "), _label(" "), _label(None)
    ('Menge', None, None)
    """
    return (cell or "").strip() or None


@dataclass(slots=True, frozen=True)
class Source:
    """A loaded table, with what the loading discovered about it."""

    frame: pl.LazyFrame
    file_path: str = ""
    sheet: str | None = None
    sheets: tuple[str, ...] = ()
    encoding: str | None = None
    text_columns: tuple[TextColumn, ...] = ()
    """Text columns whose values parse as another type, a mixed one still text and one a query has to cast."""

    report_complete: bool = True
    """Whether losslessly converted text columns should be reported."""

    skip_above_header: int = 0
    skip_below_header: int = 0
    """The header rows the frame was loaded with, which a guess counts from."""

    header_guess: HeaderGuess | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column name, as the loaded frame declares them."""
        return tuple(self.frame.collect_schema().names())

    def queried(self, name: str) -> QueriedTable:
        """Describe this table as the result reports it, under its SQL name."""
        return QueriedTable(
            name=name,
            file_path=self.file_path,
            sheet=self.sheet,
            sheets=self.sheets,
            encoding=self.encoding,
            text_columns=self.text_columns,
            header_guess=self.header_guess,
        )


def _column_counts(name: str) -> Iterator[pl.Expr]:
    """The value counts one text column is judged by, in a fixed order."""
    text = _text(name)
    filled = _filled(text)
    yield filled.sum()
    yield (filled & text.str.contains(_LEADING_ZERO)).sum()

    for coercion in _COERCIONS:
        yield (filled & coercion.matches(text)).sum()


def _lead(name: str, coercion: _Coercion) -> pl.Expr:
    """Rows above a column's first parsing value, if every other value sits there.

    Null for a column with an unparsed value further down, which is mixed data
    rather than header rows spilled into it.
    """
    text = _text(name)
    filled = _filled(text)
    parses = filled & coercion.matches(text)
    unparsed = filled & ~parses
    # Not `arg_max`, whose pick among ties is unspecified and is the last one
    # on the streaming engine once it splits the column across threads.
    first = pl.int_range(pl.len()).filter(parses).min()

    # The first parsing row is not unparsed itself, so the running count there
    # is every unparsed value above it.
    return pl.when(unparsed.cum_sum().get(first) == unparsed.sum()).then(first)


@dataclass(slots=True, frozen=True)
class _Leading:
    """What the first rows of a table say about where its header ends."""

    leads: tuple[int | None, ...] = ()
    """Each mixed column's `_lead`."""

    rows: tuple[tuple[str | None, ...], ...] = ()
    """The first rows as text, as far as a header guess may reach."""


def _with_unparsed(
    frame: pl.LazyFrame, typed: Sequence[tuple[TextColumn, _Coercion]]
) -> tuple[tuple[tuple[TextColumn, _Coercion], ...], _Leading]:
    """Attach a sample of the values each mixed column could not parse.

    One collect for all of them rather than one apiece, and only for the
    columns that fell short: a column every value parsed in has nothing to
    sample.  Sorted so the same file always names the same values.  What a
    header guess reads of the first rows rides the same collect.
    """
    mixed = [pair for pair in typed if not pair[0].complete]

    if not mixed:
        return tuple(typed), _Leading()

    # One column per frame, aliased so the filter and the read address it by a
    # name no source column can shadow.
    sample = "value"
    value = pl.col(sample)
    leads, head, *sampled = pl.collect_all(
        [
            frame.select(
                _lead(column.name, coercion).alias(str(index))
                for index, (column, coercion) in enumerate(mixed)
            ),
            frame.head(_HEADER_SCAN).select(pl.all().cast(pl.String)),
            *(
                frame.select(_text(column.name).alias(sample))
                .filter(_filled(value) & ~coercion.matches(value))
                .unique()
                .sort(sample)
                .head(_UNPARSED_SAMPLE)
                for column, coercion in mixed
            ),
        ]
    )
    values = {
        column.name: tuple(collected[sample].to_list())
        for (column, _), collected in zip(mixed, sampled, strict=True)
    }

    return tuple(
        (replace(column, unparsed=values.get(column.name, ())), coercion)
        for column, coercion in typed
    ), _Leading(leads=leads.row(0), rows=tuple(head.rows()))


def _header_guess(source: Source, leading: _Leading) -> HeaderGuess | None:
    """Guess the header rows when the unparsed values crowd the first rows.

    Taken only when most mixed columns agree, at the lead they share most,
    since one sparse column starts its values late and one odd cell is not a
    header.  The names are the row among the leading ones, the loaded header
    included, that fills the most cells, the first of equals: a title or group
    row fills fewer, and a units row repeats the names' gaps.
    """
    counts = Counter(lead for lead in leading.leads if lead)

    if 2 * counts.total() <= len(leading.leads):
        return None

    top = max(counts.values())
    lead = min(lead for lead, count in counts.items() if count == top)

    if lead > _HEADER_SCAN:
        return None

    rows = [
        tuple(None if _UNNAMED.fullmatch(name) else name for name in source.columns),
        *leading.rows[:lead],
    ]
    filled = [sum(1 for cell in row if _label(cell)) for row in rows]
    best = filled.index(max(filled))
    above, below = source.skip_above_header, source.skip_below_header

    if best:
        above, below = above + below + best, lead - best
    else:
        below += lead

    return HeaderGuess(
        skip_above_header=above,
        skip_below_header=below,
        names=tuple(label for cell in rows[best] if (label := _label(cell))),
    )


MAX_STRAY_TEXT = 5
"""Most cells of text a column of fewer numbers holds and still counts as mixed rather than as text."""


def _typed(
    names: Sequence[str], counts: Sequence[int]
) -> list[tuple[TextColumn, _Coercion]]:
    """The best parsing dtype of each text column from the counts of `_column_counts`, none if nothing parses."""
    stride = 2 + len(_COERCIONS)
    typed: list[tuple[TextColumn, _Coercion]] = []

    for index, name in enumerate(names):
        base = index * stride
        # One zero-padded value makes the whole column an identifier, so it is
        # neither retyped nor reported as a number a cast would recover.
        padded = counts[base + 1]
        hits = [
            0 if coercion.numeric and padded else counts[base + 2 + rank]
            for rank, coercion in enumerate(_COERCIONS)
        ]
        # ``max`` keeps the first of equal counts, so a whole number stays one.
        rank = max(range(len(hits)), key=hits.__getitem__)

        # A column of text is not a typing problem to report: a name, a note,
        # and a `1,5` decimal comma are one thing to every test here, and
        # flagging them would put a hint on every string column of every
        # table.  Nor is a model name such as `200` among `a4` and `a6`, so a
        # column counts as mixed only where its numbers outweigh its text, or
        # its text is a few stray cells such as header rows or an `N/A`.
        unparsed = counts[base] - hits[rank]

        if not hits[rank] or (unparsed > hits[rank] and unparsed > MAX_STRAY_TEXT):
            continue

        coercion = _COERCIONS[rank]
        column = TextColumn(
            name=name,
            dtype=coercion.sql,
            parsed=hits[rank],
            total=counts[base],
        )
        typed.append((column, coercion))

    return typed


def _coerce(source: Source) -> Source:
    """Retype every text column whose values all parse as one other dtype.

    A CSV column of numbers that one ``N/A`` keeps as text, or a spreadsheet
    that hands out its dates as strings, is what otherwise turns a plain
    ``SUM`` or ``WHERE`` into a dtype error and a ``TRY_CAST`` retry.  The
    conversion is applied only when it loses nothing: every non-empty value
    parses, so no cell silently becomes null, and a zero-padded identifier is
    never mistaken for a number.  An empty cell reads as a missing value,
    which is what it means in every format here.

    A column that falls short is recorded with the count that did parse and a
    sample of what did not, so a genuinely mixed one is reported before a query
    fails over it, not after.  When those values sit in the first rows, the
    header rows that would drop them are guessed.
    """
    frame = source.frame
    names = [
        name for name, dtype in frame.collect_schema().items() if dtype == pl.String
    ]

    if not names:
        return source

    counts = (
        frame.select(
            expr.alias(str(position))
            for position, expr in enumerate(
                chain.from_iterable(_column_counts(name) for name in names)
            )
        )
        .collect()
        .row(0)
    )
    typed = _typed(names, counts)
    sampled, leading = _with_unparsed(frame, typed)
    text_columns = tuple(
        column for column, _ in sampled if source.report_complete or not column.complete
    )

    return replace(
        source,
        frame=frame.with_columns(
            coercion.convert(_text(column.name)).alias(column.name)
            for column, coercion in sampled
            if column.complete
        ),
        text_columns=text_columns,
        header_guess=_header_guess(source, leading),
    )


def _distinct(names: Sequence[str]) -> tuple[str, ...]:
    """Suffix a repeated name the way Polars does for a CSV header.

    >>> _distinct(["a", "b", "a"])
    ('a', 'b', 'a_duplicated_0')
    """
    seen: Counter[str] = Counter()
    distinct: list[str] = []

    for name in names:
        distinct.append(f"{name}_duplicated_{seen[name] - 1}" if seen[name] else name)
        seen[name] += 1

    return tuple(distinct)


def _rehead(source: Source, above: int, below: int) -> Source:
    """Take the column names *above* rows down and drop *below* rows under them.

    Done on the loaded frame rather than by the readers, so a spreadsheet and a
    delimited file count rows alike, as the sample shows them: the loaded
    header first, an empty spreadsheet row not at all.
    """
    if not above and not below:
        return source

    frame = source.frame
    columns = source.columns

    if above:
        cells = frame.slice(above - 1, 1).select(pl.all().cast(pl.String)).collect()

        if cells.is_empty():
            raise Retry(
                f"skip_above_header={above} skips every row of '{source.file_path}'"
            )

        names = _distinct(
            [_label(cell) or f"__UNNAMED__{i}" for i, cell in enumerate(cells.row(0))]
        )
        frame = frame.rename(dict(zip(columns, names, strict=True)))

    return replace(
        source,
        frame=frame.slice(above + below),
        skip_above_header=above,
        skip_below_header=below,
    )


def _load_excel(data: bytes, file_path: str, sheet: str | None) -> Source:
    """Load one worksheet and retain the workbook's sheet names."""
    # Deferred, since only spreadsheets need it.
    import fastexcel

    sheets = tuple(fastexcel.read_excel(data).sheet_names)

    if sheet is not None and sheet not in sheets:
        raise Retry(
            f"'{file_path}' has no sheet named '{sheet}'. "
            f"Available sheets: {', '.join(sheets)}."
        )

    name = sheet if sheet is not None else sheets[0]

    # Excel has no lazy scan, but only the query result reaches the context.
    # Reading through Polars rather than off `book` narrows a column of whole
    # numbers back to an integer, which Excel stores as a double either way.
    return Source(
        frame=pl.read_excel(data, sheet_name=name, infer_schema_length=None).lazy(),
        file_path=file_path,
        sheet=name,
        sheets=sheets,
    )


def _check_unpacked(data: bytes, file_path: str, max_chars: int) -> None:
    """Refuses a zipped workbook, such as an xlsx, whose parts unpack to more bytes than the budget allows.

    The sizes the zip declares are checked before it is decoded, since a zip bomb fits the budget packed.
    A workbook that is no zip, such as an xls, is left to the check of its packed size.

    Raises:
        Retry: If the parts cannot fit.
    """
    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile:
        return

    with archive:
        unpacked = sum(info.file_size for info in archive.infolist())

    check_read_budget(file_path, unpacked, max_chars)


def _load(
    mount: Mount, local: str, file_path: str, sheet: str | None, max_chars: int
) -> Source:
    """Load the file at *local* of *mount* as a frame, however its format has to be read.

    The file is read once through no symlink within the bytes *max_chars* allows, so every query reads it from memory.
    Parquet stays a lazy scan of those bytes, so a query decodes only the columns and row groups it needs.
    A delimited file is decoded whole within *max_chars*, every encoding alike,
    and a larger one refused, then parsed once.
    """
    suffix = PurePosixPath(local).suffix.lower()

    if suffix not in DELIMITERS:
        data = read_bytes_or_retry(
            mount,
            local,
            file_path,
            lambda size: check_read_budget(file_path, size, max_chars),
        )

        if suffix == ".parquet":
            return Source(frame=pl.scan_parquet(data), file_path=file_path)

        _check_unpacked(data, file_path, max_chars)

        return _load_excel(data, file_path, sheet)

    decoded = read_text_or_retry(mount, local, file_path, max_chars)

    if len(decoded.text) > max_chars:
        raise Retry(
            f"'{file_path}' is too large to query here ({len(decoded.text)} characters, "
            f"and one table may hold at most {max_chars})."
        )

    return Source(
        frame=pl.read_csv(
            decoded.text.encode(), separator=DELIMITERS[suffix], infer_schema=False
        ).lazy(),
        file_path=file_path,
        encoding=decoded.encoding,
        report_complete=False,
    )


def _open(
    file_path: str,
    mount: Mount,
    local: str,
    sheet: str | None,
    above: int,
    below: int,
    max_chars: int,
) -> Source:
    """Load, rehead, and retype the file.

    Loading parses a file whole, so one that cannot be read fails now rather
    than mid-query, and the query phase is left with only its own failures to report.
    """
    try:
        return _coerce(
            _rehead(_load(mount, local, file_path, sheet, max_chars), above, below)
        )

    except pl.exceptions.PolarsError as exc:
        raise Retry(f"could not read the table: {polars_message(exc)}") from exc


@lru_cache(maxsize=_CACHED_SOURCES)
def _open_unchanged(
    file_path: str,
    mount: Mount,
    local: str,
    sheet: str | None,
    above: int,
    below: int,
    max_chars: int,
    mtime_ns: int,
    size: int,
) -> Source:
    """`_open` of a local file, remembered for as long as its modification time and size stay."""
    return _open(file_path, mount, local, sheet, above, below, max_chars)


def open_source(
    file_path: str,
    mount: Mount,
    local: str,
    sheet: str | None,
    above: int,
    below: int,
    max_chars: int,
) -> Source:
    """Load, rehead, and retype the file, as `_open` does, reusing an earlier load of a file that did not change.

    The modification time and size key the reuse, and a file that is gone meanwhile is refused with `Retry`.
    """
    status = entry_stat(mount.root() / local)

    if status is None:
        raise Retry(f"'{file_path}' cannot be read.")

    return _open_unchanged(
        file_path,
        mount,
        local,
        sheet,
        above,
        below,
        max_chars,
        status.st_mtime_ns,
        status.st_size,
    )
