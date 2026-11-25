"""The words a `Table` call says to the model, its schema preambles, hints, and the messages of SQL errors."""

import re
from collections.abc import Sequence

import duckdb

from ...text import LineClip, plural, truncate_line
from .frames import Source
from .result import RELATION, HeaderGuess, TableResult, relation_name

__all__ = ["failure_message", "result_hints", "table_preamble"]

_BARE_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""What SQL takes unquoted; anything else has to be spelled in double quotes."""


def _quoted(name: str) -> str:
    """Spell a column the way a query has to type it.

    The schema listing is the only place a model ever sees a column name, so it
    has to be the spelling that parses: a header carrying a space, a unit, or a
    parenthesis is the norm in an exported spreadsheet and is a syntax error
    unquoted.

    >>> _quoted("amount")
    'amount'
    >>> _quoted("Zulaufmenge (D)")
    '"Zulaufmenge (D)"'
    """
    if _BARE_IDENTIFIER.fullmatch(name):
        return name

    escaped = name.replace('"', '""')

    return f'"{escaped}"'


def _query_mentions(query: str, name: str) -> bool:
    """Return whether SQL *query* spells column *name* as an identifier.

    >>> _query_mentions("SELECT id FROM t", "id")
    True
    >>> _query_mentions("SELECT paid FROM t", "id")
    False
    >>> _query_mentions('SELECT "Total cost" FROM t', "Total cost")
    True
    """
    if not _BARE_IDENTIFIER.fullmatch(name):
        return _quoted(name).casefold() in query.casefold()

    escaped = re.escape(name)
    pattern = rf'(?<![A-Za-z0-9_])(?:{escaped}|"{escaped}")(?![A-Za-z0-9_])'

    return re.search(pattern, query, flags=re.IGNORECASE) is not None


def _quoting_hint(query: str, columns: Sequence[str]) -> str:
    """Say how a column the query spells unquoted has to be typed.

    Spelled with that column, since a parse error has nothing else to correct
    itself from: the schema call it would otherwise be sent back to is the one
    the query came from.  Silent when every such column is quoted, as a hint
    about a rule the query kept only distracts from the error it did make.

    >>> _quoting_hint("SELECT Menge (kg) FROM t", ["Menge (kg)"])
    ' A column name that is not a bare identifier must be double-quoted, as in `SELECT "Menge (kg)" FROM t`.'
    >>> _quoting_hint('SELECT "Menge (kg)" FROM t', ["Menge (kg)"])
    ''
    """
    folded = query.casefold()
    example = next(
        (
            name
            for name in columns
            if _quoted(name) != name
            and name.casefold() in folded.replace(_quoted(name).casefold(), "")
        ),
        None,
    )
    if example is None:
        return ""

    return (
        " A column name that is not a bare identifier must be double-quoted, "
        f"as in `SELECT {_quoted(example)} FROM {RELATION}`."
    )


def _sentence(hint: str) -> str:
    """Capitalise a lowercase hint and close it, to stand alone in a message.

    >>> _sentence("pass x=1")
    'Pass x=1.'
    """
    return f"{hint[:1].upper()}{hint[1:]}."


def _naming(values: Sequence[str]) -> str:
    """Spell the values a column could not parse, when any were sampled.

    The count says a column is mixed; the values say what it is mixed with,
    which is what the answer turns on.  Naming them is what lets that be
    decided before a program crashes on one, or worse, quietly coerces it.
    """
    if not values:
        return ""

    spelled = ", ".join(f'"{truncate_line(value, 40)}"' for value in values)

    return f"; unparsed: {spelled}"


def failure_message(
    exc: duckdb.Error,
    sources: tuple[Source, ...],
    query: str | None,
    max_named: int,
) -> str:
    """Explain a failed query, naming the columns a cast would rescue.

    DuckDB's message already points at the spot in the query and suggests
    the names it meant, so only what the tables know is added to it.
    """
    query = query or ""
    detail = (
        f"query failed: {str(exc).strip()}\nThe tables are named "
        f"{', '.join(relation_name(index) for index in range(len(sources)))}; "
        "call without queries to see their columns and types, or run "
        "`SHOW TABLES`."
    )

    # A parse, binding, or unknown-function error is almost always an unquoted
    # header, since an exported sheet names its columns with spaces, units, and
    # parentheses, and `Menge (kg)` unquoted reads as a call of `Menge`.
    if isinstance(
        exc, duckdb.ParserException | duckdb.BinderException | duckdb.CatalogException
    ):
        detail += _quoting_hint(query, [name for s in sources for name in s.columns])

    # A column missing from a table is answered with its columns, and one whose
    # header is off with that header, since the query alone never says so.
    if isinstance(exc, duckdb.BinderException):
        detail += "".join(
            f" {_sentence(_named(f'table {relation_name(index)} has the columns', [_quoted(name) for name in source.columns], max_named))}"
            for index, source in enumerate(sources)
        )
        detail += "".join(
            f" {_sentence(_header_hint(relation_name(index), source.header_guess, max_named))}"
            for index, source in enumerate(sources)
            if source.header_guess is not None
        )

    # A type error over a mixed column, which the coercion pass had to leave
    # as text, is that text, so name it with the type a cast recovers.  A
    # column of plain text has no such type and goes unnamed.
    named = [
        f"{_quoted(column.name)} as {column.dtype}"
        for source in sources
        for column in source.text_columns
        if not column.complete and _query_mentions(query, column.name)
    ]

    if named and isinstance(exc, duckdb.BinderException | duckdb.ConversionException):
        detail += (
            " "
            + _named("These columns mix text into their values", named, max_named)
            + ". Wrap one in TRY_CAST(col AS type) to compare or "
            "aggregate it, which reads a value that does not parse as NULL."
        )

    return detail


def table_preamble(
    named: tuple[tuple[str, Source], ...],
    columns: tuple[str, ...],
    dtypes: tuple[str, ...],
    total_rows: int | None,
    query: str | None,
) -> list[str]:
    """Render a summary line per table, plus the schema when asked for.

    Every table is named on its own line, since a query that can join them
    has to know what each one is called and a list of paths says nothing
    about which took which name.
    """
    lines = [
        _summary(source, name, total_rows if not index else None)
        for index, (name, source) in enumerate(named)
    ]

    if query is not None:
        return lines

    source = named[0][1]

    if len(source.sheets) > 1:
        lines += ["", f"sheets: {', '.join(source.sheets)}"]

    # One column per line rather than a wide preview: the schema is the
    # point of this call, and a vertical list of it is never too wide.
    lines += ["", "columns:"]
    lines += [
        f"{_quoted(name)}: {dtype}" for name, dtype in zip(columns, dtypes, strict=True)
    ]

    return [*lines, "", "sample:"]


def _summary(source: Source, name: str, total_rows: int | None) -> str:
    """Say what one registered table is, and what a query addresses it as."""
    parts = [f"{name}: {source.file_path}"]

    if source.sheet is not None:
        parts.append(f"sheet '{source.sheet}'")

    if total_rows is not None:
        parts.append(f"{total_rows} {plural(total_rows, 'row')}")

    if source.encoding is not None:
        parts.append(f"decoded from {source.encoding}")

    return ", ".join(parts)


def result_hints(
    result: TableResult, query: str | None, clip: LineClip, max_named: int
) -> list[str]:
    """Name every cut, so a partial table cannot read as a complete one."""
    hints = clip.hints()

    if result.truncated and query is not None:
        hints.append(
            f"{len(result.rows)} rows returned, increase limit, narrow "
            "with WHERE, or aggregate with GROUP BY"
        )

    # What is wrong with the data rides every call, not only the schema
    # call: the query that most needs to hear it is the one that named the
    # column, and a run that opens with a SELECT never asks for the schema
    # afterwards.
    hints += [
        _header_hint(table.name, table.header_guess, max_named)
        for table in result.tables
        if table.header_guess is not None
    ]
    hints += _typing_hints(result, query, max_named)

    if query is None:
        hints.append(
            f"pass SQL queries over '{result.tables[0].name}' to filter or aggregate"
        )

    return hints


def _typing_hints(result: TableResult, query: str | None, max_named: int) -> list[str]:
    """Say which text columns were retyped, and which a cast has to reach.

    A mixed column is the one worth spending a hint on, named with the
    share that parses and the values that did not.  The values are what
    make this one hint enough for every shape of table: a repeated label
    says the header spilled a row into the data, a `<100` says a reading
    past a measurement limit, a `0,05` says a decimal comma, and each
    wants a different answer that is not this tool's to pick.  Reporting
    the values and asking generalises where a rule per shape does not.
    A two-row header and a three-row header read the same here.

    The worst shortfall leads, so the cap can only ever cut the columns
    least worth reading about.  Scoped to the columns this call touched,
    since a warning about a column the query never named is noise read
    past, and it was the cap on that noise that once cut the only column
    a query was about from behind nine it was not.  A column counts as
    touched when the result carries it or the query spells it, and it
    takes both: ``SELECT *`` names no column at all, while a column
    filtered on but not selected reaches the result under no name.
    """
    qualify = len(result.tables) > 1
    columns = [
        # Spelled as a query would have to type it: bare against one table,
        # and `t2."Menge (D)"` against several, which is the qualified name
        # SQL takes and not the quoted `"t2.Menge (D)"` that spelling the
        # two halves as one identifier would produce.
        (
            f"{table.name}.{_quoted(column.name)}" if qualify else _quoted(column.name),
            column,
        )
        for table in result.tables
        for column in table.text_columns
        if query is None
        or column.name in result.columns
        or _query_mentions(query, column.name)
    ]
    retyped = [name for name, column in columns if column.complete]
    mixed = sorted(
        ((name, column) for name, column in columns if not column.complete),
        key=lambda pair: pair[1].total - pair[1].parsed,
        reverse=True,
    )
    spelled = [
        f"{name} ({column.parsed} of {column.total} parse "
        f"as {column.dtype}{_naming(column.unparsed)})"
        for name, column in mixed
    ]
    hints: list[str] = []

    if retyped:
        hints.append(
            _named("stored as text, queryable as the type shown", retyped, max_named)
        )

    if spelled:
        hints.append(
            _named("mixed text, wrap in TRY_CAST to compare", spelled, max_named)
            + ". The values named are not numbers, so ask the user how each "
            "should be treated rather than substituting or skipping one, "
            "which changes every total drawn from that column"
        )

    return hints


def _header_hint(name: str, guess: HeaderGuess, max_named: int) -> str:
    """Name the header rows a table seems to have, and the names they give."""
    return _named(
        f"the first rows of {name} hold the values no type parses, so they "
        "look like header rows rather than data; pass "
        f"skip_above_header={guess.skip_above_header}, "
        f"skip_below_header={guess.skip_below_header} to name the columns",
        [_quoted(column) for column in guess.names],
        max_named,
    )


def _named(label: str, values: Sequence[str], max_named: int) -> str:
    """Spell a capped list without letting the cap pass for the whole of it, nor cut a single value.

    >>> _named("x", ["a", "b", "c"], 2), _named("x", ["a", "b", "c", "d"], 2)
    ('x: a, b, c', 'x: a, b, and 2 more columns')
    """
    rest = len(values) - max_named

    if rest <= 1:
        return f"{label}: {', '.join(values)}"

    spelled = ", ".join(values[:max_named])

    return f"{label}: {spelled}, and {rest} more {plural(rest, 'column')}"
