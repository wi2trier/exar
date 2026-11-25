"""The data of a `Table` call, which tables a query ran against and the rows it returned."""

from dataclasses import dataclass

__all__ = [
    "RELATION",
    "HeaderGuess",
    "QueriedTable",
    "TableResult",
    "TextColumn",
    "relation_name",
]

RELATION = "t"
"""Fixed name of the first relation, which avoids interpolating a file path."""


def relation_name(index: int) -> str:
    """Name the *index*-th table a query addresses: ``t``, ``t2``, ``t3``, ...

    Positional rather than derived from the filename, which in an exported
    workbook is a sentence with spaces in it and would need quoting in every
    query that named it.
    """
    return RELATION if not index else f"{RELATION}{index + 1}"


@dataclass(slots=True, frozen=True)
class TextColumn:
    """A column the file stores as text, and the dtype its values parse as."""

    name: str
    dtype: str
    parsed: int
    """Non-empty values that parse as *dtype*, of *total* that were tried."""

    total: int
    unparsed: tuple[str, ...] = ()
    """A few distinct values that did not parse, as they appear in the file."""

    @property
    def complete(self) -> bool:
        """Whether every value parses, which is what makes retyping lossless."""
        return self.parsed == self.total


@dataclass(slots=True, frozen=True)
class HeaderGuess:
    """The header rows that would put a table's column names where they seem to be.

    Guessed when the values no type parses sit in the first rows of the data,
    which is what a title, group, or units row next to the names looks like.
    Both counts are absolute, so they replace what the call passed.
    """

    skip_above_header: int
    skip_below_header: int
    names: tuple[str, ...]
    """The column names that skipping those rows would give."""


@dataclass(slots=True, frozen=True)
class QueriedTable:
    """One file the query was run against, under the SQL name it took."""

    name: str
    """What the query addresses it as: ``t``, then ``t2``, ``t3``, ..."""

    file_path: str
    sheet: str | None = None
    sheets: tuple[str, ...] = ()
    """Every sheet in the workbook, so a follow-up can name a different one."""

    encoding: str | None = None
    text_columns: tuple[TextColumn, ...] = ()
    """Text columns that hold numbers or dates, retyped where every value did."""

    header_guess: HeaderGuess | None = None


@dataclass(slots=True, frozen=True)
class TableResult:
    """The rows a query returned, with the tables they came from.

    The two halves are kept apart because they answer different questions and
    no longer stand one to one: ``tables`` describes each file that was
    registered, while the columns and rows describe what the query made of
    them, which for a join belongs to no single file.  A call without a query
    describes each table on its own, with ``query`` unset and ``tables``
    holding the one table described.
    """

    query: str | None = None
    tables: tuple[QueriedTable, ...] = ()
    columns: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    """Cells rendered as text, so a date or decimal survives the JSON trip."""

    total_rows: int | None = None
    """Rows in the source table, or ``None`` when a query decided the count."""

    truncated: bool = False

    @property
    def text_columns(self) -> tuple[TextColumn, ...]:
        """Every registered table's text columns, under their own names.

        Unqualified, because a column belongs to the table that holds it and
        `tables` already says which that is.  Where the two have to be
        spelled as one (a hint a join would have to type back), they are
        joined at the point that holds both.
        """
        return tuple(column for table in self.tables for column in table.text_columns)
