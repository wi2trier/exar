"""`Table`, which runs DuckDB SQL over the tables below mounts, loaded and typed by Polars, which it imports when it is built or called."""

from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Literal

from pydantic import Field
from pydantic_ai import ToolReturn

from ..._lazy import extra_hint
from ...schema import serializable
from .. import BatchData
from ..batch import batch_field
from ..common import FullLinesArg
from ._fields import TableFields
from .result import RELATION, TableResult

__all__ = [
    "Table",
    "TableFilePathsArg",
    "TableLimitArg",
    "TableQueriesArg",
    "TableSheetArg",
    "TableSkipAboveHeaderArg",
    "TableSkipBelowHeaderArg",
]

_DEFAULT_LIMIT = 100

TableFilePathsArg = Annotated[
    list[str],
    batch_field(
        "Full workspace paths of the tables to query together. The first is "
        f"render as '{RELATION}', the second '{RELATION}2', the third "
        f"'{RELATION}3', so a join reads "
        f'"FROM {RELATION} JOIN {RELATION}2 ON ...".'
    ),
]

TableQueriesArg = Annotated[
    list[str] | None,
    batch_field(
        f"DuckDB SQL SELECTs over the tables, the first named '{RELATION}', "
        f'e.g. "SELECT region, SUM(amount) AS total FROM {RELATION} '
        'GROUP BY region ORDER BY total DESC". Each runs on its own over the '
        "tables loaded once, so ask every question of them in one call. "
        "Supports the usual aggregates, WHERE, HAVING, ORDER BY, CTEs, window "
        "functions, joins, and `SHOW TABLES`. Rows have no set order without "
        "ORDER BY, and a long result is cut after its first rows, so add one "
        "whenever order or those first rows matter. DuckDB follows PostgreSQL, so "
        "write PostgreSQL spellings, and a refused function is named in the "
        "error with the names it may have meant. Omit the queries "
        "to get each table's columns, their types, the row count, and a few "
        "sample rows, which is the cheap first call. Numbers and dates are already "
        "typed as such, so cast only a column still shown as VARCHAR."
    ),
]

TableSheetArg = Annotated[
    str | None,
    Field(
        description=(
            "Worksheet to query in a spreadsheet that has more than one, "
            "applied to every spreadsheet given. Defaults to the first sheet; "
            "omit the queries to see them all."
        ),
    ),
]

TableSkipAboveHeaderArg = Annotated[
    int,
    Field(
        description=(
            "Rows to drop above the column names, counting the loaded header as "
            "the first, so the row after them names the columns. For a sheet "
            "whose names sit under a title or group row. Applied to every table "
            "given."
        ),
        ge=0,
    ),
]

TableSkipBelowHeaderArg = Annotated[
    int,
    Field(
        description=(
            "Rows to drop between the column names and the first data row, such "
            "as a units row. Applied to every table given."
        ),
        ge=0,
    ),
]

TableLimitArg = Annotated[
    int,
    Field(
        description=(
            f"Maximum query rows to return, capped by the tool. Defaults to {_DEFAULT_LIMIT}, "
            "increase it when the task needs more complete row-level data."
        ),
        ge=1,
    ),
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Table(TableFields):
    """Queries spreadsheets, delimited files, and Parquet files below mounts with SQL, a tool of effect `"read"`.

    The query is run instead of reading a table line by line.
    Row and cell budgets bound the result, and every cut is named in the formatted output.
    The argument `limit` lifts the first up to `max_rows`, `full_lines` the second.
    The query picks the columns, so every one it selects is shown.
    A call without a query describes each table, with `preview_rows` sample rows.

    Args:
        mounts: The mounts of the tables, a sequence or a function of the run.
        preview_rows: Sample rows of a table that is described.
        max_rows: Most rows a query returns whatever its `limit`, since a join can return far more rows than its tables hold.
        max_cell_chars: Characters per cell before it is clipped in the text of the result.
        max_named_columns: Most columns a hint spells out before it is noise itself.
        max_document_chars: Most characters of a delimited file, which is decoded whole in any encoding, a larger one refused.
        timeout_seconds: Bound of a query, after which DuckDB is interrupted.
        threads: Threads a query may use.
        memory_limit: Memory a query may use, in DuckDB's notation, past which it fails rather than spills.
        hint: What to add to a refusal of a directory or a file of another format, a tool to use instead.

    Examples:
        >>> import asyncio, tempfile
        >>> from pathlib import Path
        >>> folder = Path(tempfile.mkdtemp())
        >>> _ = (folder / "a.csv").write_text("id,amount\\n1,5\\n2,7\\n")
        >>> from exar.memory import Mount
        >>> table = Table(mounts=[Mount(source=folder)])
        >>> print(asyncio.run(table(["a.csv"], ["SELECT SUM(amount) AS total FROM t"])).return_value)
        ==> SELECT SUM(amount) AS total FROM t <==
        t: a.csv
        | total |
        | --- |
        | 12 |
    """

    extra: ClassVar[str | None] = "table"
    kind: Literal["table"] = field(default="table", repr=False)
    name: str = "query_table"

    async def __call__(
        self,
        file_paths: TableFilePathsArg,
        queries: TableQueriesArg = None,
        sheet: TableSheetArg = None,
        skip_above_header: TableSkipAboveHeaderArg = 0,
        skip_below_header: TableSkipBelowHeaderArg = 0,
        limit: TableLimitArg = _DEFAULT_LIMIT,
        full_lines: FullLinesArg = False,
    ) -> ToolReturn[BatchData[TableResult]]:
        """Query one or more spreadsheets or delimited documents with SQL.

        Runs DuckDB SQL SELECTs against the files and returns the resulting
        rows of each under its query.  The first file is render as ``t``,
        the second ``t2``, and so on, so a join reads ``FROM t JOIN t2 ON
        ...``.  The files are loaded once for all the queries, so ask every
        question of them in one call.  Prefer this over reading a table
        document: filtering and aggregating in the query costs a fraction of
        the context that reading the rows would, and it cannot silently lose
        the trailing columns of a wide row the way a line read does.  Call it
        without queries first to learn each table's columns, their types, and
        the row count.

        The dialect is DuckDB SQL, which follows PostgreSQL: it covers the usual
        aggregates, WHERE, GROUP BY, HAVING, ORDER BY, CTEs, window functions,
        and joins, and a function it lacks is refused with the names it may
        have meant.  ``SHOW TABLES`` lists what is registered.

        A sheet whose column names sit under a title or group row, or above a
        units row, is read with ``skip_above_header`` and ``skip_below_header``,
        which the result suggests when the first rows look like that.
        """
        with extra_hint("Table", "table"):
            from ._table import query_tables

        return await query_tables(
            self,
            file_paths,
            queries,
            sheet,
            skip_above_header,
            skip_below_header,
            min(limit, self.max_rows),
            full_lines,
        )
