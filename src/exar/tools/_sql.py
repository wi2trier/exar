"""The selections and statements of `Sql` over SQLAlchemy, which is imported only when the tools are built or called."""

import asyncio
import sqlite3
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import sqlalchemy as sa
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import ToolReturn
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql import operators
from sqlalchemy.sql.operators import OperatorType

from ..text import markdown_cell, markdown_table, truncate_line
from . import Retry
from ._host import HostEngine
from ._sql_model import Aggregate, Filter, Rows, Selection, SqlFields, Statement
from .batch import tool_return

_OPERATORS: Mapping[str, OperatorType] = {
    "=": operators.eq,
    "!=": operators.ne,
    "<": operators.lt,
    "<=": operators.le,
    ">": operators.gt,
    ">=": operators.ge,
    "in": operators.in_op,
    "like": operators.like_op,
    "is_null": operators.is_,
    "not_null": operators.is_not,
}
"""The SQLAlchemy operator of each operator of a `Filter`."""


def _limit_sqlite(
    conn: sa.Connection, timeout_seconds: float, *, read_only: bool
) -> Callable[[], None]:
    """Aborts the statements of a SQLite connection after the timeout and makes it read-only if asked, returns what undoes both."""
    raw = conn.connection.driver_connection

    if not isinstance(raw, sqlite3.Connection):
        raise TypeError("Only the sqlite3 driver is supported here")

    deadline = time.monotonic() + timeout_seconds
    raw.set_progress_handler(lambda: time.monotonic() > deadline, 1000)

    if read_only:
        _ = conn.exec_driver_sql("PRAGMA query_only = ON")

    def undo() -> None:
        raw.set_progress_handler(None, 0)

        if read_only:
            _ = conn.exec_driver_sql("PRAGMA query_only = OFF")

    return undo


def _limit_mysql(
    engine: sa.Engine, conn: sa.Connection, timeout_seconds: float
) -> Callable[[], None]:
    """Kills the statement of a MySQL connection that runs past the timeout, returns what disarms it.

    MySQL bounds only a `SELECT` by `max_execution_time`, so a write is bounded from a second connection.
    Disarming waits for a kill under way, so none reaches the next user of the pooled connection.
    The id of the connection is kept with the pooled connection, so it is queried once per connection.
    """
    info = conn.connection.info

    if (thread_id := info.get("mysql_connection_id")) is None:
        thread_id = info["mysql_connection_id"] = int(
            conn.exec_driver_sql("SELECT CONNECTION_ID()").scalar_one()
        )
    lock = threading.Lock()
    disarmed = threading.Event()

    def kill() -> None:
        with lock:
            if disarmed.is_set():
                return

            with engine.connect() as killer:
                _ = killer.exec_driver_sql(f"KILL QUERY {thread_id}")

    timer = threading.Timer(timeout_seconds, kill)
    timer.daemon = True
    timer.start()

    def disarm() -> None:
        timer.cancel()

        with lock:
            disarmed.set()

    return disarm


@contextmanager
def _connection(
    engine: sa.Engine, timeout_seconds: float, *, write: bool
) -> Generator[sa.Connection]:
    """A connection in a transaction with a statement timeout, read-only unless it may `write`, rolled back unless committed.

    PostgreSQL and MariaDB bound every statement on the server and MySQL its reads,
    while its writes are killed from a second connection and SQLite aborts the statements itself.

    Raises:
        ValueError: If the dialect has no read-only mode here.
        TypeError: If a SQLite connection is not driven by `sqlite3`.
    """
    millis = int(timeout_seconds * 1000)
    undo: Callable[[], None] = lambda: None

    with engine.connect() as conn:
        match engine.dialect.name:
            case "postgresql":
                if not write:
                    _ = conn.exec_driver_sql("SET TRANSACTION READ ONLY")

                _ = conn.exec_driver_sql(f"SET LOCAL statement_timeout = {millis}")
            case "mysql" | "mariadb":
                if not write:
                    _ = conn.exec_driver_sql("SET TRANSACTION READ ONLY")

                if (
                    isinstance(engine.dialect, MySQLDialect)
                    and engine.dialect.is_mariadb
                ):
                    _ = conn.exec_driver_sql(
                        f"SET SESSION max_statement_time = {timeout_seconds:f}"
                    )
                else:
                    _ = conn.exec_driver_sql(
                        f"SET SESSION max_execution_time = {millis}"
                    )

                    if write:
                        undo = _limit_mysql(engine, conn, timeout_seconds)
            case "sqlite":
                undo = _limit_sqlite(conn, timeout_seconds, read_only=not write)
            case dialect:
                raise ValueError(f"The {dialect} dialect is not supported")

        try:
            yield conn
        finally:
            conn.rollback()
            undo()


def _markdown(
    columns: Sequence[str], rows: Sequence[Sequence[object]], clip: int
) -> str:
    def cell(value: object) -> str:
        return markdown_cell(None if value is None else truncate_line(str(value), clip))

    return markdown_table(columns, (map(cell, row) for row in rows), compact=True)


def _rows(result: sa.CursorResult[Any], max_rows: int, clip: int) -> ToolReturn[Rows]:
    """The rows of a result cut after `max_rows`, for the model as a Markdown table, or the count of rows a statement changed."""
    if not result.returns_rows:
        return tool_return(
            Rows(affected=result.rowcount), f"({result.rowcount} rows affected)"
        )

    fetched = result.fetchmany(max_rows + 1)
    rows = Rows(
        columns=tuple(result.keys()),
        rows=tuple(tuple(row) for row in fetched[:max_rows]),
        truncated=len(fetched) > max_rows,
    )
    cut = (
        f"\n\n(cut at {max_rows} rows, narrow the selection)" if rows.truncated else ""
    )

    return tool_return(rows, _markdown(rows.columns, rows.rows, clip) + cut)


def _sqlalchemy(engine: HostEngine) -> sa.Engine:
    bound = engine()

    if not isinstance(bound, sa.Engine):
        raise TypeError(f"Engine {engine.name!r} is not a sqlalchemy.Engine")

    return bound


def check(engine: HostEngine) -> None:
    """Checks that the driver of the dialect of the engine is installed.

    Raises:
        ValueError: If the dialect has no usable driver.
    """
    dialect = _sqlalchemy(engine).dialect

    try:
        _ = dialect.import_dbapi()
    except ImportError as error:
        raise ValueError(
            f"The {dialect.name} dialect has no usable driver: {error}"
        ) from error


def check_statement(name: str, statement: Statement) -> None:
    """Checks that the declared parameters of a statement are exactly those its text binds.

    Raises:
        ValueError: If they differ.
    """
    bound = set(sa.text(statement.sql).compile().params)

    if bound != statement.params.keys():
        raise ValueError(
            f"The statement {name!r} binds {sorted(bound)} but declares the parameters {sorted(statement.params)}"
        )


async def _run[R](
    config: SqlFields, work: Callable[[sa.Connection], R], *, write: bool = False
) -> R:
    """Runs blocking work in a thread on a connection that commits if it may `write`, turning database errors into `Retry`."""

    def blocking() -> R:
        with _connection(
            _sqlalchemy(config.engine), config.timeout_seconds, write=write
        ) as conn:
            done = work(conn)

            if write:
                conn.commit()

            return done

    try:
        return await asyncio.to_thread(blocking)
    except SQLAlchemyError as error:
        raise Retry(str(getattr(error, "orig", None) or error)) from error


def _reflect(config: SqlFields, conn: sa.Connection, name: str) -> sa.Table:
    """The allowed table or view *name* as the database describes it."""
    unknown = Retry(f"unknown table {name!r}, list the tables first.")

    if name not in config.tables:
        raise unknown

    try:
        return sa.Table(name, sa.MetaData(), autoload_with=conn)
    except NoSuchTableError as error:
        raise unknown from error


async def list_tables(config: SqlFields) -> str:
    """The sorted names of the allowed tables and views that exist."""

    def work(conn: sa.Connection) -> list[str]:
        inspector = sa.inspect(conn)
        existing = {*inspector.get_table_names(), *inspector.get_view_names()}

        return sorted(existing.intersection(config.tables))

    return "\n".join(await _run(config, work)) or "(no tables)"


async def describe(config: SqlFields, table: str) -> str:
    """The DDL of an allowed table or view in the dialect of the database, with its columns, primary key, and foreign keys."""

    def work(conn: sa.Connection) -> str:
        return str(CreateTable(_reflect(config, conn, table)).compile(conn)).strip()

    return await _run(config, work)


def _column(table: sa.Table, name: str) -> sa.Column[Any]:
    try:
        return table.columns[name]
    except KeyError:
        raise Retry(
            f"unknown column {name!r} of {table.name}, describe the table first."
        ) from None


def _value(column: sa.Column[Any], value: object) -> object:
    """The value the model wrote for a column, a string parsed into the Python type of the column, such as a date."""
    if isinstance(value, list):
        return [_value(column, item) for item in value]

    try:
        kind = column.type.python_type
    except NotImplementedError:
        return value

    if not isinstance(value, str) or kind is str:
        return value

    try:
        return TypeAdapter(kind).validate_python(value)
    except ValidationError as error:
        raise Retry(
            f"{value!r} is no {kind.__name__} for the column {column.name!r}."
        ) from error


def _condition(table: sa.Table, condition: Filter) -> sa.ColumnElement[bool]:
    """The condition of a filter, whose misuse, such as a single value for `in`, SQLAlchemy refuses and `_run` retries."""
    column = _column(table, condition.column)
    value = (
        None
        if condition.op in ("is_null", "not_null")
        else _value(column, condition.value)
    )

    return column.operate(_OPERATORS[condition.op], value)


def _aggregate(table: sa.Table, aggregate: Aggregate) -> sa.Label[Any]:
    function = getattr(sa.func, aggregate.function)
    arguments = () if aggregate.column is None else (_column(table, aggregate.column),)

    return function(*arguments).label(aggregate.alias)


def build_select(table: sa.Table, selection: Selection) -> sa.Select[*tuple[Any, ...]]:
    """The statement of a selection, every name checked against the table and every value bound.

    Raises:
        Retry: If the selection names a column the table lacks or returns a column it does not group by.
        sqlalchemy.exc.ArgumentError: If a filter is misused, such as a single value for `in`.
    """
    columns = [_column(table, name) for name in selection.columns]
    groups = [_column(table, name) for name in selection.group_by]
    aggregates = [_aggregate(table, aggregate) for aggregate in selection.aggregates]
    aliases = {label.name: label for label in aggregates}

    if len(aliases) < len(aggregates):
        raise Retry("the aliases of the aggregates must differ.")

    if (aggregates or groups) and (
        ungrouped := [
            column.name for column in columns if column.name not in selection.group_by
        ]
    ):
        raise Retry(
            f"group by the returned columns {', '.join(ungrouped)} or aggregate them."
        )

    def order(name: str) -> sa.ColumnElement[Any]:
        return aliases[name] if name in aliases else _column(table, name)

    return (
        sa.select(
            *(columns or (groups if groups or aggregates else table.columns)),
            *aggregates,
        )
        .select_from(table)
        .where(*(_condition(table, condition) for condition in selection.filters))
        .group_by(*groups)
        .order_by(
            *(
                order(item.column).desc() if item.descending else order(item.column)
                for item in selection.order_by
            )
        )
        .limit(selection.limit + 1)
    )


async def select(config: SqlFields, selection: Selection) -> ToolReturn[Rows]:
    """The rows of a selection, cut after its limit."""

    def work(conn: sa.Connection) -> ToolReturn[Rows]:
        statement = build_select(_reflect(config, conn, selection.table), selection)

        return _rows(conn.execute(statement), selection.limit, config.max_cell_chars)

    return await _run(config, work)


async def run_statement(
    config: SqlFields, statement: Statement, values: dict[str, object]
) -> ToolReturn[Rows]:
    """The rows a statement returns, or the count of rows it changed, committed if it writes."""

    def work(conn: sa.Connection) -> ToolReturn[Rows]:
        return _rows(
            conn.execute(sa.text(statement.sql), values),
            config.max_rows,
            config.max_cell_chars,
        )

    return await _run(config, work, write=statement.effect == "write")
