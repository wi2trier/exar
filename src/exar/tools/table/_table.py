"""The queries of `Table`, which need the `table` extra and are imported only when the tool is built or called."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from pydantic_ai import ToolReturn

from ...memory.mounts import Mount, current
from ...text import LineClip, hint_suffix, markdown_cell, markdown_table
from .. import BatchData, Retry
from ..batch import BatchShare, serve_batch, tool_return
from ..common import line_clip, resolve_file_or_retry
from ..formats import TABULAR_SUFFIXES
from ._fields import TableFields
from .frames import Source, open_source
from .hints import failure_message, result_hints, table_preamble
from .result import TableResult, relation_name

_LOCKED_DOWN: dict[str, str | bool | int | float | list[str]] = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "allow_community_extensions": False,
    "python_enable_replacements": False,
}
"""What a query connection may not do: read or write files, load extensions, or see Python variables by name."""


@dataclass(slots=True, frozen=True)
class _ArrowStream:
    """A frame as a bare Arrow stream, which DuckDB scans without pyarrow, where it converts a Polars frame through it."""

    frame: pl.DataFrame

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self.frame.__arrow_c_stream__(requested_schema)


def _connect(
    frames: Sequence[pl.DataFrame], threads: int, memory_limit: str
) -> duckdb.DuckDBPyConnection:
    """A throwaway in-memory database with the frames registered by their relation names, and its configuration locked.

    The query is the model's, so the lock is what keeps it to the frames:
    a file, an extension, a setting, or a Python variable is out of reach whatever it spells.
    """
    connection = duckdb.connect(
        config={
            **_LOCKED_DOWN,
            "threads": threads,
            "memory_limit": memory_limit,
            "lock_configuration": True,
        }
    )

    for index, frame in enumerate(frames):
        _ = connection.register(relation_name(index), _ArrowStream(frame))

    return connection


def _render(
    values: Sequence[tuple[Any, ...]],
    columns: tuple[str, ...],
    lines: list[str],
    clip: LineClip,
) -> tuple[tuple[tuple[str, ...], ...], str]:
    """Render every row as a markdown table under *lines*, keeping each one whole.

    Only the display clips a cell by *clip*, so ``TableResult.rows`` is the
    whole of what the row limit allowed.
    """
    if not values:
        return (), "\n".join([*lines, "(no rows)"])

    rows = tuple(tuple(markdown_cell(value) for value in row) for row in values)
    table = markdown_table(columns, (map(clip, row) for row in rows))

    return rows, "\n".join([*lines, table])


def _run(
    table: TableFields,
    connection: duckdb.DuckDBPyConnection,
    sources: tuple[Source, ...],
    query: str | None,
    limit: int,
    full_lines: bool,
    subject: int,
) -> ToolReturn[TableResult]:
    """Run the query over the registered frames and render what it returned.

    Every file is registered whether or not the query names it, so a join
    needs nothing but the SQL, and ``SHOW TABLES`` answers from the same
    database the query runs in.  Without a query the *subject* table alone
    is described by the same means, so its types read as a query sees them.
    """
    named = tuple((relation_name(i), source) for i, source in enumerate(sources))
    limit = table.preview_rows if query is None else limit
    total: int | None = None

    if query is None:
        named = (named[subject],)
        relation = named[0][0]
        total = int(
            connection.execute(f"SELECT count(*) FROM {relation}").fetchall()[0][0]
        )
        query_run = f"SELECT * FROM {relation}"
    else:
        query_run = query

    cursor = connection.execute(query_run)
    described = cursor.description or []
    columns = tuple(str(column[0]) for column in described)
    dtypes = tuple(str(column[1]) for column in described)
    # One row past the limit is what separates "all of it" from "the first
    # N", without fetching the rest to find out.
    fetched = cursor.fetchmany(limit + 1) if described else []

    preamble = table_preamble(
        named=named,
        columns=columns,
        dtypes=dtypes,
        total_rows=total,
        query=query,
    )
    clip = line_clip(table.max_cell_chars, full_lines, "cells")
    rows, body = _render(fetched[:limit], columns, preamble, clip)
    result = TableResult(
        query=query,
        tables=tuple(source.queried(name) for name, source in named),
        columns=columns,
        dtypes=dtypes,
        rows=rows,
        total_rows=total,
        truncated=len(fetched) > limit,
    )

    text = body + hint_suffix(
        result_hints(result, query, clip, table.max_named_columns)
    )

    return tool_return(result, text)


def _answer(
    table: TableFields,
    connection: duckdb.DuckDBPyConnection,
    sources: tuple[Source, ...],
    query: str | None,
    limit: int,
    full_lines: bool,
    subject: int,
) -> ToolReturn[TableResult]:
    """Run one query, or describe the *subject* table when there is none, with DuckDB's errors as retries."""
    try:
        return _run(table, connection, sources, query, limit, full_lines, subject)

    except duckdb.InterruptException as exc:
        raise Retry(
            f"the query took longer than {table.timeout_seconds:g} seconds, "
            "so filter or aggregate before joining"
        ) from exc

    except duckdb.Error as exc:
        raise Retry(
            failure_message(exc, sources, query, table.max_named_columns)
        ) from exc


async def _bounded(
    table: TableFields,
    frames: Sequence[pl.DataFrame],
    sources: tuple[Source, ...],
    query: str | None,
    limit: int,
    full_lines: bool,
    subject: int = 0,
) -> ToolReturn[TableResult]:
    """`_answer` on a connection of its own, interrupted once the timeout passes or the call is cancelled."""
    with _connect(frames, table.threads, table.memory_limit) as connection:
        work = asyncio.ensure_future(
            asyncio.to_thread(
                _answer, table, connection, sources, query, limit, full_lines, subject
            )
        )

        # Interrupting is what stops DuckDB, so the thread ends before the connection closes.
        try:
            return await asyncio.wait_for(asyncio.shield(work), table.timeout_seconds)

        except TimeoutError:
            connection.interrupt()

            return await work

        finally:
            if not work.done():
                connection.interrupt()
                _ = await asyncio.wait([work])


async def query_tables(
    table: TableFields,
    file_paths: Sequence[str],
    queries: Sequence[str] | None,
    sheet: str | None,
    skip_above_header: int,
    skip_below_header: int,
    limit: int,
    full_lines: bool,
) -> ToolReturn[BatchData[TableResult]]:
    """Loads the tables once and runs every query over them, or describes each table without queries."""
    mounts = current(table.mounts)
    resolved: list[tuple[str, Mount, str]] = []

    for path in file_paths:
        mount, local, _absolute = resolve_file_or_retry(mounts, path, table.hint)

        if Path(path).suffix.lower() not in TABULAR_SUFFIXES:
            raise Retry(
                f"'{path}' is not a tabular file. Queryable formats: "
                f"{', '.join(sorted(TABULAR_SUFFIXES))}.{table.hint('format', path)}"
            )

        resolved.append((path, mount, local))

    # Polars releases the GIL, but the calls still block, so they stay off
    # the event loop, where the queries of one batch overlap.  The files
    # are loaded and retyped once, and every query runs against what they
    # turned out to be.
    sources = tuple(
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    open_source,
                    path,
                    mount,
                    local,
                    sheet,
                    skip_above_header,
                    skip_below_header,
                    table.max_document_chars,
                )
                for path, mount, local in resolved
            )
        )
    )

    # Every query reads the same frames, collected once for the call.
    frames = await asyncio.to_thread(
        lambda: [source.frame.collect() for source in sources]
    )

    async def describe(index: int, _share: BatchShare) -> ToolReturn[TableResult]:
        return await _bounded(table, frames, sources, None, limit, full_lines, index)

    async def run(query: str, _share: BatchShare) -> ToolReturn[TableResult]:
        return await _bounded(table, frames, sources, query, limit, full_lines)

    if queries is None:
        return await serve_batch(
            range(len(sources)),
            describe,
            key=lambda index: sources[index].file_path,
            concurrency=table.concurrency,
        )

    return await serve_batch(queries, run, key=str, concurrency=table.concurrency)
