"""Tools that explore a SQL database, read its rows by a typed selection, and run the statements of the host, configured as a serializable class.

`Sql` is a keyword-only dataclass like the tools of `exar.tools.files`, whose `tools` returns its functions.
The model never writes SQL: it lists and describes the allowed `tables` and reads them through `select`,
whose table, columns, and filters SQLAlchemy compiles against the reflected table with every value bound,
and it calls the `statements` the host wrote, one tool each, passing their bound parameters alone.
A selection and a statement of effect `"read"` run in a read-only transaction with a statement timeout,
and a statement of effect `"write"` commits, so policies ask before it runs.
The engine is not serialized but bound by name (`HostEngine`), so the host owns the URL with its credentials and the pool.
It needs the `sql` extra only when the tools are built or called, and supports PostgreSQL, MySQL, and SQLite.
"""

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from pydantic_ai import ToolReturn

from .._lazy import extra_hint
from ..schema import serializable
from ..text import first_line
from ..typing import Tool
from . import Retry, SchemaTool, as_tool
from ._host import Engine, HostEngine
from ._sql_model import (
    PARAM_TYPES,
    Aggregate,
    Filter,
    Order,
    ParamType,
    Rows,
    Selection,
    SqlFields,
    Statement,
)

__all__ = [
    "Aggregate",
    "Engine",
    "Filter",
    "HostEngine",
    "Order",
    "ParamType",
    "Rows",
    "Selection",
    "Sql",
    "Statement",
]

_EXPLORING = ("tables", "describe", "select")


TableArg = Annotated[str, Field(description="A table or view the tables tool lists.")]
ColumnsArg = Annotated[
    list[str] | None,
    Field(
        description="The columns to return, all when neither these nor aggregates are given."
    ),
]
FiltersArg = Annotated[
    list[Filter] | None, Field(description="Conditions every returned row meets.")
]
GroupByArg = Annotated[
    list[str] | None,
    Field(
        description="The columns to group by, which must include every returned column besides aggregates."
    ),
]
AggregatesArg = Annotated[
    list[Aggregate] | None, Field(description="Aggregates to return, per group.")
]
OrderByArg = Annotated[
    list[Order] | None, Field(description="What to sort by, first to last.")
]
LimitArg = Annotated[
    int | None, Field(ge=1, description="The most rows to return, capped by the tool.")
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Sql(SqlFields):
    """Lists, describes, and selects from the allowed tables of a database, and runs the statements of the host.

    Args:
        engine: The engine that connects, bound by the host.
        prefix: What to put before the names of the tools, such as the database they serve.
        tables: The tables and views the model may describe and select from, no exploring tools when empty.
        statements: The statements the model may run by their names, which follow `prefix` in the names of their tools.
        max_rows: Rows a selection or a statement returns, the rest is reported as cut.
        max_cell_chars: Characters of a cell before it is clipped.
        timeout_seconds: Bound of a statement, enforced by the database.
        notes: What to add to the description of the select tool, such as the meaning of the data.
    """

    kind: Literal["sql"] = field(default="sql", repr=False)

    def instructions(self) -> str:
        """What to tell the model about the tools, naming them after `prefix`."""
        lines: list[str] = []

        if self.tables:
            lines.append(
                f"List its tables with {self.prefix}tables, describe one with {self.prefix}describe,"
                f" and read rows with {self.prefix}select."
            )

        if self.statements:
            named = "\n".join(
                f"- {self.prefix}{name}: {first_line(statement.description)}"
                for name, statement in self.statements.items()
            )
            lines.append(f"It offers the statements\n{named}")

        return "\n".join(lines)

    def tools(self) -> list[Tool]:
        """The tools that list, describe, and select from the tables, if any, and one per statement.

        Raises:
            exar.errors.MissingExtra: If the `sql` extra is missing.
            exar.errors.Unbound: If no engine is bound to its name.
            ValueError: If the dialect of the engine has no usable driver, there are neither tables nor statements,
                or a statement has an unusable name or parameters that differ from those of its text.
        """
        with extra_hint("Sql", "sql"):
            from ._sql import check, check_statement

        check(self.engine)

        if not self.tables and not self.statements:
            raise ValueError("A database needs tables or statements")

        for name, statement in self.statements.items():
            if not name.isidentifier() or name in _EXPLORING:
                raise ValueError(
                    f"The statement name {name!r} is no identifier or names an exploring tool"
                )

            check_statement(name, statement)

        exploring: list[Tool] = (
            [
                as_tool(self.list_tables, name=f"{self.prefix}tables", effect="read"),
                as_tool(self.describe, name=f"{self.prefix}describe", effect="read"),
                as_tool(
                    self.select,
                    name=f"{self.prefix}select",
                    effect="read",
                    notes=self.notes,
                ),
            ]
            if self.tables
            else []
        )

        return [
            *exploring,
            *(
                self._statement_tool(name, statement)
                for name, statement in self.statements.items()
            ),
        ]

    def _statement_tool(self, name: str, statement: Statement) -> SchemaTool:
        adapters = {
            param: TypeAdapter(PARAM_TYPES[kind])
            for param, kind in statement.params.items()
        }

        async def call(**arguments: JsonValue) -> ToolReturn[Rows]:
            if unknown := sorted(arguments.keys() - adapters.keys()):
                raise Retry(f"unknown arguments {', '.join(unknown)}.")

            if missing := sorted(adapters.keys() - arguments.keys()):
                raise Retry(f"missing arguments {', '.join(missing)}.")

            try:
                values = {
                    param: adapter.validate_python(arguments[param])
                    for param, adapter in adapters.items()
                }
            except ValidationError as error:
                raise Retry(str(error)) from error

            with extra_hint("Sql", "sql"):
                from ._sql import run_statement

            return await run_statement(self, statement, values)

        return SchemaTool(
            name=f"{self.prefix}{name}",
            description=statement.description,
            schema={
                "type": "object",
                "properties": {
                    param: adapter.json_schema() for param, adapter in adapters.items()
                },
                "required": list(adapters),
                "additionalProperties": False,
            },
            effect=statement.effect,
            call=call,
        )

    async def list_tables(self) -> str:
        """List the tables and views of the database."""
        with extra_hint("Sql", "sql"):
            from ._sql import list_tables

        return await list_tables(self)

    async def describe(self, table: TableArg) -> str:
        """Describe a table with its columns, their types, its primary key, and its foreign keys."""
        with extra_hint("Sql", "sql"):
            from ._sql import describe

        return await describe(self, table)

    async def select(
        self,
        table: TableArg,
        columns: ColumnsArg = None,
        filters: FiltersArg = None,
        group_by: GroupByArg = None,
        aggregates: AggregatesArg = None,
        order_by: OrderByArg = None,
        limit: LimitArg = None,
    ) -> ToolReturn[Rows]:
        """Read the rows of a table as a Markdown table, filtered, grouped, aggregated, and sorted.

        Select the columns and filter or aggregate, since the result is cut after a number of rows.
        """
        selection = Selection(
            table=table,
            columns=tuple(columns or ()),
            filters=tuple(filters or ()),
            group_by=tuple(group_by or ()),
            aggregates=tuple(aggregates or ()),
            order_by=tuple(order_by or ()),
            limit=min(limit or self.max_rows, self.max_rows),
        )

        with extra_hint("Sql", "sql"):
            from ._sql import select

        return await select(self, selection)
