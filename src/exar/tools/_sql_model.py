"""The configuration, the arguments of a selection, and the statements of `exar.tools.sql.Sql`, which import no SQLAlchemy."""

import datetime
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schema import serializable
from ._host import HostEngine

__all__ = [
    "PARAM_TYPES",
    "Aggregate",
    "Filter",
    "Order",
    "ParamType",
    "Rows",
    "Selection",
    "SqlFields",
    "Statement",
]

type ParamType = Literal["string", "integer", "number", "boolean", "date", "datetime"]
"""The type of a parameter of a statement, as the model passes it in JSON."""

PARAM_TYPES: Mapping[ParamType, type[object]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "date": datetime.date,
    "datetime": datetime.datetime,
}

type Scalar = bool | int | float | str


class _Argument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Filter(_Argument):
    """A condition on a column, all of which a row must meet."""

    column: str = Field(description="A column of the table.")
    op: Literal[
        "=", "!=", "<", "<=", ">", ">=", "in", "like", "is_null", "not_null"
    ] = Field(
        description="How the column compares to the value, `like` with `%` and `_` as wildcards."
    )
    value: Scalar | list[Scalar] | None = Field(
        default=None,
        description="What to compare with, a list for `in`, a string for `like`, none for `is_null` and `not_null`.",
    )


class Aggregate(_Argument):
    """An aggregate over the rows of each group, or of all rows without `group_by`."""

    function: Literal["count", "sum", "avg", "min", "max"]
    column: str | None = Field(
        default=None, description="The column to aggregate, none to count rows."
    )
    alias: str = Field(
        description="The name of the result column, which `order_by` may use."
    )


class Order(_Argument):
    """A column or aggregate alias to sort by."""

    column: str
    descending: bool = False


@dataclass(slots=True, frozen=True, kw_only=True)
class Selection:
    """What `Sql.select` reads, checked against the reflected table when it runs."""

    table: str
    columns: tuple[str, ...] = ()
    filters: tuple[Filter, ...] = ()
    group_by: tuple[str, ...] = ()
    aggregates: tuple[Aggregate, ...] = ()
    order_by: tuple[Order, ...] = ()
    limit: int


@dataclass(slots=True, frozen=True, kw_only=True)
class Rows:
    """What a selection or a statement returned, the data of its tool result, with the cells as the driver typed them.

    Args:
        columns: Names of the columns.
        rows: The rows up to the limit, one cell per column each.
        truncated: Whether rows past the limit were cut.
        affected: How many rows a statement changed that returns none, None for rows.
    """

    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    truncated: bool = False
    affected: int | None = None


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Statement:
    """A statement the host wrote, offered as a tool of its own whose arguments are its bound parameters.

    Args:
        sql: The statement in the dialect of the database, each parameter written `:name`.
        description: What the statement does, for the model.
        params: The type of each parameter, all of which the model passes.
        effect: `"read"` runs it in a read-only transaction, `"write"` commits it, so policies ask first.
    """

    sql: str
    description: str
    params: Mapping[str, ParamType] = field(default_factory=dict)
    effect: Literal["read", "write"] = "read"


@dataclass(slots=True, frozen=True, kw_only=True)
class SqlFields:
    """The configuration of `exar.tools.sql.Sql`, which its selections and statements read."""

    engine: HostEngine
    prefix: str = ""
    tables: tuple[str, ...] = ()
    statements: Mapping[str, Statement] = field(default_factory=dict)
    max_rows: int = 100
    max_cell_chars: int = 200
    timeout_seconds: float = 20.0
    notes: str = ""
