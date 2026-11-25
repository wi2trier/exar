from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import ArgumentError

from exar.helpers import get_name
from exar.tools import Retry, SchemaTool, effect_of
from exar.tools._sql import build_select
from exar.tools.batch import unpack
from exar.tools.sql import (
    Aggregate,
    Filter,
    HostEngine,
    Order,
    Rows,
    Selection,
    Sql,
    Statement,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[sa.Engine]:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")

    with engine.begin() as conn:
        _ = conn.exec_driver_sql(
            "create table pets (id integer primary key, name text not null, kind text)"
        )
        _ = conn.exec_driver_sql("create table secrets (id integer primary key)")
        _ = conn.exec_driver_sql(
            "insert into pets (name, kind) values ('a|b', 'cat'), ('c', 'dog'), ('d', 'cat')"
        )

    yield engine
    engine.dispose()


def sql(engine: sa.Engine, **statements: Statement) -> Sql:
    return Sql(
        engine=HostEngine("db", lambda: engine),
        prefix="sql_",
        max_rows=2,
        tables=("pets",),
        statements=statements,
    )


async def test_explore_and_select(engine: sa.Engine) -> None:
    tools = {
        get_name(tool): tool
        for tool in sql(engine).tools()
        if not isinstance(tool, SchemaTool)
    }

    assert set(tools) == {"sql_tables", "sql_describe", "sql_select"}
    assert set(map(effect_of, tools.values())) == {"read"}
    assert await tools["sql_tables"]() == "pets"
    described = str(await tools["sql_describe"]("pets"))

    assert described.startswith("CREATE TABLE pets")
    assert "name TEXT NOT NULL" in described and "PRIMARY KEY (id)" in described

    with pytest.raises(Retry, match="unknown table"):
        await tools["sql_describe"]("secrets")

    with pytest.raises(Retry, match="unknown table"):
        await tools["sql_select"]("secrets")

    names = unpack(
        await tools["sql_select"](
            "pets", columns=["name"], order_by=[Order(column="id")]
        )
    )

    assert (
        names.text
        == "| name |\n|---|\n| a\\|b |\n| c |\n\n(cut at 2 rows, narrow the selection)"
    )
    assert names.data == Rows(
        columns=("name",), rows=(("a|b",), ("c",)), truncated=True
    )
    assert (
        unpack(
            await tools["sql_select"](
                "pets",
                group_by=["kind"],
                aggregates=[Aggregate(function="count", alias="n")],
                filters=[Filter(column="kind", op="in", value=["cat", "dog"])],
                order_by=[Order(column="n", descending=True)],
            )
        ).text
        == "| kind | n |\n|---|---|\n| cat | 2 |\n| dog | 1 |"
    )


def test_selection_binds_values_and_refuses_unknown_names() -> None:
    table = sa.Table(
        "pets", sa.MetaData(), sa.Column("id", sa.Integer), sa.Column("name", sa.String)
    )
    injected = "x'; drop table pets; --"
    statement = build_select(
        table,
        Selection(
            table="pets",
            columns=("name",),
            filters=(Filter(column="name", op="=", value=injected),),
            limit=5,
        ),
    )
    compiled = statement.compile()

    assert injected not in str(compiled) and injected in compiled.params.values()

    for selection in (
        Selection(table="pets", columns=("name; drop table pets",), limit=5),
        Selection(
            table="pets", filters=(Filter(column="id", op="in", value=1),), limit=5
        ),
        Selection(
            table="pets",
            columns=("name",),
            aggregates=(Aggregate(function="count", alias="n"),),
            limit=5,
        ),
    ):
        with pytest.raises((Retry, ArgumentError)):
            _ = build_select(table, selection)


async def test_statements_bind_their_parameters_and_declare_their_effect(
    engine: sa.Engine,
) -> None:
    tools = {
        tool.name: tool
        for tool in sql(
            engine,
            rename=Statement(
                sql="update pets set name = :name where id = :id",
                description="Rename a pet.",
                params={"id": "integer", "name": "string"},
                effect="write",
            ),
            named=Statement(
                sql="select id from pets where name = :name",
                description="Find a pet.",
                params={"name": "string"},
            ),
        ).tools()
        if isinstance(tool, SchemaTool)
    }

    assert {name: tool.effect for name, tool in tools.items()} == {
        "sql_rename": "write",
        "sql_named": "read",
    }
    assert tools["sql_rename"].schema["required"] == ["id", "name"]
    assert unpack(await tools["sql_rename"](id=2, name="x")).data == Rows(affected=1)
    assert unpack(await tools["sql_named"](name="x")).text == "| id |\n|---|\n| 2 |"

    with pytest.raises(Retry, match="missing"):
        await tools["sql_named"].call()

    (wipe,) = Sql(
        engine=HostEngine("db", lambda: engine),
        statements={"wipe": Statement(sql="delete from pets", description="Wipe.")},
    ).tools()

    assert isinstance(wipe, SchemaTool)

    with pytest.raises(Retry, match="readonly"):
        await wipe.call()


def test_statements_must_declare_what_they_bind(engine: sa.Engine) -> None:
    for statement in (
        Statement(sql="select * from pets where id = :id", description="x"),
        Statement(sql="select * from pets", description="x", params={"id": "integer"}),
    ):
        with pytest.raises(ValueError, match="binds"):
            _ = sql(engine, find=statement).tools()

    with pytest.raises(ValueError, match="tables or statements"):
        _ = Sql(engine=HostEngine("db", lambda: engine)).tools()
