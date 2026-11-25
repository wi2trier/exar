import zipfile
from pathlib import Path

import pytest

from exar.helpers import get_name
from exar.memory import Mount
from exar.tools import ItemFailure, Retry, effect_of
from exar.tools.table import Table


def table(path: Path) -> Table:
    return Table(mounts=[Mount(source=path)])


async def test_table(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("region,amount\nEU,5\nUS,7\nEU,N/A\n")
    (tmp_path / "a.md").write_text("no table")
    tool = table(tmp_path)

    described = await tool(["a.csv"])
    assert described.metadata[0].columns == ("region", "amount")
    assert described.metadata[0].total_rows == 3
    assert "amount: VARCHAR" in str(described.return_value)
    assert "mixed text, wrap in TRY_CAST" in str(described.return_value)

    answered = await tool(
        ["a.csv"],
        [
            "SELECT region, SUM(TRY_CAST(amount AS DOUBLE)) AS n FROM t GROUP BY region ORDER BY region",
            "SELECT nope FROM t",
        ],
    )
    rows, failure = answered.metadata
    assert rows.rows == (("EU", "5.0"), ("US", "7.0"))
    assert isinstance(failure, ItemFailure)
    assert "==> SELECT nope FROM t <==\nfailed: query failed" in str(
        answered.return_value
    )

    with pytest.raises(Retry, match="not a tabular file"):
        await tool(["a.md"])

    for query in (
        "SELECT * FROM read_csv('a.csv')",
        f"SELECT * FROM t, \"READ_CSV\"('{tmp_path / 'a.csv'}')",
        "COPY t TO 'out.csv'",
        "SET threads = 64",
        "INSTALL httpfs",
    ):
        with pytest.raises(
            Retry, match="disabled|Cannot change configuration|Permission"
        ):
            await tool(["a.csv"], [query])

    allowed = await tool(["a.csv"], ["SELECT 'read_only' AS read_count FROM t LIMIT 1"])
    assert allowed.metadata[0].rows == (("read_only",),)

    assert (get_name(tool.tool()), effect_of(tool.tool())) == ("query_table", "read")


async def test_table_hints_name_only_mixed_columns(tmp_path: Path) -> None:
    makes = ["a4", "a6", "golf", "polo", "astra", "corsa", "200"]
    (tmp_path / "cars.csv").write_text(
        "make,price\n" + "".join(f"{make},1\n" for make in makes)
    )
    tool = table(tmp_path)

    assert "mixed text" not in str((await tool(["cars.csv"])).return_value)

    with pytest.raises(Retry) as failed:
        await tool(["cars.csv"], ["SELECT nope, make FROM t"])

    assert "Table t has the columns: make, price." in str(failed.value)
    assert "TRY_CAST" not in str(failed.value)


async def test_table_interrupts_a_slow_query(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("n\n" + "\n".join(map(str, range(5000))))
    tool = Table(mounts=[Mount(source=tmp_path)], timeout_seconds=0.5)

    with pytest.raises(Retry, match="took longer than 0.5 seconds"):
        await tool(["a.csv"], ["SELECT count(*) FROM t, t AS u, t AS v"])


async def test_table_coercion(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text(
        "aware,bad,phone,exp,word,big,num\n"
        "2024-01-02T10:00:00+02:00,2024-13-45 10:00:00,+4915112345,12E3,Inf,"
        f"{'1' * 41},1.5\n"
        "2024-01-02T10:00:00Z,2024-01-02 10:00:00,+4915112346,1E3,NaN,2,2\n"
    )
    result = (await table(tmp_path)(["a.csv"])).metadata[0]

    assert dict(zip(result.columns, result.dtypes, strict=True)) == {
        "aware": "TIMESTAMP WITH TIME ZONE",
        "bad": "VARCHAR",
        "phone": "VARCHAR",
        "exp": "VARCHAR",
        "word": "VARCHAR",
        "big": "VARCHAR",
        "num": "DOUBLE",
    }


async def test_table_refuses_a_delimited_file_beyond_its_budget(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_bytes("name\nM\xfcller\n".encode("cp1252") * 3)
    tool = Table(mounts=[Mount(source=tmp_path)], max_document_chars=20)

    with pytest.raises(Retry, match="too large"):
        await tool(["a.csv"])

    assert "decoded from" in str((await table(tmp_path)(["a.csv"])).return_value)


async def test_table_refuses_a_workbook_that_unpacks_beyond_its_budget(
    tmp_path: Path,
) -> None:
    with zipfile.ZipFile(tmp_path / "bomb.xlsx", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", b"\0" * 1_000_000)

    tool = Table(mounts=[Mount(source=tmp_path)], max_document_chars=10_000)

    with pytest.raises(Retry, match="too large"):
        await tool(["bomb.xlsx"])
