import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic_ai import ToolReturn

from exar.agents import RunConfig, run_async
from exar.errors import Conflict, Refused
from exar.memory import (
    Document,
    FileMemory,
    GatedMemory,
    InMemoryMemory,
    Mount,
    memory_tools,
)
from exar.policy import EffectPolicy, allow_all
from exar.tools import Retry, effect_of
from exar.tools.files import file_tools
from exar.typing import Action, Add, Changes, Remove, Update


def note(key: str, text: str, **meta: str | float) -> Update[str, Document]:
    return Update(key=key, value=Document(text=text, meta=meta), reason="test")


async def test_in_memory_memory() -> None:
    memory = InMemoryMemory[int, str](entries={0: "red apple"})
    changes = await memory.write(
        Changes(items=[Add(value="green apple", reason="seen")])
    )

    assert changes.applied and changes.items[0].key == 1
    assert [hit.key for hit in await memory.search("green apple")] == [1, 0]
    assert await memory.read([0, 5]) == {0: "red apple"}

    with pytest.raises(Conflict):
        await memory.write(
            Changes(items=[Remove(key=0, reason="old")]), expected={0: "x"}
        )


async def test_gated_memory() -> None:
    approved: list[Action] = []

    async def approve(action: Action) -> bool:
        approved.append(action)

        return action.target == "a"

    inner = InMemoryMemory[str, str](
        new_keys=lambda taken, count: [f"k{len(taken) + i}" for i in range(count)]
    )
    memory = GatedMemory(memory=inner, policy=EffectPolicy(), name="notes")

    async def write(key: str) -> None:
        await memory.write(Changes(items=[Update(key=key, value=key, reason="r")]))

    await run_async(write, "a", RunConfig(policy=allow_all, approve=approve))

    with pytest.raises(Refused):
        await run_async(write, "b", RunConfig(policy=allow_all, approve=approve))

    with pytest.raises(Refused):
        await write("c")

    assert await inner.read(["a", "b", "c"]) == {"a": "a"}
    assert [(a.name, a.target) for a in approved] == [("notes", "a"), ("notes", "b")]

    changes = await inner.write(Changes(items=[Add(value="x", reason="r")]))

    assert changes.items[0].key == "k1"


class SlowIndex(InMemoryMemory[str, Document]):
    """An index whose writes suspend, as one of a database does."""

    async def write(
        self,
        changes: Changes[str, Document],
        /,
        *,
        expected: Mapping[str, Document] | None = None,
    ) -> Changes[str, Document]:
        await asyncio.sleep(0)

        return await super().write(changes, expected=expected)


async def test_file_memory_syncs_a_mount_at_a_time(tmp_path: Path) -> None:
    (tmp_path / "lake.md").write_text("The lake is calm")
    memory = FileMemory(
        mounts=[Mount(source=tmp_path, target="/notes")],
        indexes={"/notes": SlowIndex()},
    )
    await memory.sync("/notes")
    (tmp_path / "lake.md").unlink()
    first, second = await asyncio.gather(memory.sync("/notes"), memory.sync("/notes"))

    assert [c.op for c in [*first.items, *second.items]] == ["remove"]


async def test_file_memory(tmp_path: Path) -> None:
    index = InMemoryMemory[str, Document]()
    memory = FileMemory(
        mounts=[
            Mount(source=tmp_path / "notes", target="/notes", write="allow"),
            Mount(source=tmp_path / "docs", target="/docs"),
        ],
        indexes={"/notes": index},
        trust={"/docs": 0.5},
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "sky.md").write_text(
        "---\nsource: web\n---\nThe sky is blue\n"
    )
    (tmp_path / "docs" / ".hidden.md").write_text("The sky is green")
    await memory.write(
        Changes(items=[note("/notes/sea.md", "The sea is blue", confidence=0.9)])
    )

    assert (
        (tmp_path / "notes" / "sea.md")
        .read_text()
        .startswith("---\nconfidence: 0.9\n---\n")
    )
    assert await memory.read(["/docs/sky.md", "/docs/missing.md"]) == {
        "/docs/sky.md": Document(text="The sky is blue\n", meta={"source": "web"})
    }
    assert [hit.key for hit in await memory.search("blue sea")] == [
        "/notes/sea.md",
        "/docs/sky.md",
    ]
    assert await memory.keys("/docs") == ["/docs/sky.md"]
    assert not (await memory.sync("/notes")).items
    # The index scores the text of a document, not its representation.
    assert [hit.score for hit in await index.search("meta")] == [0.0]

    with pytest.raises(ValueError, match="no mount"):
        await memory.sync("/missing")

    (tmp_path / "notes" / "sea.md").write_text("The sea is green")
    (tmp_path / "notes" / "lake.md").write_text("The lake is calm")
    changes = await memory.sync("/notes")

    assert {change.key for change in changes.items} == {
        "/notes/lake.md",
        "/notes/sea.md",
    }
    assert (await index.read(["/notes/sea.md"]))[
        "/notes/sea.md"
    ].text == "The sea is green"

    (tmp_path / "notes" / "lake.md").unlink()

    assert [c.op for c in (await memory.sync("/notes")).items] == ["remove"]

    # A new process reads the digests back from the index instead of indexing again.
    fresh = FileMemory(
        mounts=[Mount(source=tmp_path / "notes", target="/notes")],
        indexes={"/notes": index},
    )

    assert not (await fresh.sync("/notes")).items

    (tmp_path / "notes" / "sea.md").write_text(
        "---\nsource: web\n---\nThe sea is green"
    )

    assert [c.op for c in (await fresh.sync("/notes")).items] == ["update"]

    # A new process removes the files deleted meanwhile.
    (tmp_path / "notes" / "pond.md").write_text("The pond is still")
    await fresh.sync("/notes")
    (tmp_path / "notes" / "pond.md").unlink()
    fresh = FileMemory(
        mounts=[Mount(source=tmp_path / "notes", target="/notes")],
        indexes={"/notes": index},
    )

    assert [c.op for c in (await fresh.sync("/notes")).items] == ["remove"]

    # Links to files outside a mount are left out.
    (tmp_path / "secret.md").write_text("secret")
    (tmp_path / "notes" / "link.md").symlink_to(tmp_path / "secret.md")

    assert await memory.keys("/notes") == ["/notes/sea.md"]

    with pytest.raises(Refused):
        await memory.write(Changes(items=[note("/docs/new.md", "x")]))

    with pytest.raises(Conflict):
        await memory.write(
            Changes(
                items=[
                    Add(
                        key="/notes/sea.md",
                        value=Document(text="x"),
                        reason="r",
                    )
                ]
            )
        )

    with pytest.raises(ValueError, match="no file"):
        await memory.read(["/notes/../docs/sky.md"])

    # Excluded files are refused as keys, as they are left out of listings.
    with pytest.raises(ValueError, match="excluded"):
        await memory.write(Changes(items=[note("/notes/.git/config", "x")]))

    with pytest.raises(ValueError, match="excluded"):
        await memory.read(["/docs/.hidden.md"])

    # Concurrent additions of a key are serialized, so only one of them writes.
    def add(text: str) -> Changes[str, Document]:
        return Changes(
            items=[
                Add(
                    key="/notes/new.md",
                    value=Document(text=text),
                    reason="r",
                )
            ]
        )

    results = await asyncio.gather(
        *(memory.write(add(text)) for text in "ab"), return_exceptions=True
    )

    assert sorted(type(result).__name__ for result in results) == [
        "Changes",
        "Conflict",
    ]


async def test_tools(tmp_path: Path) -> None:
    memory = FileMemory(mounts=[Mount(source=tmp_path, target="/notes", write="allow")])
    await memory.write(Changes(items=[note("/notes/a.md", "one two\ntwo three")]))
    list_, grep, edit = file_tools(memory, "notes")[3:]

    assert effect_of(edit) == "write"
    assert await list_() == "/notes/a.md"
    found = await grep("thr+ee", context=0)

    assert isinstance(found, ToolReturn)
    assert found.return_value == "/notes/a.md\n2:two three"

    with pytest.raises(Retry, match="occurs 2 times"):
        await edit("/notes/a.md", "two", "2")

    assert await edit("/notes/a.md", "one", "1") == "Edited /notes/a.md."
    assert (tmp_path / "a.md").read_text() == "1 two\ntwo three"

    denied = memory_tools(
        GatedMemory(
            memory=InMemoryMemory[str, str](), policy=EffectPolicy(write="deny")
        )
    )

    with pytest.raises(Retry, match="denied"):
        await denied[2](Changes(items=[Add(value="x", reason="r")]))
