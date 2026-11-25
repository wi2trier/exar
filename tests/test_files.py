import errno
import json
import os
import shutil
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from pydantic_ai import ToolReturn

from exar.helpers import frozendict, get_name
from exar.memory import Mount
from exar.memory.mounts import canonical, narrow, open_beneath, resolve
from exar.tools import Retry, effect_of
from exar.tools.batch import unpack
from exar.tools.common import resolve_file_or_retry
from exar.tools.files import DocumentRead, Grep, Jq, Ls, Read


def mounts(*pairs: tuple[Path, str]) -> list[Mount]:
    return [Mount(source=path, target=target, exclude=()) for path, target in pairs]


def test_mounts_resolve_prefixes(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x.md").write_text("x")
    both = mounts((tmp_path / "a", "@a"), (tmp_path / "b", "@b"))

    assert narrow(both, "@b/*.md") == ([both[1]], "*.md")
    assert resolve(both, "x.md") is None
    assert resolve_file_or_retry(both, "@b/x.md")[1] == "x.md"

    with pytest.raises(Retry, match="This tool addresses @a, @b"):
        resolve_file_or_retry(both, "x.md")

    with pytest.raises(Retry, match="not found"):
        resolve_file_or_retry(both, "@b/../a/../../x.md")


def test_mounts_include_exclude_hidden(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("a")
    (tmp_path / "docs" / "b.txt").write_text("b")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.md").write_text("c")
    mount = Mount(
        source=tmp_path, include=("*.md",), exclude=(".*", "docs/*", "!docs/a.md")
    )

    assert mount.excludes("docs/b.txt") and not mount.excludes("docs/a.md")
    assert [local for local, _ in mount.walk(tmp_path)] == ["docs/a.md"]


async def test_grep(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("one\nNeedle two\nthree\nfour\nneedle\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.md").write_text("needle\n")
    (tmp_path / ".hidden.md").write_text("needle\n")
    grep = Grep(mounts=mounts((tmp_path, "/notes")))

    result = await grep("needle")

    assert isinstance(result, ToolReturn)
    assert [match.file_path for match in result.metadata] == ["/notes/a.md"]
    assert (
        result.return_value
        == "/notes/a.md\n1-one\n2:Needle two\n3-three\n4-four\n5:needle"
    )
    assert (await grep("needle", output_mode="count")).return_value == "/notes/a.md: 2"
    assert (
        await grep("needle", case_sensitive=True, context=0)
    ).return_value == "/notes/a.md\n5:needle"
    assert (await grep("e.*e", literal=True)).return_value == "(no matches)"
    assert (await grep("needle", glob="*.txt")).return_value == "(no matches)"
    assert (
        await grep("needle", include_ignored=True, output_mode="files_with_matches")
    ).return_value == "/notes/a.md\n/notes/node_modules/b.md"

    with pytest.raises(Retry, match="Invalid pattern"):
        await grep("needle(")


async def test_grep_is_a_tool(tmp_path: Path) -> None:
    tool = Grep(
        mounts=mounts((tmp_path, "")),
        name="search_files",
        description="Searches.",
        arguments=frozendict(pattern="What to find."),
    ).tool()

    assert (get_name(tool), tool.__doc__, effect_of(tool)) == (
        "search_files",
        "Searches.",
        "read",
    )
    assert isinstance(await tool("x"), ToolReturn)


async def test_grep_is_serializable(tmp_path: Path) -> None:
    adapter = TypeAdapter(Grep)
    grep = adapter.validate_python(
        {
            "mounts": [{"source": str(tmp_path), "target": "/notes"}],
            "engine": {"kind": "python"},
        }
    )

    assert (await grep("x")).return_value == "(no matches)"
    assert adapter.dump_python(grep)["mounts"][0]["target"] == "/notes"


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
async def test_jq(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(json.dumps({"items": [1, 2]}))
    (tmp_path / "b.md").write_text("text")
    jq = Jq(mounts=mounts((tmp_path, "")))

    result = await jq(["a.json", "b.md"], ".items | add")

    assert (
        result.return_value
        == "==> a.json <==\n3\n\n==> b.md <==\nfailed: 'b.md' is not a JSON document. Filterable formats: .json."
    )
    shape = await jq(["a.json"])

    assert str(shape.return_value).endswith(
        "[shape only, pass a filter to select values]"
    )


async def test_read(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("one\ntwo\n" + "x" * 30 + "\nfour\n")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01\x02")
    read = Read(
        mounts=mounts((tmp_path, "/notes")),
        max_line_chars=10,
        viewer="look",
        media_type=lambda path: "image/png" if path.endswith(".png") else None,
        advice=lambda mount, local: f"see {local}",
    )

    result = await read(
        [
            DocumentRead("/notes/a.md", limit=3),
            DocumentRead("/notes/pic.png"),
            DocumentRead("/notes/b.md"),
        ]
    )

    assert result.return_value == (
        "==> /notes/a.md (limit=3) <==\n"
        f"lines 1-3 of 4 (hash {result.metadata[0].content_hash}):\n1: one\n2: two\n3: xxxxxxxxx…\n\n"
        "[see a.md; lines clipped at 10 chars, pass full_lines=true to see them whole; 1 more line, call again with offset=4]\n\n"
        "==> /notes/pic.png <==\nfailed: '/notes/pic.png' is a image/png binary, use look to send it to a vision model.\n\n"
        "==> /notes/b.md <==\nfailed: '/notes/b.md' not found."
    )
    assert "x" * 30 in result.metadata[0].content
    assert str(
        (await read([DocumentRead("/notes/a.md", offset=9)])).return_value
    ).endswith("(offset 9 is past the end of the file with 4 lines)")

    with pytest.raises(Retry, match="is not text-like content"):
        await read([DocumentRead("/notes/bin.dat")])


async def test_ls(tmp_path: Path) -> None:
    (tmp_path / "docs" / "deep").mkdir(parents=True)
    (tmp_path / "docs" / "a.md").write_text("one")
    (tmp_path / "docs" / "deep" / "b.md").write_text("two")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "c.md").write_text("three")
    ls = Ls(
        mounts=mounts((tmp_path, "/notes")),
        case_base=lambda mount, local: local == "docs/a.md",
    )

    top = await ls()
    names = [entry.file_path for entry in top.metadata]

    assert names == ["/notes/docs"]
    assert (
        "[1 hidden entry (common build and vendor directories), pass include_ignored=True to reveal them; 3 entries below max_depth=1"
        in unpack(top).text
    )
    assert [
        entry.file_path for entry in (await ls("/notes/docs", max_depth=2)).metadata
    ] == ["/notes/docs/a.md", "/notes/docs/deep", "/notes/docs/deep/b.md"]
    assert "(case base)" in unpack(await ls("docs")).text
    assert [
        entry.file_path
        for entry in (await ls(patterns=["*.md"], include_ignored=True)).metadata
    ] == ["/notes/docs/a.md", "/notes/docs/deep/b.md", "/notes/node_modules/c.md"]
    assert (await ls(patterns=["*.txt"])).return_value == "(no matches under '/notes')"

    with pytest.raises(
        Retry, match="Directory 'ghost' does not exist. This tool addresses /notes."
    ):
        await ls("ghost")


def test_symlinks_and_special_files_are_never_followed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "dir").mkdir(parents=True)
    (root / "dir" / "a.md").write_text("a")
    (tmp_path / "secret.md").write_text("secret")
    (root / "link.md").symlink_to(tmp_path / "secret.md")
    (root / "inner.md").symlink_to(root / "dir" / "a.md")
    (root / "via").symlink_to(root / "dir")
    os.mkfifo(root / "pipe")

    assert canonical(root, "dir/../dir/a.md") == (
        "dir/a.md",
        root.resolve() / "dir/a.md",
    )
    assert canonical(root, "link.md") is None
    assert canonical(root, "inner.md") is None
    assert canonical(root, "via/a.md") is None
    assert canonical(root, "new/b.md") is not None

    with open_beneath(root, "dir/a.md") as file:
        assert file.read() == b"a"

    for local, code in (
        ("link.md", errno.ELOOP),
        ("via/a.md", errno.ELOOP),
        ("pipe", errno.EINVAL),
        ("../secret.md", errno.EINVAL),
    ):
        with pytest.raises(OSError) as raised:
            _ = open_beneath(root, local)

        assert raised.value.errno in (code, errno.ENOTDIR)


async def test_listing_never_leaves_the_mount(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "dir").mkdir(parents=True)
    (root / "dir" / "a.md").write_text("a")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.md").write_text("secret")
    (root / "away").symlink_to(tmp_path / "outside")
    (root / "via").symlink_to(root / "dir")
    [mount] = mounts((root, ""))

    assert [local for local, _ in mount.walk(root.resolve(), "*/*.md")] == ["dir/a.md"]

    for escape in ("../outside/*", "/etc/*"):
        assert list(mount.walk(root.resolve(), escape)) == []

        with pytest.raises(Retry, match="must be relative"):
            await Ls(mounts=[mount])(patterns=[escape])
