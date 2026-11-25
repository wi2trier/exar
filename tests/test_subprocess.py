"""Unit tests for async subprocess wrappers."""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from exar.subprocess import (
    CommandFailed,
    OutputTooLarge,
    ProcessLimits,
    RgLine,
    RgMatch,
    SubprocessError,
    jq_filter,
    require,
    rg_count,
    rg_search,
    run,
)

needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="rg is not installed")
needs_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")


class TestRun:
    """Tests for the base ``run`` function."""

    async def test_captures_stdout(self) -> None:
        result = await run(["echo", "hello"])
        assert result.stdout_text.strip() == "hello"
        assert result.returncode == 0

    async def test_captures_stderr(self) -> None:
        result = await run(
            ["sh", "-c", "echo oops >&2; exit 0"],
        )
        assert "oops" in result.stderr_text

    async def test_raises_on_nonzero(self) -> None:
        with pytest.raises(SubprocessError):
            await run(["sh", "-c", "exit 42"])

    async def test_allowed_returncodes(self) -> None:
        result = await run(
            ["sh", "-c", "exit 1"],
            allowed_returncodes=(1,),
        )
        assert result.returncode == 1

    async def test_stdin_pipe(self) -> None:
        result = await run(["cat"], stdin=b"piped input")
        assert result.stdout_text == "piped input"

    async def test_stdout_json(self) -> None:
        result = await run(["echo", '{"key": "value"}'])
        parsed = result.stdout_json(dict)
        assert parsed == {"key": "value"}

    async def test_stdout_ndjson(self) -> None:
        result = await run(
            ["sh", "-c", "echo '{\"a\":1}'\necho '{\"b\":2}'"],
        )
        items = list(result.stdout_ndjson())
        assert items == [{"a": 1}, {"b": 2}]


async def _search(pattern: str, root: Path, **kwargs: Any) -> list[RgMatch]:
    return await rg_search(pattern, root, [p.name for p in root.iterdir()], **kwargs)


@needs_rg
class TestRgSearch:
    """Tests for ``rg_search``."""

    async def test_finds_matches(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("Hello World\nGoodbye World\n")
        matches = await _search("Hello", tmp_path)
        assert len(matches) == 1
        assert matches[0].lines[0].line_number == 1
        assert matches[0].lines[0].is_match
        assert "Hello" in matches[0].lines[0].text

    async def test_no_matches_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("nothing here\n")
        matches = await _search("zzz_missing", tmp_path)
        assert matches == []

    async def test_pattern_starting_with_dash(self, tmp_path: Path) -> None:
        (tmp_path / "options.txt").write_text("--version\n")

        matches = await _search("--version", tmp_path)

        assert len(matches) == 1
        assert matches[0].lines[0].text == "--version"

    async def test_glob_filter(self, tmp_path: Path) -> None:
        (tmp_path / "include.md").write_text("target\n")
        (tmp_path / "exclude.txt").write_text("target\n")
        assert await rg_count("target", tmp_path, glob="*.md") == {"include.md": 1}

    async def test_decodes_legacy_encoded_lines(self, tmp_path: Path) -> None:
        # ripgrep hands back non-UTF-8 lines base64-encoded under a `bytes`
        # key instead of `text`, in CRLF sources including the terminator.
        (tmp_path / "legacy.txt").write_bytes("Leitfähigkeit\r\n".encode("cp1252"))
        matches = await _search("Leitf", tmp_path)
        assert matches[0].lines == (
            RgLine(line_number=1, text="Leitfähigkeit", is_match=True),
        )

    async def test_context_lines(self, tmp_path: Path) -> None:
        (tmp_path / "test.txt").write_text("aaa\nbbb\näb ccc\nddd\neee\n")
        matches = await _search("ccc", tmp_path, context_lines=1)
        assert len(matches) == 1
        # The offset counts characters, not the two bytes UTF-8 spends on ä.
        assert matches[0].lines == (
            RgLine(line_number=2, text="bbb", is_match=False),
            RgLine(line_number=3, text="äb ccc", is_match=True, match_start=3),
            RgLine(line_number=4, text="ddd", is_match=False),
        )


@needs_jq
class TestJqFilter:
    """Tests for ``jq_filter``."""

    async def test_identity(self) -> None:
        result = await jq_filter(".", '{"key": "value"}')
        assert result == [{"key": "value"}]

    async def test_field_access(self) -> None:
        result = await jq_filter(".name", '{"name": "Alice", "age": 30}')
        assert result == ["Alice"]

    async def test_array_iteration(self) -> None:
        result = await jq_filter(".[].x", '[{"x": 1}, {"x": 2}]')
        assert result == [1, 2]

    async def test_invalid_expression_raises(self) -> None:
        with pytest.raises(CommandFailed, match="jq failed"):
            await jq_filter("invalid [[[", '{"x": 1}')

    async def test_malformed_document_raises(self) -> None:
        with pytest.raises(CommandFailed, match="jq failed"):
            await jq_filter(".", "{not json")

    async def test_empty_input(self) -> None:
        result = await jq_filter(".", "[]")
        assert result == [[]]

    @pytest.mark.parametrize(
        "filter_expr",
        ['import "x" as $d {search: "/"}; $d', '# c\ninclude "x"; .'],
    )
    async def test_module_directive_refused(self, filter_expr: str) -> None:
        with pytest.raises(CommandFailed, match="import and include"):
            await jq_filter(filter_expr, "{}")

    async def test_keywords_as_fields_pass(self) -> None:
        result = await jq_filter('.include, ."import"', '{"include": 1, "import": 2}')
        assert result == [1, 2]

    async def test_server_environment_hidden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXAR_TEST_SECRET", "secret")
        [keys] = await jq_filter("$ENV | keys", "{}")
        assert isinstance(keys, list)
        assert "PATH" in keys
        assert set(keys) <= {"PATH", "SHLVL"}


def test_require() -> None:
    assert require("sh")

    with pytest.raises(FileNotFoundError, match="not installed"):
        require("exar-missing-binary")


class TestProcessLimits:
    """Tests for the resource caps of ``run``."""

    async def test_timeout_kills_child(self) -> None:
        with pytest.raises(TimeoutError):
            await run(["sleep", "5"], limits=ProcessLimits(timeout=0.2))

    async def test_timeout_kills_the_group_of_an_exited_child(
        self, tmp_path: Path
    ) -> None:
        # The child exits at once, while its own child holds standard output open.
        script = f"sleep 30 & echo $! > {tmp_path}/pid"

        with pytest.raises(TimeoutError):
            await run(["sh", "-c", script], limits=ProcessLimits(timeout=0.5))

        await asyncio.sleep(0.2)

        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / "pid").read_text()), 0)

    async def test_cpu_limit_kills_child(self) -> None:
        with pytest.raises(SubprocessError):
            await run(["sh", "-c", "while :; do :; done"], limits=ProcessLimits(cpu=1))

    async def test_output_limit_kills_child(self) -> None:
        with pytest.raises(OutputTooLarge, match="exceeds 1000 bytes"):
            await run(["yes"], limits=ProcessLimits(output=1000))

    async def test_file_size_limit_stops_writes(self, tmp_path: Path) -> None:
        big = tmp_path / "big"

        with pytest.raises(SubprocessError):
            await run(
                ["sh", "-c", f"head -c 1000000 /dev/zero > {big}"],
                limits=ProcessLimits(files=4096),
            )

        # The bash 3.2 of macOS reads the POSIX blocks of 512 bytes as KiB.
        assert big.stat().st_size <= 2 * 4096

    @needs_rg
    async def test_rg_count_path_with_newline(self, tmp_path: Path) -> None:
        (tmp_path / "a\nb\r.txt").write_text("x\n")

        assert await rg_count("x", tmp_path) == {"a\nb\r.txt": 1}
