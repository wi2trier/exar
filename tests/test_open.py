from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from exar.memory import GatedMemory
from exar.policy import EffectPolicy
from exar.spec import Spec
from exar.tools.files import Grep, GrepMatch
from exar.tools.web import HostClient, HostSession, SearchPage, WebQuery, WebSearch
from exar.typing import Changes, Hit


class StubEngine:
    async def count(
        self,
        pattern: str,
        root: Path,
        *,
        glob: str | None,
        case_sensitive: bool,
        literal: bool,
        exclude_dirs: tuple[str, ...],
    ) -> dict[str, int]:
        return {}

    async def search(
        self,
        pattern: str,
        root: Path,
        files: Sequence[str],
        *,
        context_lines: int,
        max_count: int,
        case_sensitive: bool,
        literal: bool,
    ) -> list[GrepMatch]:
        return []


class StubMemory:
    async def search(self, *queries: str, limit: int = 10) -> Sequence[Hit[str, str]]:
        return []

    async def read(self, keys: Collection[str], /) -> Mapping[str, str]:
        return {}

    async def write(
        self, changes: Changes[str, str], /, *, expected: Mapping[str, str] | None = None
    ) -> Changes[str, str]:
        return changes


class StubBackend:
    endpoint = "https://search.example/"

    def params(self, search: WebQuery, max_results: int) -> dict[str, str | None]:
        return {"q": search.query}

    def parse(self, content: bytes) -> SearchPage:
        return SearchPage(hits=())


def test_python_code_opens_the_kinds() -> None:
    engine, memory, backend = StubEngine(), StubMemory(), StubBackend()
    grep = Grep(mounts=[], engine=engine)
    gated = GatedMemory(memory=memory, policy=EffectPolicy())
    search = WebSearch(
        client=HostClient("c"), session=HostSession("s"), backend=backend
    )

    assert (
        grep.engine is engine and gated.memory is memory and search.backend is backend
    )

    defs = Spec.model_json_schema()["$defs"]

    for name, kind in (
        ("AnyGrepEngine", "GrepEngineKind"),
        ("AnySearchBackend", "SearchBackendKind"),
        ("AnyMemory", "MemoryKind"),
    ):
        assert defs[name]["oneOf"] == [{"$ref": f"#/$defs/{kind}"}]


def test_spec_rejects_unknown_kinds() -> None:
    with pytest.raises(ValidationError, match="bogus"):
        Spec.model_validate(
            {"tools": [{"kind": "grep", "mounts": [], "engine": {"kind": "bogus"}}]}
        )
