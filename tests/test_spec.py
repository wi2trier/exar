import json
from dataclasses import dataclass
from pathlib import Path

import httpx2
import pytest
import sqlalchemy as sa

from exar.agents import FunctionAgent, run_async
from exar.errors import Refused
from exar.helpers import get_name
from exar.memory import FileMemory, GatedMemory
from exar.policy import Rules
from exar.spec import Spec, build, load
from exar.tools import effect, effect_of
from exar.tools.sql import Sql
from exar.tools.web import WebFetch, WebSession

DATA = Path(__file__).parent / "data"


@effect("read")
def scorer(state: object) -> dict[str, float]:
    return {"score": 0.9}


@effect("read")
def publisher(state: object) -> str:
    return "published"


def rejecter(state: object) -> str:
    return "rejected"


AGENTS = {
    name: FunctionAgent(func=func)
    for name, func in {
        "scorer": scorer,
        "publisher": publisher,
        "rejecter": rejecter,
    }.items()
}


@dataclass(frozen=True)
class Review:
    score: float = 0.0
    accept: str = ""


def spec_file(tmp_path: Path) -> Path:
    mount = {"target": "/notes", "source": str(tmp_path)}
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "processes": {
                    "review": {
                        "file": str(DATA / "exclusive.bpmn"),
                        "outputs": {"accept": "accept"},
                    }
                },
                "workflows": {
                    "pipeline": {"kind": "chain", "agents": ["review"]},
                    "guarded": {"kind": "chain", "agents": ["rejecter"]},
                },
                "memories": {
                    "notes": {
                        "kind": "gated",
                        "memory": {
                            "kind": "file",
                            "mounts": [{**mount, "write": "allow"}],
                        },
                        "policy": {"kind": "effect", "write": "deny"},
                    }
                },
                "tools": [
                    {"kind": "grep", "mounts": [mount], "engine": {"kind": "python"}},
                    {
                        "kind": "sql",
                        "engine": "db",
                        "prefix": "crm_",
                        "tables": ["orders"],
                    },
                    {"kind": "web_fetch", "client": "web", "session": "run"},
                ],
                "policy": {"kind": "rules", "by_name": {"rejecter": "deny"}},
                "limits": {"max_depth": 3},
            }
        )
    )

    return path


async def test_build(tmp_path: Path) -> None:
    engine = sa.create_engine("sqlite://")
    client = httpx2.AsyncClient()
    session = WebSession()
    spec = load(
        spec_file(tmp_path),
        engines={"db": lambda: engine},
        clients={"web": lambda: client},
        sessions={"run": lambda: session},
    )
    _, query, fetch = spec.tools

    assert isinstance(query, Sql) and query.engine() is engine
    assert (
        isinstance(fetch, WebFetch)
        and fetch.client() is client
        and fetch.session() is session
    )
    assert Spec.model_validate(spec.model_dump()) == spec

    app = build(spec, AGENTS)
    notes = app.memories["notes"]

    assert isinstance(notes, GatedMemory) and isinstance(notes.memory, FileMemory)
    assert [get_name(tool) for tool in app.tools] == [
        "grep",
        "crm_tables",
        "crm_describe",
        "crm_select",
        "web_fetch",
    ]
    assert app.config.policy == Rules(by_name={"rejecter": "deny"})

    trace = await run_async(app.workflows["pipeline"], Review(), app.config)

    assert trace.output == Review(score=0.9, accept="published")

    with pytest.raises(Refused, match="rejecter"):
        await run_async(app.workflows["guarded"], Review(), app.config)


def test_unknown_name() -> None:
    spec = Spec.model_validate(
        {
            "workflows": {
                "flow": {"kind": "route", "classifier": "a", "routes": {"x": "b"}}
            }
        }
    )

    with pytest.raises(ValueError, match="unknown names: a, b"):
        build(spec, {})


def test_cycle() -> None:
    spec = Spec.model_validate(
        {
            "workflows": {
                "a": {"kind": "chain", "agents": ["b"]},
                "b": {"kind": "chain", "agents": ["a"]},
            }
        }
    )

    with pytest.raises(ValueError, match="form a cycle"):
        build(spec, {})


async def test_case_base(tmp_path: Path) -> None:
    from cbrkit.io import Source, companion
    from cbrkit.spec import infer, write
    from cbrkit.system import Outcome

    cases = {"a": {"task": "plan a trip"}, "b": {"task": "write a report"}}
    casebase = tmp_path / "cases.json"
    casebase.write_text(json.dumps(cases))
    write(companion(casebase, "sidecar"), infer(cases, Source(format="json")))
    spec = Spec.model_validate(
        {
            "$schema": "https://example.org/exar.json",
            "tools": [
                {"kind": "case_base", "casebase": str(casebase), "prefix": "cases_"}
            ],
        }
    )

    tools = build(spec, {}).tools

    assert Spec.model_validate_json(spec.model_dump_json()) == spec
    assert [(get_name(tool), effect_of(tool)) for tool in tools] == [
        ("cases_retrieve", "read"),
        ("cases_reuse", "read"),
        ("cases_retain", "write"),
        ("cases_cycle", "write"),
    ]

    outcome = await tools[0](query={"task": "write a report"}, limit=1)

    assert isinstance(outcome, Outcome) and [hit.key for hit in outcome.hits] == ["b"]
