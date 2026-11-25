import subprocess
import sys

OPTIONAL = frozenset(
    {
        "PIL",
        "a2a",
        "cbrkit",
        "claude_agent_sdk",
        "fastmcp",
        "pydantic_ai",
        "pydantic_evals",
        "pydantic_graph",
        "simpleeval",
    }
)


def test_core_imports_no_extras() -> None:
    code = "import exar, sys; print(*sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    modules = set(output.split())

    assert not {name.partition(".")[0] for name in modules} & OPTIONAL
    assert not {"exar.eval", "exar.ingest"} & modules


def test_spec_needs_extras_only_for_the_tools_it_lists() -> None:
    code = """
import sys

import pydantic_ai

for name in ("sqlalchemy", "httpx2", "trafilatura", "markdownify", "duckdb", "polars", "PIL", "cv2"):
    sys.modules[name] = None

from exar.errors import MissingExtra

from exar.spec import Spec, build

assert len(build(Spec.model_validate({"tools": [{"kind": "ls", "mounts": []}]}), {}).tools) == 1

for tool in ({"kind": "sql", "engine": "db"}, {"kind": "table", "mounts": []}, {"kind": "view", "mounts": []}):
    try:
        build(Spec.model_validate({"tools": [tool]}), {})
    except MissingExtra as error:
        print(error)
"""
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout

    assert all(f"exar[{extra}]" in output for extra in ("sql", "table", "media"))


def test_processes_name_the_bpmn_extra() -> None:
    code = """
import sys

sys.modules["cbrkit"] = None

import exar

try:
    exar.experience.ProcessReuse
except AttributeError as error:
    print(error)
"""
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout

    assert "exar[bpmn]" in output
