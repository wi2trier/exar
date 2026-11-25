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
