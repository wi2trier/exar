"""Experience-grounded agentic reasoning for any agent framework.

exar provides the protocols and building blocks of agentic applications after the EXAR architecture:
memories, reasoning agents of any framework, orchestrators from workflows to BPMN processes,
OpenTelemetry traces of every run, and case-based experience that improves future runs.

Every module is imported on first access, so importing one module, such as `exar.ingest.executors`, never loads the others.

Core modules, depending only on pydantic and OpenTelemetry, with `typing` and `errors` on the standard library only:

- `typing`: The protocols `Agent`, `Assessed`, `Policy`, `Memory`, and `Experience`,
  and the records they exchange: traces of OpenTelemetry spans, actions, verdicts, and the hits and changes of memories.
- `otel`: Capturing the OpenTelemetry spans of runs, the typed GenAI accessors of their spans, and the usage of their model calls,
  composing with the tracer provider of the host.
- `agents`: Running agents on tasks under a run config of a policy and limits, recording every nested invocation into a trace.
- `errors`: Refused actions, conflicting writes, missing extras, and failed agents.
- `helpers`: Immutable mappings, names, bounded concurrency, and durable writes.
- `schema`: Serialized forms and JSON Schemas of configurable classes, described by their docstrings.
- `tools`: Conventions for tools as plain async functions and the formatting of their output,
  and with `exar.tools.files` (needs the `pydantic-ai` extra) configurable tools over the files below mounts.
- `subprocess`: The subprocess runner with process limits and typed async wrappers of the command line tools `rg` and `jq`.
- `policy`: Policies that allow, refuse, or ask for approval of actions.
- `memory`: Memories over entries in process and files, with tools for language models.
- `workflows`: Workflow recipes such as chaining, routing, and the 4R cycle.

Integrations and extras, so `import exar` never loads an optional dependency
and a missing extra surfaces as an `AttributeError` that names it:

- `exar.cbrkit`: Case-based reasoning by cbrkit, memories over its storages and its retrieval metrics, needs the `cbrkit` extra.
- `exar.bpmn`: BPMN processes as agents, over the process graphs of cbrkit run by pydantic-graph, needs the `bpmn` extra.
- `exar.ingest`: Parsing, chunking, and syncing documents, each backend needs the extra of its name.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from ._lazy import extra_hint
from .errors import MissingExtra

if TYPE_CHECKING:
    from . import (
        agents,
        bpmn,
        cbrkit,
        errors,
        helpers,
        ingest,
        memory,
        otel,
        policy,
        schema,
        subprocess,
        tools,
        typing,
        workflows,
    )

__all__ = [
    "agents",
    "bpmn",
    "cbrkit",
    "errors",
    "helpers",
    "ingest",
    "memory",
    "otel",
    "policy",
    "schema",
    "subprocess",
    "tools",
    "typing",
    "workflows",
]

_EXTRAS = {
    "bpmn": "bpmn",
    "cbrkit": "cbrkit",
    "ingest": "ingest",
}
"""The extra that every integration needs, by module name."""


def __getattr__(name: str) -> ModuleType:
    """Imports a module on first access.

    Raises:
        AttributeError: If no module has the name or the extra of an integration is missing,
            caused by a `exar.errors.MissingExtra`.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        with extra_hint(f"exar.{name}", _EXTRAS.get(name)):
            return import_module(f".{name}", __name__)
    except MissingExtra as error:
        raise AttributeError(str(error)) from error
