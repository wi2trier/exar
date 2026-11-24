"""Experience-grounded agentic reasoning for any agent framework.

exar provides the protocols and building blocks of agentic applications after the EXAR architecture:
memories, reasoning agents of any framework, orchestrators from workflows to BPMN processes,
OpenTelemetry traces of every run, and case-based experience that improves future runs.

Every module is imported on first access, so importing one module, such as `exar.ingest.executors`, never loads the others.

Core modules, depending only on pydantic and OpenTelemetry, with `typing` and `errors` on the standard library only:

- `typing`: The protocols `Agent`, `Assessed`, `Policy`, `Memory`, and `Experience`,
  and the records they exchange: traces of OpenTelemetry spans, actions, verdicts, and the hits and changes of memories.
- `errors`: Refused actions, conflicting writes, missing extras, and failed agents.
- `helpers`: Immutable mappings, names, bounded concurrency, and durable writes.
- `schema`: Serialized forms and JSON Schemas of configurable classes, described by their docstrings.

Integrations and extras, so `import exar` never loads an optional dependency
and a missing extra surfaces as an `AttributeError` that names it:

"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from ._lazy import extra_hint
from .errors import MissingExtra

if TYPE_CHECKING:
    from . import (
        errors,
        helpers,
        schema,
        typing,
    )

__all__ = [
    "errors",
    "helpers",
    "schema",
    "typing",
]

_EXTRAS = {
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
