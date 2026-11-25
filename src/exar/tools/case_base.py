"""The operations of a cbrkit system as tools, configured by the sidecar of its case base.

`CaseBase` references the specification of cbrkit by file: `cbrkit.system.System.load` reads the case base with its sidecar,
and every operation the sidecar enables, such as `retrieve`, `reuse`, or `retain`, becomes one tool,
typed with the models of the cases (`cbrkit.system.operations`).
Operations that write the case base have the effect `"write"`, so policies ask before they run.
It needs the `cbrkit` extra only when the tools are built.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .._lazy import extra_hint
from ..schema import serializable
from ..typing import Tool
from . import as_tool

__all__ = ["CaseBase"]


@serializable
@dataclass(frozen=True, slots=True, kw_only=True)
class CaseBase:
    """A cbrkit case base on disk whose operations are tools.

    Args:
        casebase: The case base file or folder, a relative path resolves against the working directory.
        config: The sidecar, `<casebase>.cbrkit.json` if None.
        prefix: Prepended to the names of the operations, so that several case bases tell their tools apart.
    """

    kind: Literal["case_base"] = field(default="case_base", repr=False)
    casebase: Path
    config: Path | None = None
    prefix: str = ""

    def tools(self) -> list[Tool]:
        """One tool per operation that the sidecar enables.

        Raises:
            exar.errors.MissingExtra: If the `cbrkit` extra is missing.
        """
        with extra_hint("CaseBase", "cbrkit"):
            from cbrkit.system import System
            from cbrkit.system.operations import operations

        system = System.load(self.casebase, self.config)

        return [
            as_tool(
                operation,
                name=f"{self.prefix}{operation.__name__}",
                effect="read" if operation.read_only else "write",
            )
            for operation in operations(system)
        ]
