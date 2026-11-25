"""Experience of past runs as cases (Meta-CBR): retrieve, reuse, revise, and retain the runs of any agent.

The protocol `exar.typing.Experience` retrieves, retains, and forgets assessed runs (`Retained`),
the full `exar.typing.Trace` with its verdict and evaluations.
`CbrCycle` runs the cycle declaratively over any experience:
it retrieves runs as hints, reuses them in an agent, revises the run by an assessing agent, and retains it.
What a run reuses is domain dependent:

- `Examples`: The outputs as few-shot examples for language model agents.
- `Trajectories`: The trajectories (`Trajectory`), which agents and tools in what order succeeded or failed.
- `Files`: The files the runs changed in their workspace (`FileChanges`), such as their scripts,
  which `exar.pydantic_ai.RecordWorkspace` records.
- `ProcessReuse`: The processes as executable plans (`Process`), run as `exar.bpmn.BpmnProcess`.

`recall` retrieves the runs of the most similar tasks and renders them as hints, as `CbrCycle` and `exar.pydantic_ai.RetrieveExperience` do.

An `exar.typing.Adaptation` fits trajectories and processes to the new task:
`Substitute` replaces agents and tools the deployment lacks by the most similar available ones, deterministically,
and a `cbrkit.adapt.llm.Rewrite` with the instructions `REWRITE` in an `exar.cbrkit.Adapt` lets a language model adapt them.
`exar.pydantic_ai.RetrieveExperience` gives the hints to pydantic-ai agents natively.

With the `cbrkit` extra, `StorageExperience` implements the protocol with the phases of cbrkit over any cbrkit storage:
a case is a split `cbrkit.model.Case` whose problem is the task of a run and whose solution is the `Retained` run,
retrieved by a `cbrkit.retrieve.Retriever` of a measure over tasks, failed runs as negative cases that rank lower,
and retained by a `cbrkit.retain.Retainer`,
so tasks and outputs have to be JSON through pydantic to persist the cases, such as in a `cbrkit.io.TypedStorage`.
`Process` and `ProcessReuse` need the `bpmn` extra, since processes are cbrkit graphs that run as pydantic-graph graphs.
Privacy stays upstream, traces are redacted when they are captured (`exar.agents.run(..., redact=...)`),
and `StorageExperience.forget` and `cbrkit.maintain` delete and maintain cases.

Modules:
- `_retained`: `Retained`.
- `_solutions`: `Trajectory`, `Step`, `FileChanges`, `ChangedFile`, and `CHANGES`.
- `_adapt`: `Substitute` and `REWRITE`.
- `_reuse`: `Examples`, `Trajectories`, `Files`, `CbrCycle`, and `recall`.
- `_cases`: `StorageExperience`, needs the `cbrkit` extra.
- `_process`: `Process` and `ProcessReuse`, needs the `bpmn` extra.
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_exports
from ._adapt import REWRITE, Substitute
from ._retained import Retained
from ._reuse import CbrCycle, Examples, Files, Trajectories, recall
from ._solutions import CHANGES, ChangedFile, FileChanges, Step, Trajectory

if TYPE_CHECKING:
    from ._cases import StorageExperience
    from ._process import Process, ProcessReuse

__all__ = [
    "CHANGES",
    "REWRITE",
    "CbrCycle",
    "ChangedFile",
    "Examples",
    "FileChanges",
    "Files",
    "Process",
    "ProcessReuse",
    "Retained",
    "Step",
    "StorageExperience",
    "Substitute",
    "Trajectories",
    "Trajectory",
    "recall",
]

lazy_exports(__name__)
