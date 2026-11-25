"""What runs reuse from past runs: their trajectories and the files they changed, see `Process` for their processes."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import TypeAdapter

from ..otel import invoked, operation, tool_name
from ..text import tagged
from ..typing import Span, Trace

__all__ = [
    "CHANGES",
    "ChangedFile",
    "FileChanges",
    "Step",
    "Trajectory",
]

CHANGES = "exar.workspace.changes"
"""The attribute of the span of a run under which `exar.pydantic_ai.RecordWorkspace` records its `FileChanges` as JSON."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    """An agent invocation or a tool call of a run.

    Args:
        kind: Whether an agent or a tool.
        name: Its name.
        failed: Whether it ended with an error.
    """

    kind: Literal["agent", "tool"]
    name: str
    failed: bool = False


def _step(span: Span) -> Step | None:
    failed = span.status == "error"

    if name := invoked(span):
        return Step(kind="agent", name=name, failed=failed)

    if operation(span) == "execute_tool" and (name := tool_name(span)):
        return Step(kind="tool", name=name, failed=failed)

    return None


def _steps(span: Span, agent: str | None) -> Iterator[Step]:
    """The steps below a span, leaving out the spans of agent frameworks inside the invocation of the same agent."""
    for child in span.children:
        step = _step(child)

        if step is not None and not (step.kind == "agent" and step.name == agent):
            yield step

        yield from _steps(
            child, step.name if step is not None and step.kind == "agent" else agent
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Trajectory:
    """The trajectory of a run: which agents and tools it used in what order, and which failed.

    Trajectories of past runs are hints for language model agents, see `Trajectories`,
    and sequences of their names are descriptions for sequence measures such as `cbrkit.sim.collections.SmithWaterman`.

    Args:
        steps: The agent invocations and tool calls below the root of the run, depth first in order.

    Examples:
        >>> from exar.agents import FunctionAgent, RunConfig, call, run
        >>> from exar.policy import allow_all
        >>> async def lead(task: str) -> str:
        ...     return await call(FunctionAgent(func=str.upper), task, name="writer")
        >>> trajectory = Trajectory.of(run(lead, "hi", RunConfig(policy=allow_all)))
        >>> print(trajectory.rename({"writer": "author"}).render())
        1. agent author
    """

    steps: tuple[Step, ...] = ()

    @classmethod
    def of(cls, trace: Trace[Any, Any], /) -> Trajectory:
        """The trajectory of a run."""
        return cls(steps=tuple(_steps(trace.root, invoked(trace.root))))

    @property
    def references(self) -> frozenset[str]:
        """Names of the agents and tools of the steps."""
        return frozenset(step.name for step in self.steps)

    def rename(self, names: Mapping[str, str], /) -> Trajectory:
        """The trajectory with its steps renamed by their old names."""
        return Trajectory(
            steps=tuple(
                replace(step, name=names.get(step.name, step.name))
                for step in self.steps
            )
        )

    def render(self) -> str:
        """The steps as a numbered list."""
        return "\n".join(
            f"{number}. {step.kind} {step.name}{', failed' if step.failed else ''}"
            for number, step in enumerate(self.steps, 1)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChangedFile:
    """A file a run added, updated, or removed in its workspace.

    Args:
        path: Its path relative to the working directory of the workspace.
        change: Whether the run added, updated, or removed it.
        size: Its size in bytes after the run, 0 once removed.
        digest: The SHA-256 of its content after the run, empty once removed.
        text: Its content, None if removed, binary, or beyond the budget of the recording.
    """

    path: str
    change: Literal["added", "updated", "removed"]
    size: int = 0
    digest: str = ""
    text: str | None = None

    def render(self) -> str:
        """The file for a language model, its content in a tag or a line naming what is left out."""
        if self.text is not None:
            return tagged("file", self.text, path=self.path, change=self.change)

        if self.change == "removed":
            return f"{self.path} (removed)"

        return f"{self.path} ({self.change}, {self.size} bytes, content not recorded)"


@dataclass(frozen=True, slots=True, kw_only=True)
class FileChanges:
    """The files a run added, updated, or removed in its workspace, such as the scripts it wrote and ran.

    Unlike the arguments of its tool calls, they include what the commands of the run wrote.
    `exar.pydantic_ai.RecordWorkspace` records them on the span of the run under `CHANGES`.

    Args:
        files: The changed files by path.

    Examples:
        >>> changes = FileChanges(files=(ChangedFile(path="sum.py", change="added", size=9, text="print(4)\\n"),))
        >>> FileChanges.load(changes.dump()) == changes
        True
        >>> print(changes.render())
        <file path="sum.py" change="added">
        print(4)
        </file>
    """

    files: tuple[ChangedFile, ...] = ()

    @classmethod
    def of(cls, trace: Trace[Any, Any], /) -> FileChanges | None:
        """The changes recorded on the outermost span of a run that has them, None without any."""
        for span in trace.walk():
            if isinstance(value := span.attributes.get(CHANGES), str):
                return cls.load(value)

        return None

    @classmethod
    def load(cls, value: str | bytes, /) -> FileChanges:
        """The changes from their JSON, see `dump`."""
        return _CHANGES.validate_json(value)

    def dump(self) -> str:
        """The changes as JSON."""
        return _CHANGES.dump_json(self).decode()

    def render(self) -> str:
        """The changed files for a language model, see `ChangedFile.render`."""
        return "\n".join(file.render() for file in self.files)


_CHANGES = TypeAdapter(FileChanges)
