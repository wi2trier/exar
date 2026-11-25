"""The files a run changed in its pydantic-ai workspace, as `exar.experience.FileChanges` on the span of the run.

A workspace is any backend of `pydantic_ai.workspaces`, such as a local folder or the `DockerSandbox`
of pydantic-ai-harness, so the changes include what commands wrote, which no tool call shows.
`snapshot` and `changes` compare the files of a workspace before and after through its file API alone,
by their SHA-256 since workspaces report no modification times,
and `RecordWorkspace` records the changes of every run of an agent.
"""

import hashlib
import logging
import posixpath
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace
from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.workspaces import Workspace

from exar.experience import CHANGES, ChangedFile, FileChanges

__all__ = ["RecordWorkspace", "Snapshot", "changes", "snapshot"]

_logger = logging.getLogger(__name__)

type Snapshot = Mapping[str, tuple[int, str]]
"""The size and SHA-256 of each file of a workspace by its path relative to the working directory."""


def _skipped(name: str) -> bool:
    return name.startswith(".") or name == "__pycache__"


async def snapshot(workspace: Workspace, /) -> Snapshot:
    """The files below the working directory of a workspace, leaving out hidden entries and `__pycache__`.

    A directory reached a second time through a symlink is listed once.
    """
    root = await workspace.working_dir()
    files: dict[str, tuple[int, str]] = {}
    pending = [root]
    seen: set[str] = set()

    while pending:
        directory = pending.pop()

        if (real := await workspace.realpath(directory)) in seen:
            continue

        seen.add(real)

        for entry in await workspace.list_dir(directory):
            if _skipped(entry.name):
                continue

            if entry.is_dir:
                pending.append(entry.path)
                continue

            data = await workspace.read_bytes(entry.path)
            files[posixpath.relpath(entry.path, root)] = (
                len(data),
                hashlib.sha256(data).hexdigest(),
            )

    return files


async def changes(
    before: Snapshot,
    workspace: Workspace,
    /,
    *,
    max_file_bytes: int = 100_000,
    max_total_bytes: int = 1_000_000,
) -> FileChanges:
    """The files added, updated, or removed in a workspace since a snapshot, by path.

    Args:
        before: The snapshot of the workspace before.
        workspace: The workspace now.
        max_file_bytes: The largest file whose text is kept.
        max_total_bytes: The text kept of all files together, beyond which files keep their size and digest only.
    """
    root = await workspace.working_dir()
    after = await snapshot(workspace)
    budget = max_total_bytes
    files: list[ChangedFile] = []

    for path in sorted(before.keys() | after.keys()):
        if path not in after:
            files.append(ChangedFile(path=path, change="removed"))
            continue

        if before.get(path) == after[path]:
            continue

        size, digest = after[path]
        text = None

        if size <= min(max_file_bytes, budget):
            with suppress(UnicodeDecodeError):
                text = (await workspace.read_bytes(posixpath.join(root, path))).decode()
                budget -= size

        files.append(
            ChangedFile(
                path=path,
                change="updated" if path in before else "added",
                size=size,
                digest=digest,
                text=text,
            )
        )

    return FileChanges(files=tuple(files))


@dataclass(kw_only=True)
class RecordWorkspace[D](AbstractCapability[D]):
    """Records the files each run changed in its workspace on the span of the run under `exar.experience.CHANGES`.

    The span is the one current when the run starts, the `invoke_agent` span of an instrumented agent,
    so `exar.experience.FileChanges.of` reads the changes from the trace and `exar.experience.Files` renders them as hints.
    A run without a workspace records nothing, and one whose workspace became unreadable, such as a crashed sandbox,
    logs a warning instead, so recording never changes the outcome of a run.
    Add it to the agent that owns the workspace only, since subagents that share it would record the same files again.

    Args:
        max_file_bytes: The largest file whose text is kept.
        max_total_bytes: The text kept of all files of a run together.
    """

    max_file_bytes: int = 100_000
    max_total_bytes: int = 1_000_000

    async def wrap_run(
        self, ctx: RunContext[D], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """Compares the files of the workspace before and after the run, which may fail."""
        workspace = ctx.workspace

        if not workspace.attached:
            return await handler()

        before = await snapshot(workspace)
        span = trace.get_current_span()

        try:
            return await handler()
        finally:
            try:
                recorded = await changes(
                    before,
                    workspace,
                    max_file_bytes=self.max_file_bytes,
                    max_total_bytes=self.max_total_bytes,
                )
            except Exception:
                _logger.warning(
                    "The changes of the workspace were not recorded", exc_info=True
                )
            else:
                span.set_attribute(CHANGES, recorded.dump())
