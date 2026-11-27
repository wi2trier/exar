"""pydantic-ai agents configured natively, as exar agents, and capabilities for their runs.

- `PydanticAiAgent`: A run function of a natively configured pydantic-ai agent as an `exar.typing.Agent`.
- `traced_events`: A streamed run that the host drives, such as one a UI adapter streams, as the root of an exar run.
- `instrumentation`: The instrumentation of pydantic-ai into the traces of exar, applied globally
  with `Agent.instrument_all(instrumentation())`.
- `CountUsage`: Counts the usage of every model request toward the exar run, apart from the spans (`usage_of`).
- `decide`, `judge`: A `exar.typing.Decide` of any output type and a `exar.typing.Judge` of the probability of yes on any model,
  a decision model fast and calibrated, a language model through structured output.
- `Generation`: Structured responses to many prompts on any model, such as for the language models of cbrkit.
- `toolset`: exar tools as a pydantic-ai toolset, with their effects as metadata (`EFFECT`, `tool_effect`).
- `EffectGate`: Derives native approvals from the `exar.policy` policy of each run by the actions of the tools (`definition_action`),
  whose effects `DeclaredEffects` declares for tools built elsewhere.
- `RetrieveExperience`: Gives an agent the hints of past runs of similar tasks (`exar.experience`),
  reported as a read of the tool `retrieve_experience` or by calling it, for the task its `task` projects a prompt onto, such as its text (`prompt_text`).
- `compaction`: `CompactOnOverflow` summarizes a history the context window refused, keeping the prompt cache,
  with `summarize`, `is_compaction`, and `compaction_cut` for hosts that compact on their own.
- `RecordWorkspace`: Records the files each run changed in its workspace as `exar.experience.FileChanges`,
  compared by `snapshot` and `changes`, for the hints `exar.experience.Files`.
- `ToolNote`, `RunNote`: Report the state of a run as tool results or instructions rendered once per run, keeping the prompt cache.
- `guards`: Guards against slips of models and limits of providers, such as `LenientArguments`, `LimitNotes`, and `PromptImageLimit`,
  and `ToolOutputFence`, which frames every tool result as data without authority.

Two shapes in the histories that hosts store are a stable contract, since exar reads them back,
and a change of either comes with a way to read the old one:

- A compaction is a response whose first part is a `CompactionPart` without a provider (`is_compaction`, `compaction_window`).
- A note of `ToolNote` is a response of tool calls followed by a request of their returns,
  found by the name and the arguments of the tool (`latest_read`).

Everything else is configured on the pydantic-ai agent directly, such as delegation to subagents
with the `SubAgents` and `TaskControl` of pydantic-ai-harness, approvals with its `ApprovalReviewer`,
bounded tool returns with its `ToolOutputLimits`,
and declarative agents with `pydantic_ai.Agent.from_file`.
"""

from . import content as content
from .agent import (
    CountUsage,
    Events,
    PydanticAiAgent,
    instrumentation,
    traced_events,
    usage_of,
)
from .decision import decide, judge
from .experience import RetrieveExperience, prompt_text
from .generation import Generation
from .notes import Read, Reads, Render, RunNote, ToolNote, latest_read
from .tools import (
    EFFECT,
    DeclaredEffects,
    EffectGate,
    definition_action,
    tool_effect,
    toolset,
)
from .workspace import RecordWorkspace, Snapshot, changes, snapshot

__all__ = [
    "EFFECT",
    "CountUsage",
    "DeclaredEffects",
    "EffectGate",
    "Events",
    "Generation",
    "PydanticAiAgent",
    "Read",
    "Reads",
    "RecordWorkspace",
    "Render",
    "RetrieveExperience",
    "RunNote",
    "Snapshot",
    "ToolNote",
    "changes",
    "decide",
    "definition_action",
    "instrumentation",
    "judge",
    "latest_read",
    "prompt_text",
    "snapshot",
    "tool_effect",
    "toolset",
    "traced_events",
    "usage_of",
]
