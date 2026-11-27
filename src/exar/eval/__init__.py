"""Evaluation of agentic applications with pydantic-evals, from single runs to learning over time, needs the `eval` extra.

pydantic-evals is the standard of exar: its evaluators assess runs, its cases carry the gold data,
and its datasets drive learning curves and optimization.

- `assessment`: `evaluate` runs pydantic-evals evaluators on an exar trace into an `Assessment`,
  `Evaluate` is the agent that does so as the revise agent of an `exar.experience.CbrCycle`,
  and `span_tree` and `span_of` convert between traces and pydantic-evals span trees.
- `evaluators`: Native pydantic-evals evaluators for key facts, answerability, and questions to a judge (`Judged`),
  and `exar.cbrkit.RetrievalMetric` for retrieval metrics.
- `sequence`: `evaluate_sequence` evaluates an application that learns on a sequence of test cases into a `LearningCurve`.
- `optimize`: Reflective optimization of the instructions of an agent.
- `export`: Assessed runs as lossless chat datasets for fine-tuning and reinforcement learning.
"""

from .assessment import Assessment, Evaluate, evaluate
from .evaluators import (
    Answerability,
    AnswerabilityJudge,
    Judged,
    JudgeModel,
    KeyFact,
    KeyFacts,
)
from .export import export
from .optimize import optimize
from .sequence import LearningCurve, evaluate_sequence

__all__ = [
    "Answerability",
    "AnswerabilityJudge",
    "Assessment",
    "Evaluate",
    "JudgeModel",
    "Judged",
    "KeyFact",
    "KeyFacts",
    "LearningCurve",
    "evaluate",
    "evaluate_sequence",
    "export",
    "optimize",
]
