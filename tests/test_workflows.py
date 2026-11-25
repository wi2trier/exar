from collections.abc import Sequence

import pytest

from exar import otel
from exar.agents import FunctionAgent, Limits, RunConfig, call, run
from exar.errors import Refused
from exar.policy import EffectPolicy, Rules, allow_all
from exar.tools import effect
from exar.typing import Action, Verdict
from exar.workflows import (
    Attempt,
    Chain,
    EvaluateOptimize,
    Grounded,
    Parallel,
    Rag,
    Route,
)


def classify(x: int) -> str:
    return "even" if x % 2 == 0 else "odd"


@effect("read")
def half(x: int) -> int:
    return x // 2


def triple(x: int) -> int:
    return 3 * x + 1


def total(xs: Sequence[int]) -> int:
    return sum(xs)


def improve(draft: Attempt[None, int] | None) -> int:
    return draft.output + 1 if draft is not None else 1


def judge(attempt: Attempt[None, int]) -> Verdict:
    return Verdict(score=attempt.output / 10, passed=attempt.output >= 3)


def search(query: str) -> list[str]:
    return [f"doc about {query}"]


def answer(grounded: Grounded[str, list[str]]) -> str:
    return f"{grounded['query']}: {grounded['context'][0]}"


HALF, TRIPLE = FunctionAgent(func=half), FunctionAgent(func=triple)


def test_workflows() -> None:
    route = Route(
        classifier=FunctionAgent(func=classify), routes={"even": HALF, "odd": TRIPLE}
    )

    assert [run(route, x, RunConfig(policy=allow_all)).output for x in (4, 3)] == [
        2,
        10,
    ]
    assert (
        run(
            Chain(agents=[Parallel(agents=[HALF, TRIPLE]), FunctionAgent(func=total)]),
            4,
            RunConfig(policy=allow_all),
        ).output
        == 15
    )
    assert (
        run(
            EvaluateOptimize(
                generator=FunctionAgent(func=improve),
                evaluator=FunctionAgent(func=judge),
                max_rounds=5,
            ),
            None,
            RunConfig(policy=allow_all),
        ).output
        == 3
    )
    assert (
        run(
            Rag(
                retrieve=FunctionAgent(func=search), generate=FunctionAgent(func=answer)
            ),
            "cbr",
            RunConfig(policy=allow_all),
        ).output
        == "cbr: doc about cbr"
    )


def test_nested_trace_policy_and_limits() -> None:
    pipeline = Chain(agents=[HALF, Chain(agents=[TRIPLE])])
    trace = run(pipeline, 8, RunConfig(policy=allow_all))
    [first, nested] = trace.root.children

    assert trace.output == 13
    assert otel.operation(trace.root) == otel.operation(nested) == "invoke_workflow"
    assert (otel.agent_name(first), otel.workflow_name(nested), otel.agents(trace.root)) == (
        "half",
        "Chain",
        ("Chain", "half", "triple"),
    )

    with pytest.raises(Refused, match="maximum depth"):
        run(pipeline, 8, RunConfig(policy=allow_all, limits=Limits(max_depth=1)))

    with pytest.raises(Refused, match="denied agent 'triple'"):
        run(pipeline, 8, RunConfig(policy=Rules(by_name={"triple": "ask"})))

    asked: list[Action] = []

    async def approve(action: Action) -> bool:
        asked.append(action)

        return True

    # Undeclared effects fail closed: the policy asks about `triple`, but not about `half` or the workflows.
    assert (
        run(
            pipeline, 8, RunConfig(policy=EffectPolicy(execute="ask"), approve=approve)
        ).output
        == 13
    )
    assert [(a.name, a.effect, a.depth) for a in asked] == [("triple", "execute", 2)]


async def test_an_invocation_outside_a_run_decides_nothing_below_it() -> None:
    leaf = FunctionAgent(func=len)

    assert await call(leaf, "abc") == 3

    with pytest.raises(RuntimeError, match="outside a run"):
        await call(Chain(agents=[leaf]), "abc")


def test_parallel_raises_the_first_failure() -> None:
    def boom(x: int) -> int:
        raise KeyError(x)

    with pytest.raises(KeyError):
        run(
            Parallel(agents=[HALF, FunctionAgent(func=boom)]),
            4,
            RunConfig(policy=allow_all),
        )


def test_evaluate_optimize_needs_a_round() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        EvaluateOptimize(
            generator=FunctionAgent(func=improve),
            evaluator=FunctionAgent(func=judge),
            max_rounds=0,
        )
