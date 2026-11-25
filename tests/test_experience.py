from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from cbrkit.adapt.llm import Rewrite
from cbrkit.indexable import InMemory
from cbrkit.io import FileStorage, Source, TypedStorage
from cbrkit.maintain import Maintainer
from cbrkit.model import Case, replan
from cbrkit.model.graph import bpmn_document, from_bpmn, parse_bpmn
from cbrkit.sim import Transpose
from cbrkit.sim.llm import LanguageModel
from cbrkit.sim.strings import WordCoverage
from pydantic import TypeAdapter

from exar import otel
from exar.agents import FunctionAgent, RunConfig, call, run_async
from exar.bpmn import BpmnProcess, substitute_agents
from exar.cbrkit import Adapt
from exar.errors import AgentFailed, Refused
from exar.experience import (
    REWRITE,
    CbrCycle,
    Process,
    ProcessReuse,
    Retained,
    StorageExperience,
    Substitute,
    Trajectories,
    Trajectory,
)
from exar.policy import allow_all
from exar.typing import Adaptation, Trace, Verdict

DATA = Path(__file__).parent / "data"
WORDS = WordCoverage(case_sensitive=False)
LAID_OUT = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
    xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
    xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
    xmlns:camunda="http://camunda.org/schema/1.0/bpmn" id="d">
  <process id="p">
    <subProcess id="sub">
      <task id="t" name="writer" camunda:assignee="ann">
        <extensionElements><camunda:properties/></extensionElements>
      </task>
    </subProcess>
  </process>
  <bpmndi:BPMNDiagram id="diagram">
    <bpmndi:BPMNPlane id="plane" bpmnElement="p">
      <bpmndi:BPMNShape id="shape" bpmnElement="t">
        <dc:Bounds x="1" y="2" width="3" height="4"/>
      </bpmndi:BPMNShape>
    </bpmndi:BPMNPlane>
  </bpmndi:BPMNDiagram>
</definitions>
"""


def contains[I, O](text: str):
    async def revise(trace: Trace[I, O]) -> Retained[I, O]:
        passed = text in str(trace.output)
        verdict = Verdict(score=float(passed), passed=passed)

        return Retained(trace=trace, verdict=verdict)

    return revise


async def test_storage_experience(tmp_path: Path) -> None:
    storage = TypedStorage[str, Case[str, Retained[str, str | None]]](
        storage=FileStorage(
            path=tmp_path / "cases", source=Source(format="json", folder=True)
        ),
        adapter=TypeAdapter(Case[str, Retained[str, str | None]]),
    )
    cases = StorageExperience(storage=storage, measure=WORDS)
    revise = contains("4")

    for task, output in [
        ("Add 2 and 2", "4"),
        ("Add 2 and 3", "6"),
        ("Add 2 and 2", "4"),
    ]:
        trace = await run_async(
            FunctionAgent(func=lambda _, o=output: o), task, RunConfig(policy=allow_all)
        )
        assert len((await cases.retain([await revise(trace)])).items) == 1

    # The failed run matches best but ranks last.
    retrieved = await cases.retrieve("Add 2 and 3", limit=3)

    assert [hit.value.trace.output for hit in retrieved] == ["4", "4", "6"]

    removed = await replan(Maintainer(measure=WORDS, duplicate=0.99), storage)

    assert [change.rule for change in removed.items] == ["redundancy"]

    forgotten = await cases.forget(
        lambda run: not run.verdict.passed, reason="Requested"
    )

    assert len(forgotten.items) == 1 and len(await storage.get_index()) == 1


async def test_cbr_cycle_with_trajectories() -> None:
    cases = StorageExperience[int, str, str](storage=InMemory(), measure=WORDS)
    seen: list[str] = []

    async def lead(task: str) -> str:
        seen.append(task)
        await call(FunctionAgent(func=str.upper), task, name="writer")

        return "4"

    cycle = CbrCycle[str, str](
        experience=cases,
        reuse=lead,
        revise=contains("4"),
        hints=Trajectories(
            adaptation=Substitute(available={"author": "writer of texts"})
        ),
        augment=lambda task, hints: f"{hints}\n\n{task}",
    )

    assert (
        await run_async(cycle, "Add 2 and 2", RunConfig(policy=allow_all))
    ).output == "4"
    assert (
        await run_async(cycle, "Add 2 and 2 again", RunConfig(policy=allow_all))
    ).output == "4"
    assert seen[0] == "Add 2 and 2"
    assert seen[1].endswith("Add 2 and 2 again") and "1. agent author" in seen[1]

    [run] = [hit.value for hit in await cases.retrieve("Add", limit=1)]

    assert run.trace.task == "Add 2 and 2"
    assert Trajectory.of(run.trace).references == {"writer"}


async def test_cbr_cycle_retains_failed_run() -> None:
    cases = StorageExperience[int, str, str](storage=InMemory(), measure=WORDS)

    async def broken(task: str) -> str:
        raise ValueError("boom")

    cycle = CbrCycle[str, str](
        experience=cases, reuse=broken, revise=contains("never revised")
    )

    with pytest.raises(AgentFailed) as failure:
        await run_async(cycle, "Add 2 and 2", RunConfig(policy=allow_all))

    assert isinstance(failure.value.__cause__, ValueError)

    [hit] = await cases.retrieve("Add 2 and 2")

    assert hit.value.trace.task == "Add 2 and 2" and hit.value.trace.output is None
    assert hit.value.verdict == Verdict(score=0.0, passed=False)
    assert otel.error_type(hit.value.trace.root) == "ValueError"


async def test_cbr_cycle_does_not_retain_refusal() -> None:
    cases = StorageExperience[int, str, str](storage=InMemory(), measure=WORDS)

    async def deep(task: str) -> str:
        return await call(FunctionAgent(func=str.upper), task)

    cycle = CbrCycle[str, str](experience=cases, reuse=deep, revise=contains("A"))

    with pytest.raises(Refused):
        await run_async(
            cycle,
            "a",
            RunConfig(policy=lambda action: "deny" if action.depth > 1 else "allow"),
        )

    assert await cases.retrieve("a") == []


@dataclass(frozen=True)
class Review:
    draft: str
    score: float = 0.0
    publish: str = ""
    verdict: str = ""


async def test_process_reuse() -> None:
    process = BpmnProcess[Review](
        graph=from_bpmn(parse_bpmn((DATA / "exclusive.bpmn").read_text())),
        agents={
            "scorer": FunctionAgent(func=lambda state: {"score": 0.9}),
            "publisher": FunctionAgent(func=lambda state: {"publish": "old"}),
            "rejecter": FunctionAgent(func=lambda state: "rejected"),
        },
        outputs={"reject": "verdict"},
    )
    cases = StorageExperience[int, Review, Review](
        storage=InMemory(),
        measure=Transpose(measure=WORDS, conversion=lambda state: state.draft),
    )
    # The deployment replaced the publisher by the printer.
    agents = {
        "scorer": FunctionAgent(func=lambda state: {"score": 0.9}),
        "printer": FunctionAgent(func=lambda state: {"publish": "printed"}),
        "rejecter": FunctionAgent(func=lambda state: "rejected"),
    }

    # The template runs the retrieved processes over the agents of the deployment.
    template = BpmnProcess[Review](
        graph=substitute_agents(process.graph, {"publisher": "printer"}),
        agents=agents,
        outputs={"reject": "verdict"},
    )

    def cycle(
        adaptation: Adaptation[Review, Process] | None = None,
    ) -> CbrCycle[Review, Review]:
        return CbrCycle(
            experience=cases,
            reuse=ProcessReuse(
                experience=cases,
                process=template,
                fallback=process,
                adaptation=adaptation,
            ),
            revise=contains("publish='"),
        )

    # Without a case the hand-written fallback runs and its process is retained.
    assert (
        await run_async(cycle(), Review(draft="A draft"), RunConfig(policy=allow_all))
    ).output.publish == "old"

    [hit] = await cases.retrieve(Review(draft="A draft"))
    recorded = Process.of(hit.value.trace)

    assert recorded is not None and "publisher" in recorded.references

    # The unadapted process is invalid for the deployment, so the fallback runs again.
    assert (
        await run_async(cycle(), Review(draft="A draft"), RunConfig(policy=allow_all))
    ).output.publish == "old"

    substitute = Substitute(
        available={"printer": "publisher of drafts", "scorer": "", "rejecter": ""}
    )

    assert (
        await run_async(
            cycle(substitute),
            Review(draft="Another draft"),
            RunConfig(policy=allow_all),
        )
    ).output.publish == "printed"

    async def model[T](prompts: Sequence[str], response_type: type[T]) -> list[T]:
        assert response_type is Process and "A third draft" in prompts[0]

        return [response_type(bpmn=recorded.rename({"publisher": "printer"}).bpmn)]

    rewrite = Rewrite[Case[Review, Process]](
        model=LanguageModel("planner", model),
        instructions=REWRITE,
        context={"available": {"printer": "Publishes drafts"}},
    )

    assert (
        await run_async(
            cycle(Adapt(func=rewrite)),
            Review(draft="A third draft"),
            RunConfig(policy=allow_all),
        )
    ).output.publish == "printed"


def test_process_rename_keeps_document() -> None:
    process = Process(bpmn=LAID_OUT)
    renamed = process.rename({"writer": "author"})

    assert bpmn_document(process.rename({}).graph) == bpmn_document(process.graph)
    assert renamed.references == {"author"}
    assert renamed.graph.nodes["t"].value == {
        **process.graph.nodes["t"].value,
        "agent": "author",
    }
    assert all(part in renamed.bpmn for part in ("BPMNShape", "assignee", "properties"))
