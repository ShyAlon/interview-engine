"""Partial-answer drill (Shy, 2026-09-24): one specific follow-up on the area a
partial reply just turned thin, then move on. Off unless INTERVIEW_ENGINE_DRILL=on."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from interview_engine import graph as graph_mod
from interview_engine.categorization import ClassificationAnswer, ProviderResult
from interview_engine.coverage import status_block, initial_coverage
from interview_engine.graph import build_graph
from interview_engine.telemetry import UsageLog
from tests.test_coverage import FakeModel, fake_models, start, structured


class RecordingModel(FakeModel):
    def __init__(self, replies):
        super().__init__(replies)
        self.systems = []

    def invoke(self, messages):
        self.systems.append(getattr(messages[0], "content", ""))
        return super().invoke(messages)


class Cat:
    """relevance comes from a script; covers_<area> fires for the named area."""
    def __init__(self, script):
        self.script = list(script)  # [(relevance, area)]

    def active(self, slot):
        return slot in ("coverage", "intent")

    def classify(self, request):
        if request.slot == "scope":
            return ProviderResult({"accepted": ClassificationAnswer("accepted", 0.99),
                                   "archetype": ClassificationAnswer("saas_crud", 0.99)}, "stub", "stub")
        if request.slot == "coverage":
            rel, area = self.script.pop(0) if self.script else ("dodged", None)
            answers = {"relevance": ClassificationAnswer(rel, 0.9)}
            if area:
                answers[f"covers_{area}"] = ClassificationAnswer(0.9, 0.9)
            return ProviderResult(answers, "stub", "stub")
        return ProviderResult({"intent": ClassificationAnswer("answer", 0.9)}, "stub", "stub")


def _drive(g, thread, answer):
    return g.invoke({"messages": [HumanMessage(content=answer)]}, {"configurable": {"thread_id": thread}})


def _run(monkeypatch, script, answers, drill="on", thread="d1"):
    if drill is None:
        monkeypatch.delenv("INTERVIEW_ENGINE_DRILL", raising=False)
    else:
        monkeypatch.setenv("INTERVIEW_ENGINE_DRILL", drill)
    probing = RecordingModel([structured(f"question {i}?") for i in range(12)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-drill-" + thread), categorizer=Cat(script))
    r = start(g, thread=thread)
    out = [r]
    for a in answers:
        out.append(_drive(g, thread, a))
    return out, probing


def test_drill_is_off_by_default(monkeypatch):
    (r0, r1), probing = _run(monkeypatch, [("partial", "goal")], ["Missed calls, basically."], drill=None, thread="off")
    assert r0["coverage_target"] == "goal"
    assert r1["coverage"]["goal"] == "thin"
    assert r1["coverage_target"] == "users"          # today's behavior: move to the next uncovered area
    assert "FOLLOW-UP (once)" not in probing.systems[-1]
    assert not r1.get("drilled_areas")


def test_partial_drills_once_then_moves_on(monkeypatch):
    (r0, r1, r2), probing = _run(monkeypatch, [("partial", "goal"), ("partial", "goal")],
                                 ["Missed calls, basically.", "Still just the calls."], thread="once")
    assert r1["coverage"]["goal"] == "thin"
    assert r1["coverage_target"] == "goal"           # the follow-up stays on the thin area
    assert r1["drilled_areas"] == ["goal"]
    assert "FOLLOW-UP (once)" in probing.systems[-2] and "goal & core value" in probing.systems[-2]
    assert "NEXT TARGET: goal & core value" in probing.systems[-2]
    # second partial on the same area: no second drill, and no closeout - it moves on, still thin
    assert r2["coverage_target"] == "users"
    assert r2["coverage"]["goal"] == "thin"
    assert "FOLLOW-UP (once)" not in probing.systems[-1]
    assert not r2["delegated_decisions"]


def test_addressed_follow_up_covers_the_area(monkeypatch):
    (r0, r1, r2), _ = _run(monkeypatch, [("partial", "goal"), ("addressed", "goal")],
                           ["Missed calls, basically.", "The front desk loses a morning a week to phone tag."], thread="cov")
    assert r1["coverage_target"] == "goal"
    assert r2["coverage"]["goal"] == "covered"
    assert r2["coverage_target"] == "users"


def test_dodged_and_addressed_do_not_drill(monkeypatch):
    (r0, r1), _ = _run(monkeypatch, [("dodged", None)], ["We love dogs."], thread="dodge")
    assert r1["coverage_target"] == "goal" and not r1["drilled_areas"]   # plain re-ask, not a drill
    (r0, r1), _ = _run(monkeypatch, [("addressed", "goal")], ["Clear answer."], thread="addr")
    assert r1["coverage_target"] == "users" and not r1["drilled_areas"]


def test_drill_cap_per_interview(monkeypatch):
    monkeypatch.setattr(graph_mod, "MAX_DRILLS", 1)
    script = [("partial", "goal"), ("dodged", None), ("partial", "users")]
    (r0, r1, r2, r3), _ = _run(monkeypatch, script, ["a1", "a2", "a3"], thread="cap")
    assert r1["drilled_areas"] == ["goal"]
    assert r3["coverage"]["users"] == "thin"
    assert r3["drilled_areas"] == ["goal"]           # cap reached: users is not drilled
    assert r3["coverage_target"] == "flows"


def test_status_block_target_override():
    cov = initial_coverage(); cov["goal"] = "thin"
    assert "NEXT TARGET: stakeholder context" in status_block(cov)
    assert "NEXT TARGET: goal & core value" in status_block(cov, "goal")
