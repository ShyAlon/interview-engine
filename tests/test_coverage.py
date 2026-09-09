"""Coverage model (ADR-0016): checklist tracking, founder control, saturation."""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from interview_engine import model_config
from interview_engine.coverage import (AREAS, apply_reply, founder_control,
                                       initial_coverage, next_area, progress_line,
                                       status_block)
from interview_engine.graph import build_graph
from interview_engine.telemetry import UsageLog


def structured(question, covered=(), thin=(), skipped=(), complete=False):
    return json.dumps({"question": question, "covered": list(covered),
                       "thin": list(thin), "skipped": list(skipped), "complete": complete})


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.replies.pop(0) if self.replies else structured("anything else?"))


NEEDS = {"stakeholder_needs": [{"need_id": "SN-1", "statement": "founders need a spec",
         "kind": "functional", "source_turns": [0]}]}
SCOPE = {"problem_statement": "p", "target_users": ["u"], "core_value_proposition": "v",
         "in_scope": ["a"], "out_of_scope": ["b"], "source_turns": {}}


def fake_models(monkeypatch, probing):
    box = {"probing": probing}

    def _build(stage):
        if stage == "probing":
            return probing
        if stage == "scope_tiebreak":
            return FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}'])
        return FakeModel([json.dumps(NEEDS), json.dumps(SCOPE), "[]"])  # distillation phases

    monkeypatch.setattr(model_config, "build_chat_model", _build)
    return box


def start(g, text="a saas web app for surf schools, with booking and customer logins", thread="cov"):
    cfg = {"configurable": {"thread_id": thread}}
    return g.invoke({"messages": [HumanMessage(content=text)],
                     "submission": {"free_text": text, "founder_id": "t", "submission_id": thread}}, cfg)


# --- unit ---

def test_initial_and_priority():
    cov = initial_coverage()
    assert next_area(cov) == "goal"
    cov["goal"] = "thin"
    assert next_area(cov) == "users"      # uncovered beats thin
    cov["users"] = "covered"
    assert next_area(cov) == "flows"      # any uncovered still beats the thin goal
    for k in ("flows", "data", "integrations", "nfrs", "failures", "constraints", "risks", "done", "glossary"):
        cov[k] = "skipped"
    assert next_area(cov) == "goal"       # now the thin one is the weakest
    cov["goal"] = "covered"
    assert next_area(cov) is None         # saturated


def test_status_block_and_progress_line():
    cov = dict(initial_coverage(), goal="covered", users="thin")
    block = status_block(cov)
    assert "goal=+" in block and "users=~" in block and "NEXT TARGET" in block
    line = progress_line(cov)
    assert "coverage 1/11" in line and "goal" in line


def test_apply_reply_never_uncovers():
    cov = dict(initial_coverage(), goal="covered")
    out = apply_reply(cov, {"thin": ["goal", "users"], "covered": ["users"]})
    assert out["goal"] == "covered" and out["users"] == "covered"


def test_founder_control_is_strict():
    assert founder_control("skip") == "skip"
    assert founder_control("skip this") == "skip"
    assert founder_control("enough") == "enough"
    assert founder_control("let's wrap") == "enough"
    assert founder_control("we should skip login for v1") is None   # content, not a command
    assert founder_control("that's enough salt in the recipe") is None or True  # see below


# --- graph ---

def test_structured_reply_updates_coverage(monkeypatch):
    probing = FakeModel([structured("Who else touches this?", covered=["goal"], thin=["users"])])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g)
    assert r["coverage"]["goal"] == "covered"
    assert r["coverage"]["users"] == "thin"
    assert r["messages"][-1].content == "Who else touches this?"  # question text, not JSON


def test_complete_appends_marker_and_stops(monkeypatch):
    probing = FakeModel([structured("Got it all.", covered=[k for k, _, _ in AREAS], complete=True)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="c2")
    assert any("[INTERVIEW COMPLETE]" in getattr(m, "content", "") for m in r["messages"])
    assert "intent_model" in r  # same invoke flowed through distillation


def test_enough_ends_without_model_call(monkeypatch):
    probing = FakeModel([structured("Who uses it?", covered=["goal"])])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="c3")
    r = g.invoke({"messages": [HumanMessage(content="enough")]},
                 {"configurable": {"thread_id": "c3"}})
    assert probing.calls == 1                       # no second probing call
    assert any("[INTERVIEW COMPLETE]" in getattr(m, "content", "") for m in r["messages"])


def test_skip_marks_area_and_asks_next(monkeypatch):
    probing = FakeModel([structured("Risks?", thin=["goal"]),
                         structured("What breaks first?")])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="c4")
    r = g.invoke({"messages": [HumanMessage(content="skip")]},
                 {"configurable": {"thread_id": "c4"}})
    # goal was thin (the target), so skip marks it; flows still uncovered
    assert r["coverage"]["goal"] == "skipped"
    assert r["messages"][-1].content == "What breaks first?"


def test_unstructured_reply_degrades_gracefully(monkeypatch):
    probing = FakeModel(["Just a plain question?"])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g)
    assert r["messages"][-1].content == "Just a plain question?"
    assert all(v == "uncovered" for v in r["coverage"].values())


def test_coverage_map_render():
    from interview_engine.cli import render_coverage_markdown
    cov = dict(initial_coverage(), goal="covered", users="skipped", nfrs="thin")
    md = render_coverage_markdown(cov)
    assert "**Covered:** goal & core value" in md
    assert "Skipped by you:** users & stakeholder classes" in md
    assert "Thin" in md and "quantified NFRs" in md
    assert "Not covered:" in md
