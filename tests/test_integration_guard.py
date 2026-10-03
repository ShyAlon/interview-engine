import pytest
from langchain_core.messages import HumanMessage

from interview_engine import integration_guard as ig
from interview_engine.coverage import AREA_KEYS, NONE_STATED, next_area, progress_line
from interview_engine.graph import build_graph
from interview_engine.rendering import render_coverage_markdown
from interview_engine.telemetry import UsageLog
from tests.test_coverage import FakeModel, fake_models, structured


@pytest.mark.parametrize("turns", [
    ["a kids music app with songs and star animations"],
    ["We write every song ourselves. Kids play them and get stars."],
])
def test_no_named_system_closes_none_stated(turns):
    assert ig.close_as_none_stated(turns)


@pytest.mark.parametrize("turns", [
    ["it should pull songs from Spotify"],
    ["login through accounts.google.com"],
    ["we expose an API to schools"],
    ["we want to sync with the school calendar"],
])
def test_named_system_or_verb_keeps_the_question(turns):
    assert not ig.close_as_none_stated(turns)


@pytest.mark.parametrize("cue", ["it works offline", "no external services", "fully self-contained"])
def test_self_contained_signal_closes_even_with_a_proper_noun(cue):
    assert ig.close_as_none_stated([f"A tool for Maria's shop, {cue}"])


def test_guard_can_be_turned_off_and_cues_are_config():
    cfg = {"integrations": {"none_stated_enabled": False, "external_system_verbs": ["pipe"],
                            "self_contained_cues": ["standalone"]}}
    assert not ig.close_as_none_stated(["plain app"], cfg)
    cfg["integrations"]["none_stated_enabled"] = True
    assert ig.names_external_system(["we pipe it out"], cfg)
    assert ig.says_self_contained(["standalone"], cfg)


def _run(monkeypatch, idea, tid):
    cov = {k: "covered" for k in AREA_KEYS}
    cov["integrations"] = "uncovered"
    cov["nfrs"] = "uncovered"
    fake_models(monkeypatch, FakeModel([structured("How fast should it feel to a user?")]))
    graph = build_graph(UsageLog(tid))
    return graph.invoke({
        "messages": [HumanMessage(content=idea)],
        "submission": {"free_text": idea, "founder_id": "t", "submission_id": tid},
        "verdict": {"accepted": True}, "coverage": cov, "coverage_target": "integrations",
    }, {"configurable": {"thread_id": tid}})


def test_graph_closes_integrations_without_asking_and_moves_on(monkeypatch):
    r = _run(monkeypatch, "a kids music app with songs and star animations", "ig-none")
    assert r["coverage"]["integrations"] == NONE_STATED
    assert next_area(r["coverage"]) == "nfrs"
    md = render_coverage_markdown(r["coverage"])
    assert "None stated" in md and "integrations" in md.lower()
    assert "**Covered:**" not in md or "integrations" not in md.split("**Covered:**")[1].split("\n")[0].lower()
    assert "integrations" in progress_line(r["coverage"])


def test_graph_keeps_the_question_when_a_system_is_named(monkeypatch):
    r = _run(monkeypatch, "a kids music app that plays songs from Spotify", "ig-named")
    assert r["coverage"]["integrations"] != NONE_STATED


def test_sn_external_integrations_evidence_counts_as_a_named_system():
    turns = ["a kids music app with songs and star animations"]
    assert ig.close_as_none_stated(turns)
    assert not ig.close_as_none_stated(turns, intent_model={"external_integrations": ["Spotify Web API"]})
    assert not ig.close_as_none_stated(turns, intent_model={"external_integrations": [],
        "stakeholder_needs": [{"statement": "Parents sign in with Google accounts"}]})
    assert ig.close_as_none_stated(turns, intent_model={"external_integrations": [" "],
        "stakeholder_needs": [{"statement": "Kids hear songs and earn stars"}]})


def test_self_contained_founder_signal_still_wins_over_sn_evidence():
    assert ig.close_as_none_stated(["it works fully offline"], intent_model={"external_integrations": ["Spotify"]})
