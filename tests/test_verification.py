import json

from interview_engine import question_config as qc
from interview_engine import verification as v
from interview_engine.session_retrieval import build_spans, rank_spans

TURNS = [(0, "a kids music app with songs and star animations"), (2, "We write every song ourselves.")]


def _spans(area="data"):
    return rank_spans(build_spans(TURNS), area)


def test_grounded_question_scores_span_overlap_and_names_the_span():
    r = v.verify_question("Where do the songs come from, and who writes them?", "data", _spans())
    assert r["grounding"]["result"] == "grounded" and r["grounding"]["span"] in ("t0", "t2")
    assert r["grounding"]["score"] >= 1 and r["passed"] and r["failed_checks"] == []


def test_question_that_shares_nothing_with_the_session_fails_grounding():
    r = v.verify_question("Which payment currencies matter most?", "data", _spans())
    assert r["grounding"]["result"] == "no_overlap" and "grounding" in r["failed_checks"]


def test_no_span_is_neutral_not_a_failure():
    r = v.verify_question("What information does the product need to keep?", "data", [])
    assert r["grounding"]["result"] == "no_span" and r["passed"]


def test_overlap_minimum_comes_from_config():
    cfg = qc.validate({"verification": {"grounding_overlap_min": 3}})
    r = v.verify_question("Where do songs come from?", "data", _spans(), [], cfg)
    assert r["grounding"]["result"] == "no_overlap"


def test_implementation_level_and_founder_role_questions_fail():
    r = v.verify_question("What framework should the songs use?", "data", _spans())
    assert r["implementation_hit"] and "implementation_level" in r["failed_checks"]
    r = v.verify_question("Who is responsible for creating the songs and maintaining them?", "data", _spans())
    assert r["founder_role_hit"] and "founder_role" in r["failed_checks"]


def test_question_clearly_about_another_area_is_off_target():
    r = v.verify_question("How fast and reliable must it be offline when it can crash?", "glossary", [])
    assert "off_target" in r["failed_checks"] and r["best_other_area"] in ("nfrs", "failures")
    ok = v.verify_question("Which terms mean something specific, what do you call them?", "glossary", [])
    assert "off_target" not in ok["failed_checks"]


def test_log_appends_one_json_line_per_record(tmp_path):
    log = v.VerificationLog(tmp_path / "x" / "verification.jsonl")
    log.write({"turn": 1, "passed": True})
    log.write({"turn": 2, "passed": False})
    rows = [json.loads(l) for l in (tmp_path / "x" / "verification.jsonl").read_text().splitlines()]
    assert [r["turn"] for r in rows] == [1, 2] and all("ts" in r for r in rows)


import pytest


@pytest.fixture(autouse=True)
def _isolate_store():
    old = qc.store()
    qc.set_store(qc.ConfigStore())
    yield
    qc.set_store(old)


def _graph_run(monkeypatch, tmp_path, replies, tid, cfg=None, coverage_target="data", idea="a kids music app. We write every song ourselves."):
    from langchain_core.messages import HumanMessage
    from interview_engine.coverage import AREA_KEYS
    from interview_engine.graph import build_graph
    from interview_engine.telemetry import UsageLog
    from tests.test_coverage import FakeModel, fake_models, structured
    if cfg:
        qc.store().put(cfg)
    cov = {k: "covered" for k in AREA_KEYS}; cov[coverage_target] = "uncovered"
    model = FakeModel([structured(q) for q in replies])
    fake_models(monkeypatch, model)
    log = v.VerificationLog(tmp_path / "verification.jsonl")
    r = build_graph(UsageLog(tid), verification_log=log).invoke({
        "messages": [HumanMessage(content=idea)],
        "submission": {"free_text": idea, "founder_id": "t", "submission_id": tid},
        "verdict": {"accepted": True}, "coverage": cov, "coverage_target": coverage_target,
    }, {"configurable": {"thread_id": tid}})
    rows = [json.loads(l) for l in (tmp_path / "verification.jsonl").read_text().splitlines()]
    return r, rows, model


def test_scores_are_written_every_turn_even_when_the_question_is_bad(monkeypatch, tmp_path):
    import pytest
    qc.set_store(qc.ConfigStore())
    r, rows, _ = _graph_run(monkeypatch, tmp_path, ["Which payment currencies matter most?"], "vr-score")
    assert len(rows) == 1
    row = rows[0]
    assert row["target_area"] == "data" and row["passed"] is False and "grounding" in row["failed_checks"]
    assert row["retry_count"] == 0 and row["fallback_used"] is False and row["non_answer"] is False
    assert row["retrieval_spans"] and row["proposed_question"] == "Which payment currencies matter most?"
    # scoring alone never rewrites a question
    assert r["messages"][-1].content == "Which payment currencies matter most?"


BAD = "Which payment currencies matter most?"
GOOD = "You said you write every song ourselves - where are the songs kept, and who adds new ones?"
RETRY_ON = {"verification": {"retry_enabled": True}}


def test_retry_is_off_by_default_so_one_probing_call_is_made(monkeypatch, tmp_path):
    _, rows, _ = _graph_run(monkeypatch, tmp_path, [BAD, GOOD], "vr-off")
    assert rows[0]["retry_count"] == 0 and rows[0]["fallback_used"] is False
    assert rows[0]["final_question"] == BAD


def test_retry_on_uses_the_retried_question_when_it_passes(monkeypatch, tmp_path):
    r, rows, _ = _graph_run(monkeypatch, tmp_path, [BAD, GOOD], "vr-pass", cfg=RETRY_ON)
    assert r["messages"][-1].content == GOOD
    row = rows[0]
    assert row["retry_count"] == 1 and row["fallback_used"] is False and row["passed"] is True
    assert row["retry_cleared"] == ["grounding"] and row["proposed_question"] == GOOD


def test_retry_on_falls_back_to_the_template_when_the_retry_also_fails(monkeypatch, tmp_path):
    from interview_engine.coverage import _FALLBACK_QUESTIONS
    r, rows, _ = _graph_run(monkeypatch, tmp_path, [BAD, "Which tax rates apply?"], "vr-fb", cfg=RETRY_ON)
    assert r["messages"][-1].content == _FALLBACK_QUESTIONS["data"]
    assert rows[0]["retry_count"] == 1 and rows[0]["fallback_used"] is True


def test_passing_question_is_never_retried_even_when_the_flag_is_on(monkeypatch, tmp_path):
    r, rows, _ = _graph_run(monkeypatch, tmp_path, [GOOD], "vr-nop", cfg=RETRY_ON)
    assert rows[0]["retry_count"] == 0 and r["messages"][-1].content == GOOD
