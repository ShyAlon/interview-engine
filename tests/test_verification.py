import json

from interview_engine import question_config as qc
from interview_engine import verification as v
from interview_engine.session_retrieval import build_spans, rank_spans

TURNS = [(0, "a kids music app with songs and star animations"), (2, "We write every song ourselves.")]


def _spans(area="data"):
    return rank_spans(build_spans(TURNS), area)


def test_grounded_question_scores_span_overlap_and_names_the_span():
    r = v.verify_question("Where do the songs come from, and who writes them?", "data", _spans())
    assert r["grounding"]["result"] == "grounded" and r["grounding"]["span"].split(".")[0] in ("t0", "t2")
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


def _graph_run(monkeypatch, tmp_path, replies, tid, cfg=None, coverage_target="data", idea="a kids music app. We write every song ourselves.", history=None):
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
        **({"question_history": history} if history else {}),
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


def test_repeat_check_is_deterministic_one_model_call_and_prints_the_boundary(monkeypatch, tmp_path):
    prior = "Where are the songs kept and who adds new songs?"
    proposed = "Where are the songs kept, and who adds new ones?"
    from interview_engine.coverage import question_similarity
    sim = question_similarity(proposed, prior)
    qc.store().put({"question_repeat": {"similarity_threshold_pct": int(round(sim * 100))}})
    r, rows, model = _graph_run(monkeypatch, tmp_path, [proposed], "vr-boundary", history=[{"area": "data", "question": prior}])
    assert model.calls == 1      # no SAME/NEW judge call
    row = rows[0]
    assert row["repeat_similarity"] == round(sim, 3) and "boundary" in row
    assert row["boundary"].startswith("repeat similarity=") and "matched=" in row["boundary"]


def test_clearly_new_question_has_no_boundary_line(monkeypatch, tmp_path):
    r, rows, _ = _graph_run(monkeypatch, tmp_path, [GOOD], "vr-noboundary",
                            history=[{"area": "users", "question": "Who are the buyers?"}])
    assert "boundary" not in rows[0] and rows[0]["repeat_threshold"] == 0.45
    assert r["messages"][-1].content == GOOD

T3 = "Does the app need to exchange data with any existing system or service (e.g. GitHub, a music library, analytics), or is it fully self-contained?"
T4 = "Does the app need to exchange data with any existing system or service (e.g. GitHub Pages hosting, analytics, audio libraries), or is it fully self-contained with none?"


def test_integrations_closing_question_hits_its_own_area_cues():
    # Shy's session 7315ff94 turn 3: target_cue_score 0, best_other data=2 -> false off_target.
    for q in (T3, T4):
        r = v.verify_question(q, "integrations", [])
        assert r["target_cue_score"] > 0 and "off_target" not in r["failed_checks"], r


def test_integrations_cues_follow_the_none_stated_config():
    cfg = qc.defaults()
    cfg["integrations"]["self_contained_cues"] = ["walled garden"]
    cfg["area_cues"]["integrations"] = ["api"]
    r = v.verify_question("Is this a walled garden with its own songs and music?", "integrations", [], cfg=cfg)
    assert r["target_cue_score"] == 1


def test_data_questions_asked_under_integrations_still_fail_off_target():
    r = v.verify_question("Where are the songs and music files stored, and who adds the content?", "integrations", [])
    assert "off_target" in r["failed_checks"]


def test_hyphenated_cue_matches_either_spelling():
    from interview_engine.session_retrieval import _score
    assert _score("it is self contained", ["self-contained"]) == 1
    assert _score("it is self-contained", ["self contained"]) == 1

def test_spans_from_one_message_get_distinct_uids_and_keep_the_message_id():
    from interview_engine.session_retrieval import build_spans
    spans = build_spans([(0, "We write every song ourselves. Kids play on tablets. Parents see progress.")])
    assert [s["id"] for s in spans] == ["t0", "t0", "t0"]          # the prompt block still quotes [t0]
    assert [s["uid"] for s in spans] == ["t0.0", "t0.1", "t0.2"]


def test_verification_record_lists_unique_span_uids(monkeypatch, tmp_path):
    idea = "a kids music app. We write every song ourselves. Songs are kept in a folder."
    r, rows, _ = _graph_run(monkeypatch, tmp_path, [GOOD], "vr-uid", idea=idea)
    uids = [x["uid"] for x in rows[0]["retrieval_spans"]]
    assert uids and len(uids) == len(set(uids))
    assert all("id" in x for x in rows[0]["retrieval_spans"])


def test_verification_write_failure_is_safe_bounded_warning(tmp_path,monkeypatch,caplog):
    import logging
    log=v.VerificationLog(tmp_path/'verification.jsonl')
    monkeypatch.setattr('builtins.open',lambda *a,**k:(_ for _ in ()).throw(OSError('private-disk-detail')))
    with caplog.at_level(logging.WARNING):
        log.write({'private-project':'text'})
        log.write({'private-project':'text'})
    assert caplog.text.count('review.evidence.degraded')==1 and 'OSError' in caplog.text
    assert 'private-project' not in caplog.text and 'private-disk-detail' not in caplog.text


def test_verification_stop_is_not_swallowed_or_reported_as_sink_failure(monkeypatch,tmp_path):
    from interview_engine.stop_fence import StopRequested
    log=v.VerificationLog(tmp_path/'v.jsonl')
    monkeypatch.setattr('interview_engine.stop_fence.guarded_publish',lambda fn:(_ for _ in ()).throw(StopRequested('stop')))
    with pytest.raises(StopRequested):log.write({})
