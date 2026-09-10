"""Run-5 polish batch regressions: invented-detail guards, glossary capture,
operational-concepts rendering, contradiction-flag false positive, NFR closure,
config-driven pricing, per-stage telemetry."""
import json

import pytest

from interview_engine import model_config
from interview_engine.coverage import nfr_quantified
from interview_engine.distill_qa import grounded_in, is_solution_language, names_technology
from interview_engine.graph import build_graph
from interview_engine.telemetry import UsageLog, price_for
from test_coverage import FakeModel, NEEDS, SCOPE, start


# --- 3a: invented-detail guards ---

def test_named_technology_is_solution_language():
    assert is_solution_language("passwords encrypted with AES-256 at rest")
    assert is_solution_language("auth via OAuth2 with JWT tokens")
    assert names_technology("data queued through Kafka")
    assert not is_solution_language("a seeded disagreement produces exactly one alert")


def test_ungrounded_criterion_detected():
    turns = ["a saas web app for surf schools with booking and logins",
             "surfers book lessons and log in"]
    assert not grounded_in("homepage shows a portfolio of completed projects and case studies", turns)
    assert grounded_in("a surfer cancels a booking and the slot reopens", turns)


# --- 3b: glossary capture ---

def test_glossary_list_shape_coerced(monkeypatch):
    scope = dict(SCOPE, glossary=[{"term": "TMS", "meaning": "transport mgmt system"},
                                  {"term": "exceptions", "meaning": "plan deviations"}])
    distills = [json.dumps(NEEDS), json.dumps(scope), "[]"]
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: FakeModel(["[INTERVIEW COMPLETE]"]) if stage == "probing"
                        else FakeModel(distills) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="gl1")
    assert r["intent_model"]["glossary"] == {"TMS": "transport mgmt system",
                                             "exceptions": "plan deviations"}


def test_glossary_from_needs_phase_merged(monkeypatch):
    needs = dict(NEEDS, glossary={"TMS": "transport mgmt system"})
    distills = [json.dumps(needs), json.dumps(SCOPE), "[]"]
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: FakeModel(["[INTERVIEW COMPLETE]"]) if stage == "probing"
                        else FakeModel(distills) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="gl2")
    assert r["intent_model"]["glossary"] == {"TMS": "transport mgmt system"}


# --- 3c: operational concepts drift ---

def test_done_criteria_dicts_render_as_statements(monkeypatch):
    needs = dict(NEEDS, done_criteria=[{"statement": "A developer can build from the spec",
                                       "source_turns": [20]}])
    distills = [json.dumps(needs), json.dumps(SCOPE), "[]"]
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: FakeModel(["[INTERVIEW COMPLETE]"]) if stage == "probing"
                        else FakeModel(distills) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="done1")
    assert r["intent_model"]["done_criteria"] == ["A developer can build from the spec"]


def test_operational_concepts_dicts_render_as_statements(monkeypatch):
    needs = dict(NEEDS, operational_concepts=[{"statement": "support team handles feed failures",
                                               "source_turns": [18]}])
    distills = [json.dumps(needs), json.dumps(SCOPE), "[]"]
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: FakeModel(["[INTERVIEW COMPLETE]"]) if stage == "probing"
                        else FakeModel(distills) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="op1")
    assert r["intent_model"]["operational_concepts"] == ["support team handles feed failures"]


# --- 3e: NFR closure needs numbers ---

def test_nfr_quantified_requires_numbers_or_explicit_waiver():
    assert nfr_quantified("we expect 5000 concurrent users and 99.9% uptime")
    assert nfr_quantified("no numbers for the demo")
    assert nfr_quantified("just a demo")
    assert not nfr_quantified("it must be very secure and reliable")
    assert not nfr_quantified("strict security, simple UI")


def test_model_cannot_close_nfrs_without_numbers(monkeypatch):
    from interview_engine.coverage import initial_coverage, next_area
    turn = json.dumps({"question": "q", "covered": ["nfrs"], "thin": [], "skipped": []})
    probing = FakeModel([turn, "[INTERVIEW COMPLETE]"])
    distills = [json.dumps(NEEDS), json.dumps(SCOPE), "[]"]
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: probing if stage == "probing"
                        else FakeModel(distills) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    g = build_graph(UsageLog("t1"))
    cfg = {"configurable": {"thread_id": "nfr1"}}
    from langchain_core.messages import HumanMessage
    idea = "a saas web app for surf schools with booking and logins"
    r = g.invoke({"messages": [HumanMessage(content=idea)],
                  "submission": {"free_text": idea, "founder_id": "t", "submission_id": "nfr1"}}, cfg)
    assert r["coverage"]["nfrs"] == "thin", "claimed NFR closure without numbers must not stick"


# --- 3d: contradiction flag ---

def test_generic_product_word_is_not_a_contradiction():
    from interview_engine.cli import render_intent_markdown
    from vmodel_contracts import IntentModel
    intent = IntentModel.model_validate({
        "session_id": "x", "problem_statement": "a supply chain optimization application",
        "target_users": ["managers"], "core_value_proposition": "AI-driven predictions",
        "in_scope": ["predictions"], "out_of_scope": ["mission-critical application tiers"],
        "stakeholder_needs": [{"need_id": "SN-1", "statement": "s", "kind": "functional",
                               "source_turns": [0]}],
        "source_turns": {"problem_statement": [0]}})
    md = render_intent_markdown(intent)
    assert "⚠" not in md


def test_real_overlap_still_flagged():
    from interview_engine.cli import render_intent_markdown
    from vmodel_contracts import IntentModel
    intent = IntentModel.model_validate({
        "session_id": "x", "problem_statement": "a supply chain optimization application",
        "target_users": ["managers"], "core_value_proposition": "AI-driven predictions",
        "in_scope": ["predictions"], "out_of_scope": ["AI-driven predictions for orders"],
        "stakeholder_needs": [{"need_id": "SN-1", "statement": "s", "kind": "functional",
                               "source_turns": [0]}],
        "source_turns": {"problem_statement": [0]}})
    assert "⚠" in render_intent_markdown(intent)


# --- 2: pricing config + telemetry rollup ---

def test_price_table_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "prices.json"
    cfg.write_text(json.dumps({"deepseek-chat": [1.0, 2.0], "new-model": [0.1, 0.2]}))
    monkeypatch.setenv("VMODEL_PRICE_TABLE", str(cfg))
    assert price_for("deepseek-chat") == (1.0, 2.0)
    assert price_for("new-model") == (0.1, 0.2)
    with pytest.raises(KeyError):
        price_for("still-unpriced-model")


def test_ollama_always_free():
    assert price_for("qwen2.5-coder:14b", "ollama") == (0.0, 0.0)


def test_usage_summary_rollup(tmp_path):
    log = UsageLog("s1")
    log.record("probing", "ollama", "qwen2.5-coder:14b",
               {"input_tokens": 100, "output_tokens": 50}, duration_s=2.0)
    log.record("probing", "ollama", "qwen2.5-coder:14b",
               {"input_tokens": 200, "output_tokens": 60}, duration_s=3.0)
    log.record("distillation", "ollama", "qwen2.5-coder:14b",
               {"input_tokens": 500, "output_tokens": 100}, duration_s=10.0)
    s = log.summary()
    assert s["stages"]["probing"]["calls"] == 2
    assert s["stages"]["probing"]["prompt_tokens"] == 300
    assert s["stages"]["probing"]["seconds"] == 5.0
    assert s["stages"]["distillation"]["seconds"] == 10.0
    assert s["total_seconds"] == 15.0
    p = tmp_path / "usage_summary.json"
    log.write_summary(p)
    assert json.loads(p.read_text())["stages"]["probing"]["calls"] == 2

