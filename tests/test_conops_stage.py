import json

from langchain_core.messages import AIMessage, HumanMessage
from vmodel_contracts import ConOpsDocument

from interview_engine.conops_stage import (comparison_question, finalize,
                                            decision_reply, validate_draft)


def draft():
    return {"stakeholder_context": [{"stakeholder": "runner", "role": "tracks habits",
             "goals": ["quick capture"], "operating_environment": "phone, often offline",
             "source_turns": [0]}],
            "operational_scenarios": [{"scenario_id": "OS-1", "name": "Log habit",
             "trigger": "habit completed", "actors": ["runner"],
             "operational_flow": ["open product", "record points"],
             "success_outcome": "points sync", "exceptions": ["offline"], "source_turns": [0]}],
            "concept_options": [{"name": "Native app", "summary": "installed app",
             "advantages": ["native APIs"], "disadvantages": ["store release"]},
             {"name": "PWA", "summary": "installable web product",
             "advantages": ["one deployment"], "disadvantages": ["platform limits"]}]}


def test_draft_provenance_and_explicit_selection():
    d = validate_draft(draft(), {0, 2})
    assert "Concept Decision" in comparison_question(d)
    assert decision_reply("1 - native APIs are required", d["concept_options"])[0].name == "Native app"
    assert decision_reply("PWA - one deployment", d["concept_options"])[0].name == "PWA"
    assert decision_reply("1", d["concept_options"]) is None
    assert decision_reply("I lean native - maybe", d["concept_options"]) is None
    doc = finalize(d, "s", "2 - one deployment is enough", 5)
    assert doc.decision.rationale == "one deployment is enough"
    assert ConOpsDocument.model_validate(doc).decision.source_turns == [5]


def test_model_cannot_fabricate_source_turn():
    d = draft(); d["operational_scenarios"][0]["source_turns"] = [99]
    try: validate_draft(d, {0})
    except ValueError as e: assert "non-founder" in str(e)
    else: assert False


def test_conops_graph_is_resumable_and_model_runs_once(monkeypatch):
    from interview_engine import model_config
    from interview_engine.graph import build_graph
    from interview_engine.telemetry import UsageLog
    from tests.test_coverage import FakeModel, NEEDS, SCOPE
    conops = FakeModel([json.dumps(draft())])
    models = {"probing": FakeModel(["[INTERVIEW COMPLETE]"]),
              "distillation": FakeModel([json.dumps(NEEDS), json.dumps(SCOPE), "[]"]),
              "conops": conops,
              "scope_tiebreak": FakeModel(['{"is_web_app": true, "archetype": "saas_crud", "confidence": 0.9, "reason": "web app"}'])}
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: models[stage])
    g = build_graph(UsageLog("s"), conops_enabled=True)
    cfg = {"configurable": {"thread_id": "conops"}}
    r = g.invoke({"messages": [HumanMessage(content="phone habit app, often offline")],
                  "submission": {"free_text": "a web-enabled phone habit app with logins",
                                 "founder_id": "f", "submission_id": "s"}}, cfg)
    assert r["conops_draft"] and not r.get("conops")
    assert "Concept Decision" in r["messages"][-1].content
    r = g.invoke({"messages": [HumanMessage(content="1 - reliable native APIs")]}, cfg)
    assert r["conops"]["decision"]["selected_concept"] == "Native app"
    assert conops.calls == 1
