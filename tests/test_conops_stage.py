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
    d=draft()
    conops = FakeModel([json.dumps({"operating_environment":"phone, often offline"}),
                        json.dumps({"trigger":"habit completed"}),
                        json.dumps({"operational_flow":["open product","record points"]}),
                        json.dumps([{"name":o["name"],"summary":o["summary"]} for o in d["concept_options"]]),
                        json.dumps({"advantages":["native APIs"]}),json.dumps({"disadvantages":["store release"]}),
                        json.dumps({"advantages":["one deployment"]}),json.dumps({"disadvantages":["platform limits"]})])
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
    assert conops.calls == 8


def test_preseed_uses_validated_intent_identity_and_provenance():
    from interview_engine.conops_stage import preseed_from_intent
    intent={"target_users":["Supply chain manager"],
            "stakeholder_needs":[{"statement":"Supply chain managers need early warnings","source_turns":[10,12]}]}
    assert preseed_from_intent(intent,{0,10,12})=={"stakeholder_context":[{
        "stakeholder":"Supply chain manager","source_turns":[10,12]}]}


def test_derivation_fills_stakeholder_fields_from_validated_intent():
    from interview_engine.conops_stage import derive_stakeholder_contexts
    intent={"target_users":["Supply chain manager"],
            "stakeholder_needs":[{"statement":"Supply chain managers need early damage warnings","source_turns":[2,12]}],
            "constraints":["Used from a global logistics operations desk"]}
    item=derive_stakeholder_contexts(intent,{2,12})[0]
    assert item["role"]=="Supply chain manager"
    assert item["goals"]==["Supply chain managers need early damage warnings"]
    assert item["operating_environment"]=="Used from a global logistics operations desk"
    assert item["source_turns"]==[2,12]
    assert set(item["_derived_fields"]) >= {"role","goals","operating_environment","source_turns"}


def test_all_null_multi_field_patch_is_not_needed_when_intent_derives_stakeholder(monkeypatch):
    """A fully-derived stakeholder costs zero calls; model starts at scenario fields."""
    from interview_engine import model_config
    from interview_engine.graph import build_graph
    from interview_engine.telemetry import UsageLog
    from tests.test_coverage import FakeModel
    d=draft()
    intent={"session_id":"derived","problem_statement":"p","target_users":["runner"],
      "core_value_proposition":"points sync","stakeholder_needs":[{"need_id":"SN-1","statement":"runner needs quick capture","kind":"functional","source_turns":[0]}],
      "in_scope":["Log habit"],"out_of_scope":["native-only"],"external_integrations":[],
      "constraints":["phone, often offline"],"assumptions":[],"assumption_items":[],"scope_exclusions":[],
      "done_criteria":[],"tech_preferences":[],"glossary":{},"operational_concepts":["Log habit"],"open_questions":[],
      "source_turns":{"operational_concepts":[0],"core_value_proposition":[0]}}
    conops=FakeModel([json.dumps({"trigger":"habit completed"}),
                      json.dumps([{"name":"Native app","summary":"installed"},{"name":"PWA","summary":"web"}]),
                      json.dumps({"advantages":["native APIs"]}),json.dumps({"disadvantages":["release"]}),
                      json.dumps({"advantages":["one deployment"]}),json.dumps({"disadvantages":["limits"]})])
    models={"probing":FakeModel([]),"distillation":FakeModel([]),"conops":conops,"scope_tiebreak":FakeModel([])}
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**kw:models[stage])
    g=build_graph(UsageLog("derived"),conops_enabled=True);cfg={"configurable":{"thread_id":"derived"}}
    r=g.invoke({"messages":[HumanMessage(content="runner uses phone offline")],"submission":{"free_text":"web app","founder_id":"f","submission_id":"derived"},
                "verdict":{"accepted":True},"intent_model":intent},cfg)
    ctx=r["conops_draft"]["stakeholder_context"][0]
    assert ctx["goals"]==["runner needs quick capture"] and ctx["operating_environment"].startswith("phone, often offline")
    assert conops.calls==6  # no stakeholder call; trigger + comparison + 4 option fields


def test_per_field_null_is_a_named_failure_with_raw_path(monkeypatch,tmp_path):
    from interview_engine import model_config
    from interview_engine.errors import UserFacingError
    from interview_engine.graph import build_graph
    from interview_engine.telemetry import UsageLog
    from tests.test_coverage import FakeModel, NEEDS, SCOPE
    conops=FakeModel([json.dumps({"operating_environment":None}), json.dumps({"operating_environment":None})])
    models={"probing":FakeModel(["[INTERVIEW COMPLETE]"]),"distillation":FakeModel([json.dumps(NEEDS),json.dumps(SCOPE),"[]"]),
            "conops":conops,"scope_tiebreak":FakeModel(['{"is_web_app":true,"archetype":"saas_crud","confidence":0.9,"reason":"web"}'])}
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**kw:models[stage])
    g=build_graph(UsageLog("null-field"),conops_enabled=True,raw_output_dir=tmp_path/"debug")
    import pytest
    with pytest.raises(UserFacingError) as caught:
      g.invoke({"messages":[HumanMessage(content="phone habit app")],"submission":{"free_text":"web phone habit app with logins","founder_id":"f","submission_id":"null-field"}},
               {"configurable":{"thread_id":"null-field"}})
    rendered=caught.value.render()
    assert "field operating_environment failed its JSON contract after one automatic retry" in rendered
    assert ".raw.txt" in rendered and str((tmp_path/"debug").resolve()) in rendered


def test_conops_has_no_whole_document_or_multifield_repair_path():
    from pathlib import Path
    source=(Path(__file__).parents[1]/"src"/"interview_engine"/"graph.py").read_text()
    assert "last-resort" not in source and "complete corrected JSON" not in source
    assert "Fill ONLY the" in source
    assert "clean_model_patch(" not in source


def test_conops_comparison_wrong_shape_retries_with_contract_last(monkeypatch):
    from interview_engine import model_config
    from interview_engine.graph import build_graph
    from interview_engine.telemetry import UsageLog
    from tests.test_coverage import FakeModel

    intent={"session_id":"repair","problem_statement":"p","target_users":["runner"],
      "core_value_proposition":"points sync","stakeholder_needs":[{"need_id":"SN-1","statement":"runner needs quick capture","kind":"functional","source_turns":[0]}],
      "in_scope":["Log habit"],"out_of_scope":["native-only"],"external_integrations":[],
      "constraints":["phone, often offline"],"assumptions":[],"assumption_items":[],"scope_exclusions":[],
      "done_criteria":[],"tech_preferences":[],"glossary":{},"operational_concepts":["Log habit"],"open_questions":[],
      "source_turns":{"operational_concepts":[0],"core_value_proposition":[0]}}
    class Capture(FakeModel):
        def __init__(self, outputs): super().__init__(outputs); self.messages=[]
        def invoke(self, messages): self.messages.append(messages); return super().invoke(messages)
    conops=Capture([json.dumps({"trigger":"habit completed"}),
                    json.dumps({"stakeholder_context": []}),
                    json.dumps([{"name":"Native","summary":"installed"},{"name":"PWA","summary":"web"}]),
                    json.dumps({"advantages":["native"]}),json.dumps({"disadvantages":["release"]}),
                    json.dumps({"advantages":["one deployment"]}),json.dumps({"disadvantages":["limits"]})])
    models={"probing":FakeModel([]),"distillation":FakeModel([]),"conops":conops,"scope_tiebreak":FakeModel([])}
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**kw:models[stage])
    graph=build_graph(UsageLog("repair"),conops_enabled=True)
    result=graph.invoke({"messages":[HumanMessage(content="runner uses phone offline")],
      "submission":{"free_text":"web app","founder_id":"f","submission_id":"repair"},
      "verdict":{"accepted":True},"intent_model":intent},{"configurable":{"thread_id":"repair"}})
    assert len(result["conops_draft"]["concept_options"]) == 2
    assert conops.messages[1][-1].content.startswith("FINAL RESPONSE CONTRACT:")
    assert "previous response failed" in conops.messages[2][-1].content


def test_conops_option_dict_statements_normalize_at_assembly(monkeypatch):
    from interview_engine.repair import normalize_model_item
    from vmodel_contracts import ConceptOption

    live={"concept_id":"concept:agent","name":"Agent","summary":"acts for garage",
          "advantages":[{"statement":"Processes responses in one go","source_turns":[12]}],
          "disadvantages":[{"statement":"Must handle insurer variation","source_turns":[12]}]}
    normalized=normalize_model_item(live,ConceptOption)
    option=ConceptOption.model_validate(normalized)
    assert option.advantages == ["Processes responses in one go"]
    assert option.disadvantages == ["Must handle insurer variation"]
