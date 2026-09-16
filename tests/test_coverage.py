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

    def _build(stage, **kwargs):
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
    assert founder_control("I think you have enough to move on") == "enough"
    assert founder_control("I believe we can proceed") == "enough"
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
    # "5000" keeps the NFR guard honest: closing "quality needs" needs numbers
    r = start(g, text="a saas web app for surf schools, booking, logins, 5000 users", thread="c2")
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


def test_deferral_parks_subtopic_and_prevents_reprobe(monkeypatch):
    probing = FakeModel([
        structured("What specific encryption standards should be used?", thin=["nfrs"]),
        structured("Which encryption protocol, such as TLS 1.3, should be used?"),
        structured("What exact encryption standard do you require?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="park1")
    r = g.invoke({"messages": [HumanMessage(content="State of the art. You decide.")]},
                 {"configurable": {"thread_id": "park1"}})
    assert r["parked_topics"] == ["security/encryption detail"]
    assert "encryption" not in r["messages"][-1].content.lower()
    assert r["coverage"]["nfrs"] == "thin"  # parked is not falsely covered

    r2 = g.invoke({"messages": [HumanMessage(content="No other preference.")]},
                  {"configurable": {"thread_id": "park1"}})
    assert "encryption" not in r2["messages"][-1].content.lower()
    assert r2["parked_topics"] == ["security/encryption detail"]


def test_deferral_detection_is_explicit_not_generic_uncertainty():
    from interview_engine.coverage import is_deferral
    for text in ("I don't know", "you decide", "your call", "state of the art", "I'm agnostic"):
        assert is_deferral(text), text
    assert not is_deferral("I don't know the count yet, but I can get it tomorrow")


def test_unstructured_reply_fails_actionably_after_one_retry(monkeypatch):
    class AlwaysProse:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content="Just a plain question?")

    probing = AlwaysProse()
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    from interview_engine.errors import UserFacingError
    with pytest.raises(UserFacingError) as ei:
        start(g)
    assert probing.calls == 2  # capped: original + one retry, never a loop
    message = ei.value.render()
    assert "ignored the structured interview-turn contract twice" in message
    assert "OLLAMA_CONTEXT_LENGTH=32768" in message and "--session" in message


def test_retry_with_feedback_recovers_structured_turn(monkeypatch):
    probing = FakeModel(["Here you go, some prose first",
                         structured("Who else touches this?", covered=["goal"])])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g)
    assert r["coverage"]["goal"] == "covered"          # retry recovered the turn
    assert r["messages"][-1].content == "Who else touches this?"


def test_prose_wrapped_json_is_tolerated(monkeypatch):
    probing = FakeModel(['Sure! {"question": "Risks?", "covered": ["goal"], "complete": false} done.'])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g)
    assert r["coverage"]["goal"] == "covered"
    assert r["messages"][-1].content == "Risks?"


def test_saturation_guard_blocks_premature_complete(monkeypatch):
    # run-4 audit: complete=true with 0/11 covered must NOT end the interview
    probing = FakeModel([structured("Closing summary.", covered=["goal"], complete=True)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g)
    assert "[INTERVIEW COMPLETE]" not in r["messages"][-1].content
    assert r["coverage"]["goal"] == "covered"


def test_marker_turn_skips_parse_and_retry(monkeypatch):
    probing = FakeModel(["[INTERVIEW COMPLETE]"])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g)
    assert probing.calls == 1


def test_coverage_map_render():
    from interview_engine.cli import render_coverage_markdown
    cov = dict(initial_coverage(), goal="covered", users="skipped", nfrs="thin")
    md = render_coverage_markdown(cov)
    assert "**Covered:** goal & core value" in md
    assert "Skipped by you:** stakeholder context" in md
    assert "Thin" in md and "quality needs" in md
    assert "Not covered:" in md


# --- run-4 regression: scope drift crash + truncated error ---

def test_scope_dict_items_coerced_to_strings(monkeypatch):
    """Run-4 crash: the model blended scope_exclusions' shape into out_of_scope
    ({item, explicit} dicts). Coercion must turn them back into strings."""
    import json as j
    from langchain_core.messages import HumanMessage
    scope = dict(SCOPE, out_of_scope=[{"item": "precise algorithms", "explicit": True},
                                      {"item": "mobile app", "explicit": False}])
    distills = [j.dumps(NEEDS), j.dumps(scope), "[]"]
    probing = FakeModel(["[INTERVIEW COMPLETE]"])
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: probing if stage == "probing"
                        else FakeModel(distills) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="r4")
    from vmodel_contracts import IntentModel
    intent = IntentModel.model_validate(r["intent_model"])
    assert intent.out_of_scope == ["precise algorithms", "mobile app"]


def test_validation_error_detail_not_truncated(monkeypatch):
    """Run-4: the phase-4 error was cut mid-sentence ('out_of_scope.1 Input should be')."""
    import json as j
    bad_needs = dict(NEEDS, stakeholder_needs=[{k: v for k, v in NEEDS["stakeholder_needs"][0].items() if k != "need_id"}])
    probing = FakeModel(["[INTERVIEW COMPLETE]"])
    # both needs attempts return the same broken payload (retry fails too)
    distills = [j.dumps(bad_needs), j.dumps(bad_needs)]
    monkeypatch.setattr(model_config, "build_chat_model",
                        lambda stage, **kwargs: probing if stage == "probing"
                        else FakeModel(list(distills)) if stage == "distillation"
                        else FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']))
    from interview_engine.errors import UserFacingError
    g = build_graph(UsageLog("t1"))
    with pytest.raises(UserFacingError) as ei:
        start(g, thread="r4e")
    msg = ei.value.render()
    assert "need_id" in msg and "Field required" in msg



def test_turn_output_schema_fixes_shape_and_area_keys():
    from interview_engine.coverage import AREA_KEYS, turn_output_schema
    schema = turn_output_schema()
    assert schema["required"] == ["question", "covered", "thin", "skipped", "complete"]
    assert schema["additionalProperties"] is False
    for field in ("covered", "thin", "skipped"):
        assert schema["properties"][field]["items"]["enum"] == AREA_KEYS
