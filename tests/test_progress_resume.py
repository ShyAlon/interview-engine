"""Progress events, distillation repair/retry, and crash-resume from disk checkpoints."""
import json
import sqlite3

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from interview_engine import model_config
from interview_engine.graph import build_graph
from interview_engine.progress import ProgressBus, track
from interview_engine.repair import coerce_for_model
from interview_engine.telemetry import UsageLog
from vmodel_contracts import IntentModel

NEEDS_REPLY = {
    "stakeholder_needs": [
        {"need_id": "SN-1", "statement": "founders need a spec before code",
         "kind": "functional", "source_turns": [0]},
        {"need_id": "SN-2", "statement": "founder data must stay secure",
         "kind": "non_functional", "source_turns": [1]},
    ],
    "assumptions": [],
}

SCOPE_REPLY = {
    "problem_statement": "founders rewrite forever",
    "target_users": ["solo founders"],
    "core_value_proposition": "spec before code",
    "in_scope": ["interviews"],
    "out_of_scope": ["codegen"],
    "external_integrations": [],
    "constraints": [],
    "open_questions": [],
    "source_turns": {"problem_statement": [0, 1]},
}

PREVIEW_REPLY = [
    {"need_id": "SN-1", "kind": "functional",
     "statement": "The system shall generate a versioned spec bundle from an approved intent model.",
     "example_criterion": "bundle/ contains manifest.json validating against SpecManifest"},
]

VALID_INTENT = {
    "session_id": "t1",
    "stakeholder_needs": NEEDS_REPLY["stakeholder_needs"],
    "assumptions": [],
    **SCOPE_REPLY,
}


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        return AIMessage(content=self.replies.pop(0) if self.replies else "[INTERVIEW COMPLETE]")


DEFAULT_REPLIES = {
    "probing": ["[INTERVIEW COMPLETE]"],
    "distillation": None,  # set below (needs NEEDS_REPLY/SCOPE_REPLY)
    "scope_tiebreak": ['{"is_web_app": true, "archetype": "saas_crud"}'],
}


@pytest.fixture
def fake_models(monkeypatch):
    box = {}

    def _build(stage, **kwargs):
        if stage not in box:
            replies = DEFAULT_REPLIES[stage] if DEFAULT_REPLIES[stage] else [
                json.dumps(NEEDS_REPLY), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)]
            box[stage] = FakeModel(list(replies))
        return box[stage]

    monkeypatch.setattr(model_config, "build_chat_model", _build)
    return box


def run_one_turn(graph, cfg, text="a saas web app for surf schools, with booking and customer logins"):
    return graph.invoke(
        {"messages": [HumanMessage(content=text)],
         "submission": {"free_text": text, "founder_id": "t", "submission_id": "t1"}},
        cfg,
    )


def test_progress_events_flow_and_elapsed(fake_models):
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    g = build_graph(UsageLog("t1"), progress=bus)
    cfg = {"configurable": {"thread_id": "t1"}}
    run_one_turn(g, cfg)  # gate + one turn (marker reply) + distill, all in one invoke
    kinds = [(e.stage, e.status) for e in events]
    assert ("scope_gate", "started") in kinds and ("scope_gate", "completed") in kinds
    assert ("interview_turn", "started") in kinds
    for phase in ("distill_needs", "distill_scope", "distill_preview", "distill_validate"):
        assert (phase, "completed") in kinds, phase
    done = [e for e in events if e.status == "completed"]
    assert all(e.elapsed_s is not None and e.elapsed_s >= 0 for e in done)


def test_broken_subscriber_never_crashes(fake_models):
    bus = ProgressBus()
    bus.subscribe(lambda ev: 1 / 0)
    g = build_graph(UsageLog("t1"), progress=bus)
    r = run_one_turn(g, {"configurable": {"thread_id": "t2"}})  # must not raise
    assert "intent_model" in r


def test_repair_empty_dict_list_drift():
    data = dict(VALID_INTENT, in_scope={}, source_turns={"in_scope": {}})
    fixed = coerce_for_model(data, IntentModel)
    intent = IntentModel.model_validate(fixed)
    assert intent.in_scope == [] and intent.source_turns == {"in_scope": []}


def test_repair_string_to_list():
    fixed = coerce_for_model(dict(VALID_INTENT, target_users="founders"), IntentModel)
    assert IntentModel.model_validate(fixed).target_users == ["founders"]


def test_distill_repairs_drift_without_retry(fake_models):
    drifted_scope = dict(SCOPE_REPLY, source_turns={"a": {}}, constraints=None, target_users="founders")
    drifted_needs = dict(NEEDS_REPLY, stakeholder_needs=[
        {"need_id": "SN-1", "statement": "a", "kind": "non-functional", "source_turns": [0]}])
    fake_models["distillation"] = FakeModel([json.dumps(drifted_needs), json.dumps(drifted_scope), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    cfg = {"configurable": {"thread_id": "t3"}}
    r = run_one_turn(g, cfg)  # marker reply on turn 1 -> distill in same invoke
    IntentModel.model_validate(r["intent_model"])  # repaired + validated in-node


def test_distill_retries_once_with_schema_feedback(fake_models):
    bad = json.dumps({"problem_statement": 1})  # unrepairable scope: missing everything
    fake_models["distillation"] = FakeModel([json.dumps(NEEDS_REPLY), bad, json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    cfg = {"configurable": {"thread_id": "t4"}}
    r = run_one_turn(g, cfg)
    assert IntentModel.model_validate(r["intent_model"]).problem_statement == "founders rewrite forever"


def test_crash_resume_from_disk_checkpoint(tmp_path, fake_models):
    from langgraph.checkpoint.sqlite import SqliteSaver

    # two-turn interview: question first, so the first invoke pauses for input
    fake_models["probing"] = FakeModel([
        '{"question": "What problem does it solve?", "covered": [], "thin": [], "skipped": [], "complete": false}',
        "[INTERVIEW COMPLETE]"])
    db = str(tmp_path / "cp.sqlite")
    cfg = {"configurable": {"thread_id": "s-resume"}}
    g1 = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    r1 = run_one_turn(g1, cfg)
    assert r1["messages"][-1].content == "What problem does it solve?"
    assert "intent_model" not in r1  # paused for the founder's answer

    # simulate the crash: drop everything in-process, rebuild from the same db
    del g1
    g2 = build_graph(UsageLog("s"), checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    saved = g2.get_state(cfg)
    assert saved.values["messages"], "checkpoint must survive process death"
    r2 = g2.invoke({"messages": [HumanMessage(content="it solves rewrite hell")]}, cfg)
    assert "intent_model" in r2, "resume continued the interview to distillation"
    assert r2.get("turn_count") == 2  # continued, not restarted


def test_state_is_json_native_no_msgpack_warning(fake_models, recwarn):
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serde = JsonPlusSerializer()
    from vmodel_contracts import IntakeSubmission
    sub = IntakeSubmission(free_text="x", founder_id="t", submission_id="t")
    dumped = sub.model_dump(mode="json")
    serde.loads_typed(serde.dumps_typed(dumped))
    import warnings
    for w in recwarn.list:
        assert "unregistered type" not in str(w.message)


# ---- v0.2.0: SE rigor, timestamps, enum drift, version mismatch ----

def test_session_id_is_timestamped_with_suffix():
    import re

    from interview_engine import cli

    # replicate the cli's session construction
    import uuid
    from datetime import datetime

    sid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", sid)


def test_renderer_splits_functional_and_nonfunctional():
    from vmodel_contracts import IntentModel

    from interview_engine.cli import render_intent_markdown

    intent = IntentModel.model_validate(VALID_INTENT)
    md = render_intent_markdown(intent)
    assert "Stakeholder needs - functional" in md
    assert "Stakeholder needs - non-functional" in md
    assert "**SN-1**" in md and "**SN-2**" in md
    assert "requirements" in md.lower()  # the 'derived downstream' note


def test_enum_drift_normalized_before_validation():
    from interview_engine.repair import normalize_enums, coerce_for_model

    drifted = dict(VALID_INTENT, stakeholder_needs=[
        {"need_id": "SN-1", "statement": "a", "kind": "Functional", "source_turns": [0]},
        {"need_id": "SN-2", "statement": "b", "kind": "non-functional", "source_turns": [1]},
    ])
    fixed = normalize_enums(coerce_for_model(drifted, IntentModel), IntentModel)
    intent = IntentModel.model_validate(fixed)
    assert [n.kind.value for n in intent.stakeholder_needs] == ["functional", "non_functional"]


def test_old_contract_payload_fails_loudly():
    import pytest

    from vmodel_contracts import ContractVersionMismatch, check_compatible

    old = dict(VALID_INTENT, contract_version="0.1.0")
    with pytest.raises(ContractVersionMismatch):
        check_compatible(old)


# ---- phased distillation (v0.3.0) ----

def test_phased_distill_emits_all_phases_and_preview(fake_models):
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    g = build_graph(UsageLog("t1"), progress=bus)
    r = run_one_turn(g, {"configurable": {"thread_id": "ph"}})
    assert "intent_model" in r
    assert r["requirement_preview"] and r["requirement_preview"][0]["need_id"] == "SN-1"
    phases = [e.stage for e in events if e.status == "started"]
    assert [p for p in phases if p.startswith("distill")] == [
        "distill_needs", "distill_scope", "distill_preview", "distill_validate"]


def test_preview_failure_degrades_gracefully(fake_models):
    fake_models["distillation"] = FakeModel([json.dumps(NEEDS_REPLY), json.dumps(SCOPE_REPLY), "garbage{{"])
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    g = build_graph(UsageLog("t1"), progress=bus)
    r = run_one_turn(g, {"configurable": {"thread_id": "pd"}})
    assert "intent_model" in r  # the run completes
    assert r["requirement_preview"] == []
    assert any("unavailable" in (e.detail or "") for e in events)


def test_scope_phase_retries_once_with_feedback(fake_models):
    bad_scope = json.dumps({"problem_statement": ""})  # empty: violates validator
    fake_models["distillation"] = FakeModel([json.dumps(NEEDS_REPLY), bad_scope, json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "pr"}})
    assert IntentModel.model_validate(r["intent_model"])


def test_needs_phase_fails_loudly_after_retry(fake_models):
    import pytest

    from interview_engine.errors import UserFacingError

    bad_needs = json.dumps({"stakeholder_needs": []})  # empty: not an intent model
    fake_models["distillation"] = FakeModel([bad_needs, bad_needs])
    g = build_graph(UsageLog("t1"))
    with pytest.raises(UserFacingError) as ei:
        run_one_turn(g, {"configurable": {"thread_id": "pf"}})
    assert "interview-level needs" in ei.value.render().lower() or "Extracting interview-level needs" in ei.value.render()


def test_legacy_flat_assumptions_dropped_as_ungrounded(fake_models):
    needs = dict(NEEDS_REPLY, assumptions=["internal and external data cannot conflict"])
    fake_models["distillation"] = FakeModel([json.dumps(needs), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "pa"}})
    intent = IntentModel.model_validate(r["intent_model"])
    assert intent.assumptions == [] and intent.assumption_items == []
    assert all("conflict" not in n.statement for n in intent.stakeholder_needs)


def test_grounded_assumption_survives(fake_models):
    needs = dict(NEEDS_REPLY,
                 assumption_items=[{"statement": "customers always book through schools",
                                    "source_turns": [0]}])
    needs.pop("assumptions", None)
    fake_models["distillation"] = FakeModel([json.dumps(needs), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "pg"}})
    intent = IntentModel.model_validate(r["intent_model"])
    assert [a.statement for a in intent.assumption_items] == ["customers always book through schools"]


def test_prompt_echo_assumption_dropped_in_graph(fake_models):
    # run-3 defect: the model echoed the prompt's own example assertions
    needs = dict(NEEDS_REPLY,
                 assumption_items=[{"statement": "these sources can't conflict", "source_turns": [0]},
                                   {"statement": "users will always have connectivity", "source_turns": [0]}])
    needs.pop("assumptions", None)
    fake_models["distillation"] = FakeModel([json.dumps(needs), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "pe"}})
    intent = IntentModel.model_validate(r["intent_model"])
    assert intent.assumption_items == []


def test_duplicate_constraints_dropped_in_graph(fake_models):
    scope = dict(SCOPE_REPLY, constraints=["founder data must stay secure", "$0 budget cap"])
    fake_models["distillation"] = FakeModel(
        [json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "pd"}})
    intent = IntentModel.model_validate(r["intent_model"])
    assert intent.constraints == ["$0 budget cap"]


def test_preview_solution_language_stripped(fake_models):
    bad = [{"need_id": "SN-1", "kind": "functional",
            "statement": "The system shall flag conflicts.",
            "example_criterion": "GET /dashboard returns a JSON response"}]
    fake_models["distillation"] = FakeModel(
        [json.dumps(NEEDS_REPLY), json.dumps(SCOPE_REPLY), json.dumps(bad)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "ps"}})
    assert r["requirement_preview"][0].get("example_criterion") is None


def test_renderer_shows_unverified_boundary_when_out_of_scope_empty():
    from vmodel_contracts import IntentModel

    from interview_engine.cli import render_intent_markdown

    intent = IntentModel.model_validate(dict(VALID_INTENT, out_of_scope=[]))
    assert "none identified" in render_intent_markdown(intent)


def test_preview_markdown_in_approval_gate():
    from interview_engine.cli import render_preview_markdown

    md = render_preview_markdown(PREVIEW_REPLY)
    assert "SN-1 -> R-p1" in md and "spec-generator" in md
    assert render_preview_markdown([]) == ""


def test_extract_json_value_shapes():
    from interview_engine.repair import extract_json_value
    assert extract_json_value('{"a": 1}') == '{"a": 1}'
    assert extract_json_value('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert json.loads(extract_json_value('Sure! Result:\n{"a": {"b": [1]}}\nDone')) == {"a": {"b": [1]}}
    assert json.loads(extract_json_value('Output: [1, {"x": "y"}] end')) == [1, {"x": "y"}]
    assert json.loads(extract_json_value('{"s": "a } brace {"}')) == {"s": "a } brace {"}
    with pytest.raises(ValueError):
        extract_json_value('{"a": {"b": [1, 2')  # truncated


def test_distill_recovers_from_prose_wrapped_json(fake_models):
    """Run-6: a transient invalid-JSON flake auto-retries once with feedback -
    no human, no lost run (his spec-generator 'tests' failure, same class)."""
    fake_models["distillation"] = FakeModel([
        "Sure, let me help! Here is your needs JSON: (no actual json)",
        json.dumps(NEEDS_REPLY), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)])
    events = []
    bus = ProgressBus()
    bus.subscribe(events.append)
    g = build_graph(UsageLog("s"), checkpointer=None, progress=bus)
    r = run_one_turn(g, {"configurable": {"thread_id": "flake1"}})
    assert "intent_model" in r
    assert any("not valid JSON - retrying once" in (e.detail or "") for e in events)


def test_distill_persistent_garbage_friendly_error_with_snippet(fake_models):
    from interview_engine.errors import UserFacingError
    fake_models["distillation"] = FakeModel(["no json here at all", "still no json"])
    g = build_graph(UsageLog("s"), checkpointer=None, progress=ProgressBus())
    with pytest.raises(UserFacingError) as ei:
        run_one_turn(g, {"configurable": {"thread_id": "flake2"}})
    rendered = ei.value.render()
    assert "automatic retry did not fix it" in rendered
    assert "still no json" in rendered
