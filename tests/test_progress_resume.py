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

VALID_INTENT = {
    "session_id": "t1",
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


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        return AIMessage(content=self.replies.pop(0) if self.replies else "[INTERVIEW COMPLETE]")


DEFAULT_REPLIES = {
    "probing": ["[INTERVIEW COMPLETE]"],
    "distillation": None,  # set below (needs VALID_INTENT)
    "scope_tiebreak": ['{"is_web_app": true, "archetype": "saas_crud"}'],
}


@pytest.fixture
def fake_models(monkeypatch):
    box = {}

    def _build(stage):
        if stage not in box:
            replies = DEFAULT_REPLIES[stage] if DEFAULT_REPLIES[stage] else [json.dumps(VALID_INTENT)]
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
    assert ("distill_intent", "completed") in kinds
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
    drifted = dict(VALID_INTENT, source_turns={"a": {}}, constraints=None, target_users="founders")
    fake_models["distillation"] = FakeModel([json.dumps(drifted)])
    g = build_graph(UsageLog("t1"))
    cfg = {"configurable": {"thread_id": "t3"}}
    r = run_one_turn(g, cfg)  # marker reply on turn 1 -> distill in same invoke
    IntentModel.model_validate(r["intent_model"])  # repaired + validated in-node


def test_distill_retries_once_with_schema_feedback(fake_models):
    bad = json.dumps({"problem_statement": 1})  # unrepairable: missing everything
    fake_models["distillation"] = FakeModel([bad, json.dumps(VALID_INTENT)])
    g = build_graph(UsageLog("t1"))
    cfg = {"configurable": {"thread_id": "t4"}}
    r = run_one_turn(g, cfg)
    assert IntentModel.model_validate(r["intent_model"]).problem_statement == "founders rewrite forever"


def test_crash_resume_from_disk_checkpoint(tmp_path, fake_models):
    from langgraph.checkpoint.sqlite import SqliteSaver

    # two-turn interview: question first, so the first invoke pauses for input
    fake_models["probing"] = FakeModel(["What problem does it solve?", "[INTERVIEW COMPLETE]"])
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
