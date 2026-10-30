"""Progress events, distillation repair/retry, and crash-resume from disk checkpoints."""
import json

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
         "kind": "non_functional", "source_turns": [0]},
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
    "source_turns": {"problem_statement": [0]},
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
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
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


def test_crash_resume_from_pg_checkpoint(tmp_path, fake_models):
    from test_pg_checkpoint import make_saver
    from conftest import managed_pg_server
    server=managed_pg_server(tmp_path/'resume-pg')
    ctx,saver,admin=make_saver(server.get_uri())

    # two-turn interview: question first, so the first invoke pauses for input
    fake_models["probing"] = FakeModel([
        '{"question": "What problem does it solve?", "covered": [], "thin": [], "skipped": [], "complete": false}',
        "[INTERVIEW COMPLETE]"])
    cfg = {"configurable": {"thread_id": "s-resume"}}
    g1 = build_graph(UsageLog("s"), checkpointer=saver)
    r1 = run_one_turn(g1, cfg)
    assert r1["messages"][-1].content == "What problem does it solve?"
    assert "intent_model" not in r1  # paused for the founder's answer

    # Rebuild graph over the same real fenced PG checkpoint.
    del g1
    g2 = build_graph(UsageLog("s"), checkpointer=saver)
    saved = g2.get_state(cfg)
    assert saved.values["messages"], "checkpoint must survive process death"
    r2 = g2.invoke({"messages": [HumanMessage(content="it solves rewrite hell")]}, cfg)
    assert "intent_model" in r2, "resume continued the interview to distillation"
    assert r2.get("turn_count") == 2  # continued, not restarted
    ctx.__exit__(None,None,None)


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

def test_renderer_splits_functional_and_nonfunctional():
    from vmodel_contracts import IntentModel

    from interview_engine.rendering import render_intent_markdown

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
        {"need_id": "SN-2", "statement": "b", "kind": "non-functional", "source_turns": [0]},
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
    assert intent.assumptions == ['internal and external data cannot conflict'] and intent.assumption_items == []
    assert r['semantic_review']['findings'][0]['rule']=='missing_provenance'
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
    assert len(intent.assumption_items)==2
    assert len(r['semantic_review']['findings'])==2


def test_duplicate_constraints_dropped_in_graph(fake_models):
    scope = dict(SCOPE_REPLY, constraints=["founder data must stay secure", "$0 budget cap"])
    fake_models["distillation"] = FakeModel(
        [json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "pd"}})
    intent = IntentModel.model_validate(r["intent_model"])
    assert intent.constraints == ['founder data must stay secure','$0 budget cap']
    assert r['semantic_review']['findings'][0]['rule']=='duplicate_need'


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

    from interview_engine.rendering import render_intent_markdown

    intent = IntentModel.model_validate(dict(VALID_INTENT, out_of_scope=[]))
    assert "none identified" in render_intent_markdown(intent)


def test_preview_markdown_in_approval_gate():
    from interview_engine.rendering import render_preview_markdown

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


def test_dict_shaped_constraint_normalized_before_dedup(fake_models):
    """Live session d630: phase 4 crashed in content_words because the scope
    model emitted a structured constraint object instead of a string."""
    scope = dict(SCOPE_REPLY, constraints=[
        {"constraint": "$0 infrastructure budget during the demo", "source_turns": [0]},
        {"statement": "founder data must stay secure", "source_turns": [0]},
    ])
    fake_models["distillation"] = FakeModel(
        [json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(PREVIEW_REPLY)])
    g = build_graph(UsageLog("t1"))
    r = run_one_turn(g, {"configurable": {"thread_id": "dict-constraint"}})
    intent = IntentModel.model_validate(r["intent_model"])
    assert intent.constraints == ['$0 infrastructure budget during the demo','founder data must stay secure']
    assert r['semantic_review']['findings'][0]['rule']=='duplicate_need'
    assert all(isinstance(c, str) for c in intent.constraints)


def test_progress_printer_prepends_local_wall_clock_timestamp(capsys):
    from interview_engine.progress import ProgressEvent, stderr_printer

    printer = stderr_printer()
    printer(ProgressEvent("x", "Working", "started", 0.0))
    printer(ProgressEvent("x", "Working", "completed", 1.0, 1.0))
    lines = capsys.readouterr().err.splitlines()
    assert all(line.startswith("[") and line[3] == ":" and line[6] == ":" and line[9] == "]" for line in lines)
    assert "  -> Working..." in lines[0]
    assert "  ok Working [1.0s]" in lines[1]


def test_console_severity_color_is_absent_when_captured_or_no_color(monkeypatch, capsys):
    from interview_engine import model_config as mc

    mc._console_line("WARNING fallback")
    assert "\x1b[" not in capsys.readouterr().err
    monkeypatch.setenv("NO_COLOR", "1")
    mc._console_line("ERROR failed")
    assert "\x1b[" not in capsys.readouterr().err



def test_legacy_graph_checkpoint_switches_pinned_model_and_resumes(monkeypatch,tmp_path):
    """Exercise the real durable graph, not just a fake session program."""
    from interview_engine import session_models
    from interview_engine.http_service import SessionRegistry,default_program_factory
    from vmodel_contracts.interactive import StartSessionRequest,ModelRouteSwitchRequest,InputResponse
    from uuid import uuid4
    monkeypatch.setenv('VMODEL_ARTIFACTS_DIR',str(tmp_path))
    monkeypatch.setattr(session_models,'validate',lambda *args,**kwargs:None)
    seen=[];replies=[
      '{"question":"What problem does it solve?","covered":[],"thin":[],"skipped":[],"complete":false}',
      '{"question":"Who uses it first?","covered":[],"thin":[],"skipped":[],"complete":false}']
    class RoutedModel:
        def invoke(self,messages):
            if isinstance(messages,list) and 'Reply ONLY SAME or NEW' in getattr(messages[0],'content',''):
                return AIMessage(content='NEW')
            seen.append(session_models.selected('probing'))
            return AIMessage(content=replies.pop(0))
    def build(stage,**kwargs):
        return RoutedModel() if stage=='probing' else FakeModel(['{"is_web_app":true,"archetype":"saas_crud","confidence":0.9,"reason":"web app"}'] if stage=='scope_tiebreak' else [json.dumps(NEEDS_REPLY),json.dumps(SCOPE_REPLY),json.dumps(PREVIEW_REPLY)])
    monkeypatch.setattr(model_config,'build_chat_model',build)
    from test_pg_application_wiring import make_conn,OWNER,_OPEN,PgSessionStore
    from conftest import managed_pool
    conn=make_conn(tmp_path);store=PgSessionStore(managed_pool(conn.info.dsn))
    registry=SessionRegistry(default_program_factory,session_store=store,owner_id=OWNER,allow_db_provider_dispatch=True)
    _OPEN['registries'].append(registry)
    session,_=registry.start(StartSessionRequest(task_id='legacy-task',idea='App for surf schools',model_route='ollama',initial_model='qwen2.5-coder:14b'))
    assert session.open_input and seen==[('ollama','qwen2.5-coder:14b')]
    saved=store.load_published(session.session_id,OWNER)['control'];saved.pop('route_version');saved.pop('route_history')
    with store._borrow() as c:
        c.execute('UPDATE engine_sessions SET control=%s::jsonb WHERE session_id=%s',(json.dumps(saved),session.session_id));c.commit()
    registry.close()
    restarted=SessionRegistry(default_program_factory,session_store=store,owner_id=OWNER,allow_db_provider_dispatch=True)
    _OPEN['registries'].append(restarted)
    old=restarted.get(session.session_id)
    assert old.program.graph.get_state(old.program.cfg).values['messages']
    request=ModelRouteSwitchRequest(model_route='ollama',model='phi4',actor_id='founder',expected_sequence=old.sequence,
        expected_input_id=old.open_input.input_id,idempotency_key=uuid4())
    _,snapshot=restarted.switch_route(session.session_id,request)
    assert snapshot.model_map['probing']==('ollama','phi4') and snapshot.open_input.input_id==old.open_input.input_id
    restarted.respond(session.session_id,InputResponse(input_id=old.open_input.input_id,value='Help surfers book lessons',actor_id='founder'))
    assert ('ollama','phi4') in seen


def test_distill_keeps_grounded_reference_names_and_flags_the_rest(fake_models):
    text = "a saas web app for surf schools, like Calendly, working title WaveDesk"
    scope = dict(SCOPE_REPLY, reference_names=[
        {"name": "Calendly", "kind": "reference_product", "source_turns": [0]},
        {"name": "WaveDesk", "kind": "working_title", "source_turns": [0]},
        {"name": "Booksy", "kind": "competitor", "source_turns": [0]},   # the founder never said it
    ])
    fake_models["distillation"] = FakeModel([json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(PREVIEW_REPLY)])
    r = run_one_turn(build_graph(UsageLog("t1")), {"configurable": {"thread_id": "t-refnames"}}, text=text)
    intent = r["intent_model"]
    assert {"Calendly", "WaveDesk"} <= {n["name"] for n in intent["reference_names"]}
    assert not any(name in " ".join(intent["constraints"]) for name in ("Calendly", "WaveDesk", "Booksy"))
    rules = [f["rule"] for f in (r.get("semantic_review") or {}).get("findings", [])]
    assert "ungrounded_reference_name" in rules


@pytest.mark.parametrize("retry_scope", [False, True])
def test_scope_receives_validated_needs_before_transcript(fake_models, retry_scope):
    from interview_engine.prompts import DISTILL_NEEDS_SYSTEM, DISTILL_SCOPE_SYSTEM

    scope = dict(SCOPE_REPLY, constraints=["Annual hosting budget is capped at 100 euros."])
    replies = [json.dumps(NEEDS_REPLY)]
    if retry_scope:
        replies.append(json.dumps({"problem_statement": ""}))
    replies.extend([json.dumps(scope), json.dumps(PREVIEW_REPLY)])
    model = FakeModel(replies)
    fake_models["distillation"] = model
    result = run_one_turn(build_graph(UsageLog("t1")),
                          {"configurable": {"thread_id": f"scope-needs-{retry_scope}"}})

    assert model.calls[0][0].content.startswith(DISTILL_NEEDS_SYSTEM + "\n\nTRANSCRIPT:\n")
    expected = (DISTILL_SCOPE_SYSTEM + "\n\nNEEDS ALREADY CAPTURED:\n"
                + "\n".join(f"- {n['statement']}" for n in NEEDS_REPLY["stakeholder_needs"])
                + "\n\nTRANSCRIPT:\n")
    for call in model.calls[1:3 if retry_scope else 2]:
        assert call[0].content.startswith(expected)
    if retry_scope:
        assert "failed validation" in model.calls[2][1].content
    intent = IntentModel.model_validate(result["intent_model"])
    assert [n.statement for n in intent.stakeholder_needs] == [
        n["statement"] for n in NEEDS_REPLY["stakeholder_needs"]]
    assert intent.constraints == scope["constraints"]


def test_scope_addendum_matches_structural_replay_wording():
    from interview_engine.prompts import DISTILL_SCOPE_SYSTEM

    assert DISTILL_SCOPE_SYSTEM.endswith(
        "\n\nThe stakeholder needs below are ALREADY CAPTURED. List a constraint only if it adds a limit that "
        "none of these needs states. Never restate a need as a constraint.")


def test_distillation_progress_has_no_hard_coded_phase_count(fake_models):
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    result = run_one_turn(build_graph(UsageLog("t1"), progress=bus),
                          {"configurable": {"thread_id": "no-phase-count"}})
    assert "intent_model" in result
    started = [e for e in events if e.stage.startswith("distill_") and e.status == "started"]
    assert started
    assert all(e.label and e.detail is None for e in started)
    assert not any("/4" in (e.detail or "") for e in events)


def test_interviewer_reference_citation_repairs_before_candidate(fake_models):
    scope = dict(SCOPE_REPLY, reference_names=[
        {'name': 'D&D Beyond', 'kind': 'reference_product', 'source_turns': [1]}])
    repaired = dict(SCOPE_REPLY, reference_names=[])
    model = fake_models['distillation'] = FakeModel([
        json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(repaired), json.dumps(PREVIEW_REPLY)])
    result = run_one_turn(build_graph(UsageLog('t1')), {'configurable': {'thread_id': 'ref-role-repair'}})
    assert result['intent_model']['reference_names'] == []
    assert not any(f['rule'] == 'ungrounded_reference_name'
                   for f in (result.get('semantic_review') or {}).get('findings', []))
    feedback = ' '.join(m.content for m in model.calls[2])
    assert 'reference_names[0].source_turns cites non-founder evidence' in feedback
    assert 'interviewer, not you' in feedback


def test_repeated_interviewer_reference_citation_is_defect_not_review(fake_models):
    from interview_engine.errors import UserFacingError
    scope = dict(SCOPE_REPLY, reference_names=[
        {'name': 'D&D Beyond', 'kind': 'reference_product', 'source_turns': [1]}])
    fake_models['distillation'] = FakeModel([
        json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(scope)])
    with pytest.raises(UserFacingError) as error:
        run_one_turn(build_graph(UsageLog('t1')), {'configurable': {'thread_id': 'ref-role-defect'}})
    assert 'non-founder evidence' in error.value.render()


def test_reference_mismatch_review_shows_exact_founder_quote(fake_models):
    scope = dict(SCOPE_REPLY, reference_names=[
        {'name': 'D&D Beyond', 'kind': 'reference_product', 'source_turns': [0]}])
    fake_models['distillation'] = FakeModel([
        json.dumps(NEEDS_REPLY), json.dumps(scope), json.dumps(PREVIEW_REPLY)])
    result = run_one_turn(build_graph(UsageLog('t1')), {'configurable': {'thread_id': 'ref-quote'}},
                          text='a saas web app like DnD Beyond')
    finding = next(f for f in result['semantic_review']['findings'] if f['rule'] == 'ungrounded_reference_name')
    assert 'This exact product name was not found in your cited words.' in finding['reason']
    assert 'Turn 0 (You): "a saas web app like DnD Beyond"' in finding['reason']
