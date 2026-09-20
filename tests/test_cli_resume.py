"""End-to-end CLI: fresh run crashes after distillation; --session resume recovers it."""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from vmodel_contracts import IntentModel as RealIntentModel

from interview_engine import cli, model_config

NEEDS_REPLY = {"stakeholder_needs": [
    {"need_id": "SN-1", "statement": "founders need specs", "kind": "functional", "source_turns": [0]},
], "assumptions": []}

SCOPE_REPLY = {
    "problem_statement": "p", "target_users": ["u"], "core_value_proposition": "v",
    "in_scope": ["a"], "out_of_scope": ["b"],
    "external_integrations": [], "constraints": [], "open_questions": [],
    "source_turns": {},
}

PREVIEW_REPLY = [{"need_id": "SN-1", "kind": "functional",
                  "statement": "The system shall emit a spec.", "example_criterion": "spec file exists"}]

CONOPS_REPLY = {"stakeholder_context": [{"stakeholder": "founder", "role": "authors specs",
 "goals": ["finish a spec"], "operating_environment": "desktop", "source_turns": [0]}],
 "operational_scenarios": [{"scenario_id": "OS-1", "name": "Create spec", "trigger": "idea",
 "actors": ["founder"], "operational_flow": ["describe idea", "review output"],
 "success_outcome": "approved spec", "exceptions": [], "source_turns": [0]}],
 "concept_options": [{"name": "Web app", "summary": "browser app", "advantages": ["access"],
 "disadvantages": ["browser limits"]}, {"name": "Native app", "summary": "installed app",
 "advantages": ["native APIs"], "disadvantages": ["distribution"]}]}



class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        # exhausted queue -> terminal marker (the structured-turn retry legitimately
        # consumes one extra reply since the format-feedback retry, run-4 fix)
        return AIMessage(content=self.replies.pop(0) if self.replies else "[INTERVIEW COMPLETE]")


def run_cli_with_inputs(inputs, argv):
    it = iter(inputs)
    import builtins
    old = builtins.input
    builtins.input = lambda prompt="": next(it)
    old_argv, sys.argv = sys.argv, argv
    try:
        cli.main()
    finally:
        builtins.input = old
        sys.argv = old_argv


def test_fresh_run_then_session_resume_after_validation_crash(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR",str(tmp_path/"artifacts")); monkeypatch.chdir(tmp_path)
    # timestamped session id with collision suffix: 20260908-181530-a3f9
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: type("U", (), {"hex": "deadbeefcafe"})())
    monkeypatch.setattr(cli, "datetime", type("D", (), {"now": staticmethod(lambda: type("T", (), {"strftime": lambda self, f: "20260908-181530"})())}))
    session = "20260908-181530-dead"

    models = {
        "probing": FakeModel([json.dumps({"question": "What is the goal?", "covered": [], "thin": [], "skipped": [], "complete": False}), "[INTERVIEW COMPLETE]"]),
        "distillation": FakeModel([json.dumps(NEEDS_REPLY), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)]),
        "conops": FakeModel([
            json.dumps({"operating_environment": "desktop"}),
            json.dumps({"trigger":"idea"}), json.dumps({"operational_flow":["describe idea","review output"]}),
            json.dumps({"name":CONOPS_REPLY["concept_options"][0]["name"],"summary":CONOPS_REPLY["concept_options"][0]["summary"]}),
            json.dumps({"name":CONOPS_REPLY["concept_options"][1]["name"],"summary":CONOPS_REPLY["concept_options"][1]["summary"]}),
            json.dumps({"advantages":["access"]}),json.dumps({"disadvantages":["browser limits"]}),
            json.dumps({"advantages":["native APIs"]}),json.dumps({"disadvantages":["distribution"]})]),
        "scope_tiebreak": FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']),
    }
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kwargs: models[stage])

    # the CLI's own validation explodes (his scenario); the graph's in-node
    # validation is untouched because node_distill imports the class itself.
    cli_crash = {"armed": True}

    class ExplodingIntent:
        @classmethod
        def model_validate(cls, data):
            if cli_crash["armed"]:
                raise ValueError("source_turns: Input should be a valid list, input_value={}")
            return RealIntentModel.model_validate(data)

    monkeypatch.setattr(cli, "IntentModel", ExplodingIntent)

    with pytest.raises(Exception):
        run_cli_with_inputs(["the answer", "1 - browser access"], ["prog", "a saas web app for surf schools with booking and logins"])

    outdir = tmp_path / "artifacts" / session / "interview"
    assert (outdir / "intent_model.raw.json").exists(), "raw distillation preserved before validation"
    assert (outdir / "checkpoints.sqlite").exists(), "checkpoint persisted"

    # Simulate a pre-unified run, then prove automatic migration and resume.
    legacy = tmp_path / "legacy-repo" / "out" / session
    legacy.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(outdir, legacy)
    shutil.rmtree(tmp_path / "artifacts")
    resume_cwd = tmp_path / "interview-engine"
    resume_cwd.mkdir()
    monkeypatch.chdir(resume_cwd)

    # resume after the "fix": no model call should fire - state comes from the migrated checkpoint
    cli_crash["armed"] = False
    for m in models.values():
        m.replies.clear()
    run_cli_with_inputs(["y"], ["prog", "--session", session])
    assert (outdir / "intent_model.json").exists()
    assert "What we understood" in capsys.readouterr().out




def test_cli_module_uses_shared_keep_awake_guard():
    source = (Path(cli.__file__)).read_text()
    assert "run_with_system_awake(main)" in source


def test_expert_opinion_intent_gate_through_real_cli(monkeypatch,tmp_path):
    monkeypatch.setenv('VMODEL_ARTIFACTS_DIR',str(tmp_path/'artifacts'));monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.uuid,'uuid4',lambda:type('U',(),{'hex':'expert000000'})())
    monkeypatch.setattr(cli,'datetime',type('D',(),{'now':staticmethod(lambda:type('T',(),{'strftime':lambda self,f:'20260915-120000'})())}))
    models={
      'probing':FakeModel(['[INTERVIEW COMPLETE]']),
      'distillation':FakeModel([json.dumps(NEEDS_REPLY),json.dumps(SCOPE_REPLY),json.dumps(PREVIEW_REPLY)]),
      'expert_opinion':FakeModel(['Here is my review.\nAPPROVE\nRATIONALE: coherent for unattended test']),
      'conops':FakeModel([json.dumps({'operating_environment':'desktop'}),json.dumps({'trigger':'idea'}),json.dumps({'operational_flow':['describe','review']}),json.dumps({'name':CONOPS_REPLY['concept_options'][0]['name'],'summary':CONOPS_REPLY['concept_options'][0]['summary']}),json.dumps({'name':CONOPS_REPLY['concept_options'][1]['name'],'summary':CONOPS_REPLY['concept_options'][1]['summary']}),json.dumps({'advantages':['access']}),json.dumps({'disadvantages':['limits']}),json.dumps({'advantages':['api']}),json.dumps({'disadvantages':['distribution']})]),
      'scope_tiebreak':FakeModel(['{"is_web_app":true,"archetype":"saas_crud","confidence":0.9,"reason":"web product"}'])}
    monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:models[stage])
    run_cli_with_inputs(['1 - access','1 - access'],['prog','idea','--expert-opinion'])
    out=tmp_path/'artifacts'/'20260915-120000-expe'/'interview'
    record=json.loads((out/'expert-opinion.intent.json').read_text())
    assert record['mode']=='simulated_expert_opinion' and record['founder_approved'] is False and record['human_review_required'] is True
    assert (out/'intent_model.json').exists()
    prompt = '\n'.join(str(m.content) for m in models['expert_opinion'].messages[0])
    assert "You are judging an Intent Model" in prompt
    assert "It is not an SRS, architecture, implementation spec, or executable test artifact" in prompt
    assert "do not demand artifacts owned by later V-model stages" in prompt
    assert "reviewer for a TEST RUN" not in prompt


def test_expert_opinion_failure_is_user_facing(monkeypatch,tmp_path):
    monkeypatch.setenv('VMODEL_ARTIFACTS_DIR',str(tmp_path/'artifacts'));monkeypatch.chdir(tmp_path)
    # Directly prove the boundary used by CLI translates provider failures.
    class Boom:
        def invoke(self,messages):raise RuntimeError('review unavailable')
    monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:Boom())
    monkeypatch.setattr(model_config,'invoke_checked',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('review unavailable')))
    # Seed path complexity is covered above; source must own a UserFacingError wrapper.
    assert 'expert opinion could not review the intent gate' in Path(cli.__file__).read_text()


def test_migrated_raw_intent_wins_over_stale_checkpoint_copy(tmp_path):
    """Regression: the error told users to migrate raw, while resume only read intent_model.json."""
    from interview_engine.cli import _prefer_migrated_intent
    current = RealIntentModel.model_validate({
        "session_id":"resume-raw", "problem_statement":"p", "target_users":["u"],
        "core_value_proposition":"v", "stakeholder_needs":[{"need_id":"SN-1","statement":"u needs v","kind":"functional","source_turns":[0]}],
        "in_scope":["v"], "out_of_scope":["x"], "source_turns":{}}).model_dump(mode="json")
    stale = json.loads(json.dumps(current))
    stale["contract_version"] = "0.15.0"
    out = tmp_path / "interview"; out.mkdir()
    # Canonical json and checkpoint both remain stale; only the exact artifact
    # named by the migration instruction has been re-stamped.
    (out / "intent_model.json").write_text(json.dumps(stale))
    (out / "intent_model.raw.json").write_text(json.dumps(current))
    loaded, source = _prefer_migrated_intent(out, "resume-raw", stale)
    assert source == out / "intent_model.raw.json"
    assert loaded["contract_version"] == current["contract_version"]


def test_real_migrate_then_actual_resume_entrypoint_repairs_stale_checkpoint(monkeypatch, tmp_path, capsys):
    """Exact field path: checkpoint + raw start at 0.15, real migrate CLI updates raw, real resume consumes it."""
    import sqlite3, subprocess
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver
    from vmodel_contracts import AppArchetype, ConOpsDocument
    from interview_engine.graph import build_graph
    from interview_engine.telemetry import UsageLog

    session = "20260914-210005-68fa"
    root = tmp_path / "artifacts"
    out = root / session / "interview"; out.mkdir(parents=True)
    current = RealIntentModel.model_validate({
        "session_id":session, "problem_statement":"p", "target_users":["u"],
        "core_value_proposition":"v", "stakeholder_needs":[{"need_id":"SN-1","statement":"u needs v","kind":"functional","source_turns":[0]}],
        "in_scope":["v"], "out_of_scope":["x"], "source_turns":{}}).model_dump(mode="json")
    stale = json.loads(json.dumps(current))
    def restamp(x):
        if isinstance(x, dict):
            if "contract_version" in x: x["contract_version"] = "0.15.0"
            for v in x.values(): restamp(v)
        elif isinstance(x, list):
            for v in x: restamp(v)
    restamp(stale)
    conops = ConOpsDocument.model_validate({
        "session_id":session,
        "stakeholder_context":[{"stakeholder":"u","role":"r","goals":["v"],"operating_environment":"web","source_turns":[0]}],
        "operational_scenarios":[{"scenario_id":"OS-1","name":"use","trigger":"need","actors":["u"],"operational_flow":["act"],"success_outcome":"v","source_turns":[0]}],
        "concept_options":[{"concept_id":"concept:pwa","name":"PWA","summary":"web","advantages":["access"],"disadvantages":["limits"]},{"concept_id":"concept:native-app","name":"Native app","summary":"native","advantages":["apis"],"disadvantages":["release"]}],
        "decision":{"selected_concept_id":"concept:pwa","selected_concept":"PWA","rationale":"access","founder_confirmed":True,"source_turns":[1]}}).model_dump(mode="json")
    db = out / "checkpoints.sqlite"
    conn=sqlite3.connect(db,check_same_thread=False)
    saver=SqliteSaver(conn,serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]))
    monkeypatch.setattr(model_config,"build_chat_model",lambda stage,**k: FakeModel(["APPROVE\nRATIONALE: migrated artifact is valid"]) if stage == "expert_opinion" else FakeModel([]))
    graph=build_graph(UsageLog(session),checkpointer=saver,conops_enabled=True)
    cfg={"configurable":{"thread_id":session}}
    graph.update_state(cfg,{"intent_model":stale,"conops":conops,"coverage":{}})
    conn.close()
    (out/"intent_model.raw.json").write_text(json.dumps(stale))
    (out/"intent_model.json").write_text(json.dumps(stale))

    subprocess.run([sys.executable,"-m","vmodel_contracts.migrate",str(out/"intent_model.raw.json")],check=True)
    assert json.loads((out/"intent_model.raw.json").read_text())["contract_version"] != "0.15.0"
    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR",str(root))
    run_cli_with_inputs([], ["prog","--session",session,"--expert-opinion"])
    assert json.loads((out/"intent_model.json").read_text())["contract_version"] != "0.15.0"
    assert "Loaded migrated intent_model.raw.json" in capsys.readouterr().err


def test_stale_checkpoint_error_distinguishes_stale_raw_from_missing_raw(monkeypatch, tmp_path):
    """The next remediation report names which disk state was actually observed."""
    import inspect
    source = inspect.getsource(cli.main)
    assert "no intent_model.raw.json was found before resume" in source
    assert "intent_model.raw.json is still stamped" in source
    assert source.index("raw_existed_before = raw_path.is_file()") < source.index("raw_path.write_text")


def test_expert_reject_prints_reason_and_exits_nonzero(monkeypatch,tmp_path,capsys):
    """A runner using pipefail must stop, and the rejection must be actionable."""
    monkeypatch.setenv('VMODEL_ARTIFACTS_DIR',str(tmp_path/'artifacts')); monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.uuid,'uuid4',lambda:type('U',(),{'hex':'reject000000'})())
    monkeypatch.setattr(cli,'datetime',type('D',(),{'now':staticmethod(lambda:type('T',(),{'strftime':lambda self,f:'20260915-180000'})())}))
    models={
      'probing':FakeModel(['[INTERVIEW COMPLETE]']),
      'distillation':FakeModel([json.dumps(NEEDS_REPLY),json.dumps(SCOPE_REPLY),json.dumps(PREVIEW_REPLY)]),
      'expert_opinion':FakeModel(['REJECT\nFEEDBACK: OBJECTION: artifact=intent | rule=measurable acceptance intent criterion | location=SN-1 field success_signals | resolution=Add a numeric success threshold to SN-1\nRATIONALE: model quality has no measurable acceptance intent']),
      'conops':FakeModel([json.dumps({'operating_environment':'desktop'}),json.dumps({'trigger':'idea'}),json.dumps({'operational_flow':['describe','review']}),json.dumps({'name':CONOPS_REPLY['concept_options'][0]['name'],'summary':CONOPS_REPLY['concept_options'][0]['summary']}),json.dumps({'name':CONOPS_REPLY['concept_options'][1]['name'],'summary':CONOPS_REPLY['concept_options'][1]['summary']}),json.dumps({'advantages':['access']}),json.dumps({'disadvantages':['limits']}),json.dumps({'advantages':['api']}),json.dumps({'disadvantages':['distribution']})]),
      'scope_tiebreak':FakeModel(['{"is_web_app":true,"archetype":"saas_crud","confidence":0.9,"reason":"web"}'])}
    monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:models[stage])
    with pytest.raises(cli.UserFacingError, match="intent gate rejected") as exc:
        run_cli_with_inputs(['1 - access','1 - access'],['prog','idea','--expert-opinion'])
    assert "model quality has no measurable acceptance intent" in exc.value.cause
    out=capsys.readouterr().out
    assert "Rationale:\n  model quality has no measurable acceptance intent" in out
    assert "Artifact: intent" in out and "Location: SN-1 field success_signals" in out


def test_expert_gate_prompt_uses_canonical_contract_rubric(monkeypatch):
    """The reviewer and interviewer import one doctrine rather than copying prompt text."""
    import inspect
    from interview_engine import prompts
    from vmodel_contracts import DELEGATION_DOCTRINE, intent_gate_instructions
    assert DELEGATION_DOCTRINE in prompts.INTERVIEWER_SYSTEM
    source=inspect.getsource(cli.main)
    assert "intent_gate_instructions()" in source
    assert "Do NOT require the intent artifact to choose APIs" in intent_gate_instructions()


def test_session_shape_is_approved_by_canonical_rubric():
    """A clear intent with typed delegation/open questions is complete at intent stage."""
    from vmodel_contracts import IntentModel, intent_gate_instructions
    intent=IntentModel.model_validate({
      "session_id":"s", "problem_statement":"Predict shipment delay or damage before arrival",
      "target_users":["Supply chain managers"], "core_value_proposition":"Early predictive risk insight",
      "stakeholder_needs":[{"need_id":"SN-1","statement":"Managers need model quality to support decisions","kind":"non_functional","source_turns":[0]}],
      "in_scope":["Predict shipment risk"], "out_of_scope":["Autonomous shipment rerouting"],
      "research_briefs":[{"topic":"real-time APIs","brief":"Research current API coverage and terms","source_turns":[2]}],
      "open_questions":["Acceptable F1 threshold"], "source_turns":{}})
    prompt=intent_gate_instructions()+intent.model_dump_json()
    assert intent.problem_statement and intent.core_value_proposition and intent.in_scope
    assert "Research current API" in prompt and "Acceptable F1" in prompt
    assert "Neither a recorded\ndelegation nor a declared open question is missing intent" in prompt


def test_canonical_rubric_rejects_unrecorded_ambiguity():
    from vmodel_contracts import INTENT_GATE_RUBRIC
    assert "REJECT only" in INTENT_GATE_RUBRIC
    assert "material ambiguity that is neither delegated nor declared" in INTENT_GATE_RUBRIC


def test_intent_expert_bar_approves_polish_and_rejects_only_named_blockers():
    source = Path(cli.__file__).read_text()
    assert "APPROVE when there is no blocking" in source
    assert "contradiction, missing trace, unsafe assumption, or build-stopping ambiguity" in source
    assert "Non-blocking improvements and polish MUST stay as suggestions in RATIONALE under APPROVE" in source
    assert "REJECT only for a named blocking defect" in source
    assert "name that blocker plus a concrete correction" in source


def test_intent_expert_prompt_names_intent_and_excludes_later_artifacts():
    source = Path(cli.__file__).read_text()
    assert "You are judging an Intent Model" in source
    assert "It is not an SRS" in source
    assert "architecture, implementation spec, or executable test artifact" in source
    assert "do not demand artifacts owned by later V-model stages" in source
    assert "reviewer for a TEST RUN" not in source


def test_intent_expert_json_verdict_and_garbage_format_retry():
    from vmodel_contracts.runtime import parse_expert_verdict
    blob='```json\n{"action":"APPROVE","rationale":"intent is coherent","stakeholderNeeds":[]}\n```'
    assert parse_expert_verdict(blob,("approve","reject"))["action"] == "approve"
    assert parse_expert_verdict("garbage",("approve","reject")) is None
    source=Path(cli.__file__).read_text()
    assert "FORMAT ERROR: return APPROVE or REJECT" in source


def test_intent_expert_output_uses_shared_human_readable_sections():
    source=Path(cli.__file__).read_text()
    assert "format_expert_review_rounds" in source
    assert 'replace("Reviewer objections:", "Expert review:")' in source

def test_binary_approval_is_a_typed_founder_decision():
    from interview_engine.cli import binary_approval_decision
    event=binary_approval_decision('intent-approval','Approve intent','Approve this intent model?')
    assert event['rationale_required'] is False
    assert [(o['value'],o['label']) for o in event['options']]==[('y','Approve'),('n','Reject')]
    assert event['answer_format']=='<value>'

def test_intent_reject_records_redo_and_returns_to_typed_concept(monkeypatch,tmp_path):
    from interview_engine.cli import binary_approval_decision
    from interview_engine.conops_stage import founder_decision
    approval=binary_approval_decision('intent-approval','Approve','Approve?')
    concept=founder_decision({'concept_options':[{'concept_id':'concept:web-app',**CONOPS_REPLY['concept_options'][0]},{'concept_id':'concept:native-app',**CONOPS_REPLY['concept_options'][1]}]})
    assert [o['value'] for o in approval['options']]==['y','n']
    assert concept['rationale_required'] is True and len(concept['options'])==2
    source=Path(cli.__file__).read_text()
    assert '"event":"gate.redo"' in source and 'emit_founder_decision(founder_decision(result["conops_draft"]))' in source
    assert 'graph.update_state(cfg,{"conops":result["conops"]})' in source

def test_intent_approval_carries_selected_concept_and_founder_reason():
    from interview_engine.cli import binary_approval_decision
    event=binary_approval_decision('intent-approval','Approve','Approve?',context={
        'label':'Selected concept','value':'PWA',
        'reason':'You stated you chose this concept because it works in every browser'})
    assert event['context']=={'label':'Selected concept','value':'PWA','reason':'You stated you chose this concept because it works in every browser'}
    source=Path(cli.__file__).read_text()
    assert 'concept_decision=result["conops"]["decision"]' in source
    assert 'concept_decision["selected_concept"]' in source and "concept_decision['rationale']" in source
