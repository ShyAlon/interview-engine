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

    def invoke(self, messages):
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
            json.dumps([{"name":o["name"],"summary":o["summary"]} for o in CONOPS_REPLY["concept_options"]]),
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
