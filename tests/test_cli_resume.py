"""End-to-end CLI: fresh run crashes after distillation; --session resume recovers it."""
import json
import sys

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
    monkeypatch.chdir(tmp_path)
    # timestamped session id with collision suffix: 20260908-181530-a3f9
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: type("U", (), {"hex": "deadbeefcafe"})())
    monkeypatch.setattr(cli, "datetime", type("D", (), {"now": staticmethod(lambda: type("T", (), {"strftime": lambda self, f: "20260908-181530"})())}))
    session = "20260908-181530-dead"

    models = {
        "probing": FakeModel(["What is the goal?", "[INTERVIEW COMPLETE]"]),
        "distillation": FakeModel([json.dumps(NEEDS_REPLY), json.dumps(SCOPE_REPLY), json.dumps(PREVIEW_REPLY)]),
        "scope_tiebreak": FakeModel(['{"is_web_app": true, "archetype": "saas_crud"}']),
    }
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage: models[stage])

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
        run_cli_with_inputs("the answer".split("\n"), ["prog", "a saas web app for surf schools with booking and logins"])

    outdir = tmp_path / "out" / session
    assert (outdir / "intent_model.raw.json").exists(), "raw distillation preserved before validation"
    assert (outdir / "checkpoints.sqlite").exists(), "checkpoint persisted"

    # resume after the "fix": no model call should fire - state comes from the checkpoint
    cli_crash["armed"] = False
    for m in models.values():
        m.replies.clear()
    run_cli_with_inputs(["y"], ["prog", "--session", session])
    assert (outdir / "intent_model.json").exists()
    assert "What we understood" in capsys.readouterr().out
