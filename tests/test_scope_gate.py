"""Deterministic gate behavior: no LLM in these tests - tiebreaker is mocked."""
from vmodel_contracts import IntakeSubmission
from interview_engine.scope_gate import verdict_deterministic
from vmodel_contracts import AppArchetype


def sub(text: str) -> IntakeSubmission:
    return IntakeSubmission(free_text=text, founder_id="t", submission_id="t1")


def test_clear_web_app_accepted():
    v = verdict_deterministic(sub("A SaaS web app where surf schools manage bookings and customers log in"))
    assert v and v.accepted and v.archetype == AppArchetype.SAAS_CRUD
    assert not v.llm_tiebreak_used


def test_native_mobile_declined_politely():
    v = verdict_deterministic(sub("A native iOS app distributed through the app store"))
    assert v and not v.accepted
    assert v.archetype == AppArchetype.NOT_WEB_APP
    assert v.decline_message and "web applications" in v.decline_message


def test_ambiguous_goes_to_tiebreaker():
    assert verdict_deterministic(sub("Something new for surfers")) is None


def test_hardware_declined():
    v = verdict_deterministic(sub("An embedded firmware controller for drone sensors"))
    assert v and not v.accepted


class _StubModel:
    """Records the prompt it was given; answers like phi4 on a good day."""

    def __init__(self):
        self.prompt = None

    class _Resp:
        content = '{"is_web_app": true, "archetype": "saas_crud", "confidence": 0.8, "reason": "web app"}'
        response_metadata = {}
        usage_metadata = {}

    def invoke(self, prompt, *a, **k):
        self.prompt = prompt
        return self._Resp()


def test_tiebreak_prompt_formats_and_reaches_the_model():
    # 2026-09-12 crash: the template's literal JSON example braces were
    # unescaped, so str.format raised KeyError: '"is_web_app"' before the
    # model ever ran. The prompt must render, keep the JSON example visible
    # to the model, and contain the founder's text.
    from interview_engine.scope_gate import TIE_BREAK_PROMPT, verdict_with_llm
    m = _StubModel()
    v = verdict_with_llm(sub("A booking thing, maybe web, maybe not"), m)
    assert v.accepted and v.llm_tiebreak_used
    assert '"is_web_app"' in m.prompt            # the model still sees the JSON example
    assert "maybe web, maybe not" in m.prompt    # the idea text was substituted


def test_tiebreak_usage_logging_uses_config_not_response_metadata(tmp_path):
    # 2026-09-12, second crash on the same resume: response_metadata on the
    # founder's ChatOllama carried NO model_provider key, the scraped value
    # was literally "unknown", and pricing a $0 local call as remote crashed
    # the run. Provider/model must come from the configured route.
    from interview_engine import model_config
    from interview_engine.scope_gate import verdict_with_llm
    from interview_engine.telemetry import UsageLog

    class Bare:
        class _Resp:
            content = '{"is_web_app": true, "archetype": "saas_crud", "confidence": 0.8, "reason": "web"}'
            response_metadata = {}   # the real metadata path: no model_provider key at all
            usage_metadata = {"input_tokens": 10, "output_tokens": 5}

        def invoke(self, prompt, *a, **k):
            return self._Resp()

    log = UsageLog("s1")
    v = verdict_with_llm(sub("A booking thing, maybe web, maybe not"), Bare(), usage_log=log)
    assert v.accepted
    (rec,) = log.records
    cfg = model_config.for_stage("scope_tiebreak")
    assert (rec.provider, rec.model) == (cfg.provider, cfg.model)
    assert rec.cost_usd == 0.0     # the configured local route prices $0
