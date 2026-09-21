"""Deterministic gate behavior: no LLM in these tests - tiebreaker is mocked."""
from vmodel_contracts import IntakeSubmission,SCOPE_DECLINE_PREFIX
from interview_engine.scope_gate import verdict_deterministic,verdict_with_categorizer
from vmodel_contracts import AppArchetype


def sub(text: str) -> IntakeSubmission:
    return IntakeSubmission(free_text=text, founder_id="t", submission_id="t1")


def test_clear_web_app_accepted():
    v = verdict_deterministic(sub("A SaaS web app where surf schools manage bookings and customers log in"))
    assert v and v.accepted and v.archetype == AppArchetype.SAAS_CRUD
    assert not v.llm_tiebreak_used


def test_native_mobile_app_accepted():
    v = verdict_deterministic(sub("A native iOS app distributed through the app store"))
    assert v and v.accepted and not v.llm_tiebreak_used


def test_games_on_web_or_mobile_are_accepted():
    for idea in ("A Pokemon battle game for mobile", "A browser game built with Unity"):
        v = verdict_deterministic(sub(idea))
        assert v and v.accepted, idea
        assert "interactive game" in v.deterministic_signals


def test_non_software_businesses_declined_with_canonical_scope():
    from vmodel_contracts import PRODUCT_SCOPE_DEFINITION
    for idea in ("I want to open a bakery business", "Start a logistics company"):
        v = verdict_deterministic(sub(idea))
        assert v and not v.accepted, idea
        assert v.archetype == AppArchetype.NOT_WEB_APP
        assert PRODUCT_SCOPE_DEFINITION in v.decline_message


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
    from vmodel_contracts import PRODUCT_SCOPE_DEFINITION
    assert PRODUCT_SCOPE_DEFINITION in m.prompt


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


def test_tiebreak_unwraps_json_and_validates_shape():
    from interview_engine.scope_gate import verdict_with_llm

    class Wrapped(_StubModel):
        class _Resp:
            content = ('Here is the classification:\n```json\n'
                       '{"is_web_app": false, "archetype": "not_web_app", '
                       '"confidence": 0.75, "reason": "physical product"}\n```')
            response_metadata = {}
            usage_metadata = {}

    v = verdict_with_llm(sub("A new physical product"), Wrapped())
    assert v.accepted is False
    assert v.archetype == AppArchetype.NOT_WEB_APP

def test_provider_neutral_scope_verdict_maps_classification_contract():
    from interview_engine.categorization import ClassificationAnswer,ProviderResult
    class Categorizer:
        def classify(self,request):
            assert request.slot=='scope' and set(request.questions)=={'accepted','archetype'}
            return ProviderResult({'accepted':ClassificationAnswer('accepted',.93),'archetype':ClassificationAnswer('marketplace',.88)},'typesafe','jev-test')
    sub=IntakeSubmission(submission_id='jev-1',founder_id='f',free_text='A marketplace')
    verdict=verdict_with_categorizer(sub,Categorizer())
    assert verdict.accepted is True and verdict.archetype==AppArchetype.MARKETPLACE and verdict.confidence==.88
    assert verdict.llm_tiebreak_used is False


def test_provider_neutral_scope_decline_is_safe_and_templated():
    from interview_engine.categorization import ClassificationAnswer,ProviderResult
    class Categorizer:
        def classify(self,request):return ProviderResult({'accepted':ClassificationAnswer('rejected',.91),'archetype':ClassificationAnswer('not_web_app',.91)},'typesafe','jev-test')
    sub=IntakeSubmission(submission_id='jev-2',founder_id='f',free_text='A bakery')
    verdict=verdict_with_categorizer(sub,Categorizer())
    assert verdict.accepted is False and verdict.decline_message.startswith(SCOPE_DECLINE_PREFIX)
