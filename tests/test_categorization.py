from types import SimpleNamespace
import pytest
from interview_engine.categorization import *

class Provider:
    def __init__(self,name,value=None,error=None):self.name,self.value,self.error=name,value,error
    def classify(self,request):
        if self.error:raise self.error
        return ProviderResult({'label':ClassificationAnswer(self.value,.9,{'x':.9})},self.name,'model')

def request():return ClassificationRequest('scope','idea',{'label':ChoiceQuestion('choose',{'x':'X','y':'Y'})})

def test_chain_falls_back_and_audits_actual_decider():
    audit=[];chain=ProviderChain('scope',[Provider('typesafe',error=TimeoutError()),Provider('llm','x')],audit.append)
    assert chain.classify(request()).answers['label'].value=='x'
    assert audit[0].provider=='llm' and audit[0].decider=='llm'
    assert audit[0].fallback_path==('typesafe','llm')

def test_chain_fails_loudly_after_all_providers():
    with pytest.raises(RuntimeError,match='all providers failed'):
        ProviderChain('scope',[Provider('rules',error=ValueError('bad'))],lambda _:None).classify(request())

def test_chain_error_names_the_admission_check_that_fired():
    # An admission refusal is actionable by the user, so the chain surfaces it as
    # a UserFacingError: the typed 422 path carries what happened, every
    # provider's detail (including the check that fired) and the next action.
    # Before 2026-10-27 the chain raised a bare RuntimeError and the UI showed
    # only "engine program failed during start: RuntimeError".
    from interview_engine.errors import UserFacingError
    denial=UserFacingError("stage 'scope' was stopped by admission/accounting checks.",
        "unresolved compute accounting requires review",
        "preserve the run evidence and review its ledger; do not retry or blame connectivity.")
    chain=ProviderChain('scope',[Provider('llm',error=denial)],lambda _:None)
    with pytest.raises(UserFacingError) as excinfo:
        chain.classify(request())
    rendered=excinfo.value.render()
    assert 'all providers failed for scope' in rendered
    assert 'unresolved compute accounting requires review' in rendered
    assert 'preserve the run evidence' in rendered

def test_a_mixed_chain_surfaces_the_user_facing_failure_and_the_other_detail():
    # The rules fallback failing after a user-facing model refusal must not
    # demote the failure to a bare RuntimeError: the render names both
    # providers and keeps the actionable next step.
    from interview_engine.errors import UserFacingError
    denial=UserFacingError("stage 'scope' was stopped by admission/accounting checks.","bound unreviewed","review the ledger.")
    chain=ProviderChain('scope',[Provider('llm',error=denial),Provider('rules',error=ValueError('signals silent'))],lambda _:None)
    with pytest.raises(UserFacingError) as excinfo:
        chain.classify(request())
    rendered=excinfo.value.render()
    assert 'llm' in rendered and 'rules' in rendered and 'signals silent' in rendered
    assert 'review the ledger.' in rendered

def test_a_chain_of_only_plain_bugs_still_raises_runtimeerror():
    # Genuine bugs keep the minimal-disclosure 503 (operator header) path.
    chain=ProviderChain('scope',[Provider('llm',error=TimeoutError()),Provider('rules',error=ValueError('bad'))],lambda _:None)
    with pytest.raises(RuntimeError,match='all providers failed'):
        chain.classify(request())

def test_registry_is_global_or_per_slot_config_not_call_site(monkeypatch):
    monkeypatch.setenv('INTERVIEW_ENGINE_CATEGORIZERS','jev,llm,rules')
    monkeypatch.setenv('INTERVIEW_ENGINE_CATEGORIZER_SCOPE','rules')
    assert configured_provider_names('scope')==('rules',)
    assert configured_provider_names('coverage')==('jev','llm','rules')

def test_jev_without_key_degrades_during_registry_build(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False);monkeypatch.setenv('INTERVIEW_ENGINE_CATEGORIZERS','jev,rules')
    rules=Provider('rules','x');cat=build_categorizer({'scope':{'rules':rules}},lambda _:None)
    assert cat.classify(request()).provider=='rules'

def test_typesafe_adapter_maps_public_sdk_contract(monkeypatch):
    class Client:
        def system_one(self,**kwargs):
            assert kwargs['model']=='jev-latest' and 'label' in kwargs['questions']
            return SimpleNamespace(model='jev-1.13.0',usage={'input_tokens':12,'output_tokens':3},answers={'label':SimpleNamespace(choice='x',confidence=.8,probabilities={'x':.8,'y':.2})})
    fake=SimpleNamespace(Choice=lambda **k:('choice',k),Noul=lambda **k:('noul',k),Score=lambda **k:('score',k))
    monkeypatch.setitem(__import__('sys').modules,'typesafe_sdk',fake)
    usage=SimpleNamespace(calls=[],record_unpriced=lambda *a,**k: usage.calls.append((a,k)))
    result=TypeSafeProvider(Client(),usage_log=usage).classify(request())
    assert result.model=='jev-1.13.0' and result.answers['label'].value=='x'
    assert result.input_tokens==12 and result.output_tokens==3 and result.latency_s is not None and result.cost_usd is None
    assert usage.calls[0][0][0:4]==("categorization_scope","typesafe","jev-1.13.0",{"input_tokens":12,"output_tokens":3})

def test_slot_mismatch_and_unknown_provider_are_rejected(monkeypatch):
    with pytest.raises(ValueError,match='does not match'):
        ProviderChain('intent',[Provider('rules','x')],lambda _:None).classify(request())
    monkeypatch.setenv('INTERVIEW_ENGINE_CATEGORIZERS','jev,magic')
    with pytest.raises(ValueError,match='unknown categorization'):
        configured_provider_names('scope')

def test_coverage_request_is_relevance_plus_one_binary_question_per_area():
    request=coverage_request('Who uses it?','Store managers','users')
    assert request.slot=='coverage' and len(request.questions)==1+len(COVERAGE_AREAS)
    assert set(request.questions['relevance'].criteria)=={'addressed','partial','dodged'}
    assert all(f'covers_{area}' in request.questions for area in COVERAGE_AREAS)


def test_intent_request_uses_code_owned_terminal_and_nonterminal_labels():
    request=intent_request('What latency matters?','You decide, move on')
    assert request.slot=='intent' and set(request.questions['intent'].criteria)==set(INTENT_LABELS)

def test_code_accumulates_per_answer_coverage_without_classifier_owned_state():
    from interview_engine.coverage import initial_coverage,COVERED,THIN
    answers={'relevance':ClassificationAnswer('addressed',.9),'covers_users':ClassificationAnswer(.9),
             **{f'covers_{a}':ClassificationAnswer(0.0) for a in COVERAGE_AREAS if a!='users'}}
    out=apply_coverage_answers(initial_coverage(),answers)
    assert out['users']==COVERED and all(v!='covered' for k,v in out.items() if k!='users')
    answers['relevance']=ClassificationAnswer('partial',.8);answers['covers_goal']=ClassificationAnswer(.8)
    assert apply_coverage_answers(initial_coverage(),answers)['goal']==THIN


def test_terminal_intent_requires_high_confidence_but_nonterminal_does_not():
    assert classified_intent({'intent':ClassificationAnswer('enough',.89)})=='answer'
    assert classified_intent({'intent':ClassificationAnswer('enough',.9)})=='enough'
    assert classified_intent({'intent':ClassificationAnswer('delegate',.6)})=='delegate'

def test_graph_does_not_apply_generating_llm_self_report_to_coverage():
    from pathlib import Path
    source=(Path(__file__).parents[1]/'src/interview_engine/graph.py').read_text()
    assert 'if not categorizer.active("coverage")' in source
    assert 'coverage = apply_coverage_answers(coverage, coverage_result.answers)' in source
    assert 'classified_control = classified_intent(intent_result.answers)' in source


def test_configured_key_with_missing_sdk_degrades_to_fallback(monkeypatch):
    import interview_engine.categorization as module
    monkeypatch.setattr(module,"_env",lambda key: "present" if key=="TYPESAFE_API_KEY" else ("jev,rules" if key=="INTERVIEW_ENGINE_CATEGORIZERS" else None))
    monkeypatch.setattr(TypeSafeProvider,"available",staticmethod(lambda:False))
    rules=Provider("rules","x");cat=build_categorizer({"scope":{"rules":rules}},lambda _:None)
    assert cat.classify(request()).provider=="rules"


def test_audit_meters_typesafe_tokens_latency_call_and_unpriced_cost():
    audits=[];result=ProviderResult({'label':ClassificationAnswer('x',.9)},'typesafe','jev-1',12,3,.125,None)
    class Metered:
        name='typesafe'
        def classify(self,_):return result
    ProviderChain('scope',[Metered()],audits.append).classify(request())
    record=audits[0]
    assert record.input_tokens==12 and record.output_tokens==3 and record.latency_s==.125
    assert record.cost_usd is None and record.pricing_status=='unpriced'


def test_typesafe_adapter_records_zero_tokens_when_sdk_omits_usage(monkeypatch):
    class Client:
        def system_one(self,**kwargs):
            return SimpleNamespace(model='jev',answers={'label':SimpleNamespace(choice='x')})
    monkeypatch.setitem(__import__('sys').modules,'typesafe_sdk',SimpleNamespace(Choice=lambda **k:k,Noul=lambda **k:k,Score=lambda **k:k))
    result=TypeSafeProvider(Client()).classify(request())
    assert result.input_tokens==0 and result.output_tokens==0


def test_registry_never_constructs_network_client_from_ambient_key(monkeypatch):
    import interview_engine.categorization as module
    monkeypatch.setattr(module,"_env",lambda key: "ambient" if key=="TYPESAFE_API_KEY" else None)
    monkeypatch.setattr(TypeSafeProvider,"available",staticmethod(lambda:True))
    rules=Provider("rules","x")
    cat=build_categorizer({"scope":{"rules":rules}},lambda _:None)
    assert cat.classify(request()).provider=="rules"


def test_typesafe_client_receives_resolved_key_explicitly(monkeypatch):
    seen={}
    class Client:
        def __init__(self,**kwargs): seen.update(kwargs)
    fake=SimpleNamespace(TypeSafeClient=Client)
    monkeypatch.setitem(__import__('sys').modules,'typesafe_sdk',fake)
    monkeypatch.setattr(TypeSafeProvider,"available",staticmethod(lambda:True))
    TypeSafeProvider(api_key="resolved-test-key")
    assert seen=={"api_key":"resolved-test-key"}


def test_typesafe_observed_rate_estimates_cost_and_labels_it(monkeypatch):
    class Client:
        def system_one(self,**kwargs):
            return SimpleNamespace(model='jev',usage={'input_tokens':9000,'output_tokens':300},answers={'label':SimpleNamespace(choice='x')})
    monkeypatch.setitem(__import__('sys').modules,'typesafe_sdk',SimpleNamespace(Choice=lambda **k:k,Noul=lambda **k:k,Score=lambda **k:k))
    result=TypeSafeProvider(Client(),usd_per_mtok=.0322580645).classify(request())
    assert result.cost_usd==pytest.approx(.0003)
    assert result.pricing_status=='estimated_observed_dashboard'


def test_llm_adapter_normalizes_typed_answer_envelopes_before_membership():
    from interview_engine.categorization_providers import llm_answers
    from interview_engine.telemetry import UsageLog
    class Model:
        def invoke(self,prompt):
            return SimpleNamespace(content='{"relevance":{"choice":"addressed","confidence":0.91},"covers_goal":{"noul":0.8},"_confidence":{"covers_goal":0.7}}',usage_metadata={})
    req=ClassificationRequest('coverage','answer',{
        'relevance':ChoiceQuestion('relevant',{'addressed':'yes','dodged':'no'}),
        'covers_goal':NoulQuestion('goal')})
    answers=llm_answers(req,Model(),UsageLog('s'))
    assert answers['relevance'].value=='addressed' and answers['relevance'].confidence==.91
    assert answers['covers_goal'].value==.8 and answers['covers_goal'].confidence==.7

def test_llm_adapter_rejects_ambiguous_wrapped_answer_cleanly():
    from interview_engine.categorization_providers import llm_answers
    from interview_engine.telemetry import UsageLog
    class Model:
        def invoke(self,prompt):
            return SimpleNamespace(content='{"label":{"choice":"x","value":"x"},"_confidence":{}}',usage_metadata={})
    with pytest.raises(ValueError,match='invalid wrapped label answer keys'):
        llm_answers(ClassificationRequest('scope','x',{'label':ChoiceQuestion('x',{'x':'x'})}),Model(),UsageLog('s'))
