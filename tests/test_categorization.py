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
            return SimpleNamespace(model='jev-1.13.0',answers={'label':SimpleNamespace(choice='x',confidence=.8,probabilities={'x':.8,'y':.2})})
    fake=SimpleNamespace(Choice=lambda **k:('choice',k),Noul=lambda **k:('noul',k),Score=lambda **k:('score',k))
    monkeypatch.setitem(__import__('sys').modules,'typesafe_sdk',fake)
    result=TypeSafeProvider(Client()).classify(request())
    assert result.model=='jev-1.13.0' and result.answers['label'].value=='x'

def test_slot_mismatch_and_unknown_provider_are_rejected(monkeypatch):
    with pytest.raises(ValueError,match='does not match'):
        ProviderChain('intent',[Provider('rules','x')],lambda _:None).classify(request())
    monkeypatch.setenv('INTERVIEW_ENGINE_CATEGORIZERS','jev,magic')
    with pytest.raises(ValueError,match='unknown categorization'):
        configured_provider_names('scope')
