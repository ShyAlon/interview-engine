"""Recorded-shape fake benchmark: quality parity before reduced call claims."""
import json
from langchain_core.messages import HumanMessage
from interview_engine.progress import ProgressBus
from tests.test_coverage import FakeModel


def run_case(monkeypatch,option_replies=None,tid='option-benchmark',count=1):
 from interview_engine import model_config
 from interview_engine.graph import build_graph
 from interview_engine.telemetry import UsageLog
 class Capture(FakeModel):
  def __init__(self,replies):super().__init__(replies);self.messages=[]
  def invoke(self,messages):self.messages.append(messages);return super().invoke(messages)
 intent={'session_id':tid,'problem_statement':'p','target_users':['runner'],'core_value_proposition':'points sync',
  'stakeholder_needs':[{'need_id':'SN-1','statement':'runner needs quick capture','kind':'functional','source_turns':[0]}],
  'in_scope':['Log habit '+str(i) for i in range(count)],'out_of_scope':['native-only'],'external_integrations':[],
  'constraints':['phone, often offline'],'assumptions':[],'assumption_items':[],'scope_exclusions':[],
  'done_criteria':[],'tech_preferences':[],'glossary':{},'operational_concepts':['Log habit '+str(i) for i in range(count)],
  'open_questions':[],'source_turns':{'operational_concepts':[0],'core_value_proposition':[0]}}
 scenario_replies=[{'trigger':'habit done','operational_flow':['open app','tap habit'],'exceptions':['offline']} for _ in range(count)]
 options=option_replies if option_replies is not None else OPTIONS
 model=Capture([json.dumps(x) for x in scenario_replies+options]);models={x:FakeModel([]) for x in ('probing','distillation','scope_tiebreak')};models['conops']=model
 monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:models[stage])
 progress=ProgressBus();events=[];progress.subscribe(events.append)
 graph=build_graph(UsageLog(tid),conops_enabled=True,progress=progress)
 result=graph.invoke({'messages':[HumanMessage(content='runner uses phone offline')],
  'submission':{'free_text':'web app','founder_id':'f','submission_id':tid},'verdict':{'accepted':True},'intent_model':intent},
  {'configurable':{'thread_id':tid}})
 return result,model,events


OPTIONS=[{'name':'Native','summary':'installed','advantages':['native'],'disadvantages':['release']},
 {'name':'PWA','summary':'web','advantages':['one deployment'],'disadvantages':['limits']}]




def test_nine_recorded_shape_scenarios_keep_parity_with_two_option_calls(monkeypatch):
 result,model,events=run_case(monkeypatch,count=9)
 assert model.calls==11 # 9 scenario batches + 2 complete options, no paid call.
 assert len(result['conops_draft']['operational_scenarios'])==9
 options=result['conops_draft']['concept_options']
 for actual,expected in zip(options,OPTIONS):
  for key,value in expected.items():assert actual[key]==value
  from vmodel_contracts import canonical_concept_id
  assert actual['concept_id']==canonical_concept_id(expected['name'])
 assert 'Native' in model.messages[-1][1].content
 assert 'genuinely distinct' in model.messages[-1][1].content


def test_missing_pros_repairs_one_field_not_whole_option(monkeypatch):
 first={k:v for k,v in OPTIONS[0].items() if k!='advantages'}
 result,model,_=run_case(monkeypatch,[first,{'advantages':['native']},OPTIONS[1]])
 assert model.calls==4
 assert 'Fill ONLY the `advantages`' in model.messages[2][1].content
 assert result['conops_draft']['concept_options'][0]['disadvantages']==['release']


def test_invalid_list_peer_repairs_without_overwriting_identity(monkeypatch):
 first={**OPTIONS[0],'advantages':None,'concept_id':'forged','source_turns':[999]}
 result,model,_=run_case(monkeypatch,[first,{'advantages':['native']},OPTIONS[1]])
 option=result['conops_draft']['concept_options'][0]
 assert model.calls==4 and option['name']=='Native' and option['concept_id']!='forged'
 assert option['advantages']==['native'] and option['disadvantages']==['release']
 assert 'source_turns' not in option


def test_duplicate_identity_contract_retries_only_second_option(monkeypatch):
 result,model,_=run_case(monkeypatch,[OPTIONS[0],OPTIONS[0],OPTIONS[1]])
 assert model.calls==4
 assert [o['name'] for o in result['conops_draft']['concept_options']]==['Native','PWA']
 assert any('duplicates' in m.content for m in model.messages[-1])


def test_weak_model_container_echo_keeps_bounded_retry(monkeypatch):
 result,model,_=run_case(monkeypatch,[{'concept_options':OPTIONS},OPTIONS[0],OPTIONS[1]])
 assert model.calls==4
 assert result['conops_draft']['concept_options'][0]['advantages']==['native']


def test_statement_lists_normalize_and_keep_payload_parity(monkeypatch):
 first={**OPTIONS[0],'advantages':[{'statement':'native','source_turns':[0]}]}
 result,model,_=run_case(monkeypatch,[first,OPTIONS[1]])
 assert model.calls==3
 assert result['conops_draft']['concept_options'][0]['advantages']==['native']


def test_bad_identity_fails_after_exactly_one_contract_retry(monkeypatch):
 import pytest
 from interview_engine.errors import UserFacingError
 with pytest.raises(UserFacingError,match='failed its JSON contract after one automatic retry'):
  run_case(monkeypatch,[{'name':{'forged':'name'},'summary':'invalid'}, {'name':None,'summary':'still invalid'}])


def test_empty_tradeoff_repairs_only_that_field(monkeypatch):
 result,model,_=run_case(monkeypatch,[{**OPTIONS[0],'disadvantages':[]},{'disadvantages':['release']},OPTIONS[1]])
 assert model.calls==4
 assert 'Fill ONLY the `disadvantages`' in model.messages[2][1].content
 assert result['conops_draft']['concept_options'][0]['advantages']==['native']
