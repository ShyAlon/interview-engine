"""Recorded-shape fake benchmark: quality parity before reduced call claims."""
import json
from langchain_core.messages import HumanMessage
from interview_engine.progress import ProgressBus
from tests.test_coverage import FakeModel


def run_case(monkeypatch,scenario_replies,tid='batch-benchmark',count=1):
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
 options=[{'name':'Native','summary':'installed'},{'name':'PWA','summary':'web'},
  {'advantages':['native']},{'disadvantages':['release']},{'advantages':['one deployment']},{'disadvantages':['limits']}]
 model=Capture([json.dumps(x) for x in scenario_replies+options]);models={x:FakeModel([]) for x in ('probing','distillation','scope_tiebreak')};models['conops']=model
 monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:models[stage])
 progress=ProgressBus();events=[];progress.subscribe(events.append)
 graph=build_graph(UsageLog(tid),conops_enabled=True,progress=progress)
 result=graph.invoke({'messages':[HumanMessage(content='runner uses phone offline')],
  'submission':{'free_text':'web app','founder_id':'f','submission_id':tid},'verdict':{'accepted':True},'intent_model':intent},
  {'configurable':{'thread_id':tid}})
 return result,model,events


GOOD={'trigger':'habit done','operational_flow':['open app','tap habit'],'exceptions':['offline']}


def test_nine_scenarios_take_nine_calls_not_twenty_seven(monkeypatch):
 result,model,events=run_case(monkeypatch,[GOOD]*9,count=9)
 assert model.calls==15  # 9 scenarios + unchanged 6 option calls. No paid call.
 scenarios=result['conops_draft']['operational_scenarios'];assert len(scenarios)==9
 for i,row in enumerate(scenarios):
  assert row['trigger']==GOOD['trigger'] and row['operational_flow']==GOOD['operational_flow'] and row['exceptions']==GOOD['exceptions']
  assert row['name']=='Log habit '+str(i) and row['actors']==['runner'] and row['success_outcome']=='points sync' and row['source_turns']==[0]
 from interview_engine.application import durable_progress_signal
 saved=[e for e in events if e.stage=='conops' and durable_progress_signal(e)]
 assert len(saved)<=14
 assert [e.completed for e in saved if e.label=='Operational scenarios']==list(range(1,10))
 assert saved[-1].status=='completed'


def test_partial_batch_repairs_only_missing_field(monkeypatch):
 result,model,events=run_case(monkeypatch,[{k:v for k,v in GOOD.items() if k!='exceptions'},{'exceptions':['offline']}])
 assert model.calls==8
 assert result['conops_draft']['operational_scenarios'][0]['operational_flow']==GOOD['operational_flow']
 assert 'Fill ONLY the `exceptions`' in model.messages[1][1].content


def test_batch_never_rewrites_derived_identity_or_source_turns(monkeypatch):
 hostile={**GOOD,'scenario_id':'forged','name':'forged','actors':['forged'],'success_outcome':'forged','source_turns':[99]}
 result,model,_=run_case(monkeypatch,[hostile])
 row=result['conops_draft']['operational_scenarios'][0]
 assert row['name']=='Log habit 0' and row['actors']==['runner'] and row['source_turns']==[0]
 assert row['success_outcome']=='points sync' and model.calls==7


def test_invalid_peer_gets_targeted_repair_without_regenerating_good_fields(monkeypatch):
 result,model,_=run_case(monkeypatch,[{**GOOD,'trigger':None},{'trigger':'habit done'}])
 row=result['conops_draft']['operational_scenarios'][0]
 assert row['trigger']=='habit done' and row['operational_flow']==GOOD['operational_flow']
 assert model.calls==8 and 'Fill ONLY the `trigger`' in model.messages[1][1].content


def test_batch_depth_retry_remains_targeted(monkeypatch):
 result,model,_=run_case(monkeypatch,[{**GOOD,'operational_flow':['Log habit 0']},{'operational_flow':['open app','tap habit']}])
 assert model.calls==8 and result['conops_draft']['operational_scenarios'][0]['operational_flow']==GOOD['operational_flow']
 assert 'TOO SHALLOW' in model.messages[1][1].content and 'Fill ONLY the `operational_flow`' in model.messages[1][1].content


def test_malformed_batch_has_existing_bounded_contract_repair(monkeypatch):
 result,model,_=run_case(monkeypatch,[[],GOOD])
 assert model.calls==8 and result['conops_draft']['operational_scenarios'][0]['trigger']==GOOD['trigger']
 assert any('previous response failed' in message.content.lower() for message in model.messages[1])


def test_actual_pg_journal_writes_only_validated_item_and_phase_boundaries(monkeypatch,tmp_path):
 from test_pg_live_progress import setup,journal
 from interview_engine.application import durable_progress_signal
 result,model,events=run_case(monkeypatch,[GOOD]*9,count=9,tid='journal-benchmark')
 store,claim=setup(tmp_path);j=journal(store,claim)
 saved=[e for e in events if durable_progress_signal(e)]
 for event in saved:
  j.append(phase_key='conops',stage=event.stage,label=event.label,status=event.status,detail=event.detail,completed=event.completed,total=event.total)
 rows=store.read_journal('live',claim.owner_id,0,100)
 assert len(rows)==len(saved)==14
 assert [r['page']['completed'] for r in rows if r['page']['label']=='Operational scenarios']==list(range(1,10))
 assert rows[-1]['page']['status']=='completed'
 assert store.load_published('live',claim.owner_id)['control'] is None,'progress is not execution authority'
