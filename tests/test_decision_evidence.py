from pathlib import Path
from uuid import uuid4
import json
import pytest
from langchain_core.messages import HumanMessage
from vmodel_contracts.interactive import InputRequest,InputKind,InputOption,InputResponse
from interview_engine.application import InterviewApplication
from tests.test_conops_stage import draft

def application(tmp_path):
 app=InterviewApplication.__new__(InterviewApplication);app.session_id='review-test';app.outdir=tmp_path;app.finished=False
 app.result={'messages':[HumanMessage(content='Habit app')],'conops_draft':draft()};return app

def test_code_owned_concept_evidence_is_bounded_and_content_addressed(tmp_path):
 app=application(tmp_path);first=app._decision_context('conops_concept')
 assert first['decision_purpose']=='conops_concept';assert first['session_id']=='review-test'
 assert first['evidence']['conops_draft']['operational_scenarios'][0]['name']=='Log habit'
 assert app._decision_context('conops_concept')['evidence_revision']==first['evidence_revision']
 app.result['conops_draft']['concept_options'][0]['summary']='Different proposal'
 assert app._decision_context('conops_concept')['evidence_revision']!=first['evidence_revision']

def test_invalid_provenance_cannot_be_shown_as_valid_review(tmp_path):
 app=application(tmp_path);app.result['conops_draft']['stakeholder_context'][0]['source_turns']=[99]
 with pytest.raises(ValueError,match='non-founder'):app._decision_context('conops_concept')

def test_approval_edit_archives_and_regenerates_conops_without_losing_intent(tmp_path):
 app=application(tmp_path);app.cfg={'configurable':{'thread_id':'x'}};app.result['intent_model']={'problem_statement':'Unchanged intent'};app.result['conops']={'old':'review'}
 pending=InputRequest(input_id=uuid4(),kind=InputKind.APPROVAL,prompt='Review',options=[InputOption(value=1,label='Proceed'),InputOption(value=0,label='Edit')]);app.open_input=pending
 class Graph:
  def update_state(self,cfg,value):self.cleared=value
  def invoke(self,state,cfg,**kwargs):
   self.input=state
   return {'messages':[HumanMessage(content='Habit app')],'intent_model':app.result['intent_model'],'conops_draft':draft()}
 app.graph=Graph()
 from contextlib import nullcontext
 app._runtime_context=lambda:nullcontext()
 request=app.advance(InputResponse(input_id=pending.input_id,value=0,rationale='Add offline scenarios',actor_id='test-founder'))
 assert request.kind==InputKind.CHOICE;assert request.context['decision_purpose']=='conops_concept'
 assert app.graph.cleared=={'conops':None,'conops_draft':None,'conops_decision_attempts':0}
 assert app.graph.input['messages'][0].content=='Add offline scenarios'
 history=json.loads((tmp_path/'review-history'/f'{pending.input_id}.json').read_text())
 assert history['intent_model']=={'problem_statement':'Unchanged intent'};assert history['conops']=={'old':'review'}

def test_failed_rework_restores_checkpoint_and_old_review_identity(tmp_path):
 from contextlib import nullcontext
 from types import SimpleNamespace
 app=application(tmp_path);app.cfg={};app.result['intent_model']={'unchanged':1};app.result['conops']={'old':'accepted review'}
 old=HumanMessage(content='Habit app',id='old');app.result['messages']=[old]
 pending=InputRequest(input_id=uuid4(),kind=InputKind.APPROVAL,prompt='Review',options=[InputOption(value=1,label='Proceed'),InputOption(value=0,label='Edit')]);app.open_input=pending
 class Graph:
  updates=[]
  def update_state(self,cfg,value):self.updates.append(value)
  def invoke(self,state,cfg,**kwargs):raise RuntimeError('model failed')
  def get_state(self,cfg):return SimpleNamespace(values={'messages':[old,HumanMessage(content='new guidance',id='new')]})
 app.graph=Graph();app._runtime_context=lambda:nullcontext()
 with pytest.raises(RuntimeError,match='model failed'):app.advance(InputResponse(input_id=pending.input_id,value=0,actor_id='founder'))
 assert app.open_input is pending;assert app.result['conops']=={'old':'accepted review'}
 assert app.graph.updates[-1]['conops']=={'old':'accepted review'}
 assert app.graph.updates[-1]['messages'][0].id=='new';assert app.state is None

def test_rework_also_records_founder_input_resolutions_with_history(tmp_path):
 app=application(tmp_path);app.cfg={'configurable':{'thread_id':'x'}}
 register={'version':1,'session_id':'s','advisories':[],'items':[{'item_id':'FD-style','data_class':'styling','question':'Style?','owner':'founder','status':'open','scope':'UI','source':'Interview','blocks':['UI']}]}
 app.result['intent_model']={'problem_statement':'P','founder_register':register};app.result['conops']={'old':'review'}
 pending=InputRequest(input_id=uuid4(),kind=InputKind.APPROVAL,prompt='Review',options=[InputOption(value=1,label='Proceed'),InputOption(value=0,label='Edit')]);app.open_input=pending
 class Graph:
  def __init__(self):self.updates=[]
  def update_state(self,cfg,value):self.updates.append(value)
  def invoke(self,state,cfg,**kwargs):return {'messages':[HumanMessage(content='Habit app')],'intent_model':app.result['intent_model'],'conops_draft':draft()}
 app.graph=Graph()
 from contextlib import nullcontext
 app._runtime_context=lambda:nullcontext()
 app.advance(InputResponse(input_id=pending.input_id,value=0,rationale='Rework',actor_id='ann',founder_inputs=[{'item_id':'FD-style','status':'decided','decision':'Blue','reason':'Calm'}]))
 item=app.result['intent_model']['founder_register']['items'][0]
 assert (item['status'],item['decision'])==('decided','Blue');assert [h['actor'] for h in item['history']]==['ann']
 assert any('intent_model' in u for u in app.graph.updates)
 assert json.loads((tmp_path/'review-history'/f'{pending.input_id}.json').read_text())['intent_model']['founder_register']['items'][0]['decision']=='Blue'
