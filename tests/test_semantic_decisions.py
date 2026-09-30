import json
from contextlib import nullcontext
from uuid import uuid4
import pytest
from langchain_core.messages import HumanMessage
from vmodel_contracts.interactive import InputRequest,InputKind,InputResponse
from vmodel_contracts.semantic_review import make_review
from interview_engine.application import InterviewApplication
from test_progress_resume import VALID_INTENT
from tests.test_conops_stage import draft

def app(tmp_path):
 a=InterviewApplication.__new__(InterviewApplication);a.session_id='t1';a.cfg={};a.outdir=tmp_path;a.finished=False
 candidate={**VALID_INTENT,'assumptions':['unsupported assertion']}
 review=make_review(candidate['session_id'],'intent',candidate,[{'finding_id':'f1','item_id':'a1','path':['assumptions',0],'rule':'missing_provenance','reason':'No source','original':'unsupported assertion'}])
 a.result={'intent_model':candidate,'semantic_review':review.model_dump(mode='json'),'semantic_review_resolved':False,'messages':[HumanMessage(content='Habit app')]}
 a.open_input=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Review',context={'decision_purpose':'semantic_review'})
 class Graph:
  def update_state(self,cfg,value):self.updated=value
  def invoke(self,state,cfg):return {**a.result,'conops_draft':draft()}
 a.graph=Graph();a._runtime_context=lambda:nullcontext();return a,review

def test_remove_resumes_conops_with_valid_intent(tmp_path):
 a,r=app(tmp_path);pending=a.open_input
 payload={'revision':r.revision,'decisions':[{'item_id':'a1','action':'remove','actor':'founder'}]}
 result=a.advance(InputResponse(input_id=pending.input_id,value=json.dumps(payload),actor_id='founder'))
 assert result.context['decision_purpose']=='conops_concept'
 assert a.graph.updated['intent_model']['assumptions']==[]
 assert json.loads((tmp_path/'semantic-review.json').read_text())['decisions'][0]['action']=='remove'

def test_actor_and_revision_and_incomplete_choices_are_rejected(tmp_path):
 a,r=app(tmp_path)
 for payload in [{'revision':'wrong','decisions':[]},{'revision':r.revision,'decisions':[]},{'revision':r.revision,'decisions':[{'item_id':'a1','action':'accept','actor':'another'}]}]:
  with pytest.raises(ValueError):a.advance(InputResponse(input_id=a.open_input.input_id,value=json.dumps(payload),actor_id='founder'))
 assert not hasattr(a.graph,'updated')
