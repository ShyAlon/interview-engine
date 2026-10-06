import json,sqlite3
from pathlib import Path
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph,END
from typing import TypedDict
from vmodel_contracts import CONTRACT_VERSION,AppArchetype
from vmodel_contracts.facets import COVERAGE_SCHEMA,SessionSchemaBinding
from vmodel_contracts.interactive import InputRequest,InputKind,StartSessionRequest,InputResponse,ProcessStatus
from interview_engine.evidence import EvidenceLake
import importlib.util,sys
spec=importlib.util.spec_from_file_location('binding_operator_script',Path(__file__).parents[1]/'scripts/migrate-session-bindings.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
migrate,inventory,main=module.migrate,module.inventory,module.main
from interview_engine.http_service import SessionRegistry,create_app

class State(TypedDict):
 submission:dict
 messages:list

def fixture(root,sid='cef8ed07-0894-4f4e-bd40-2455fbfd0fa3',task='task-one'):
 session=root/sid/'interview';session.mkdir(parents=True)
 conn=sqlite3.connect(session/'checkpoints.sqlite',check_same_thread=False)
 saver=SqliteSaver(conn,serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]))
 graph=StateGraph(State);graph.add_node('hold',lambda s:s);graph.set_entry_point('hold');graph.add_edge('hold',END);graph=graph.compile(checkpointer=saver)
 graph.invoke({'submission':{'submission_id':sid},'messages':['private founder transcript']},{'configurable':{'thread_id':sid}});conn.close()
 pending=InputRequest(input_id=uuid4(),kind=InputKind.TEXT,prompt='Who uses it?')
 request=StartSessionRequest(session_id=sid,task_id=task,idea='Private idea',model_route='m')
 payload={'session_id':sid,'task_id':task,'phase_id':10,'sequence':7,'status':30,'next_step':20,'open_input':pending.model_dump(mode='json'),'responses':[],'request':request.model_dump(mode='json'),'model_map':None}
 (session/'http-session.json').write_text(json.dumps(payload))
 lake=EvidenceLake(root/'evidence-lake',task);lake.bind_session(sid)
 binding=lake.paths.run/f'session_id={sid}'/'schema-binding.json';body=json.loads(binding.read_text());body['contract_version']='0.19.0';binding.write_text(json.dumps(body))
 (lake.paths.run/'evidence').mkdir();(lake.paths.run/'evidence/events.jsonl').write_text('original evidence\n')
 return binding,payload,pending

def test_dry_run_no_mutations_apply_stamp_only_full_backup_and_repeat(tmp_path):
 path,payload,pending=fixture(tmp_path);before_session=inventory(tmp_path/'cef8ed07-0894-4f4e-bd40-2455fbfd0fa3');before_run=inventory(path.parent.parent)
 preview=migrate(tmp_path,path);assert preview['status']=='ready';assert not (tmp_path/'migration-backups').exists()
 assert inventory(tmp_path/'cef8ed07-0894-4f4e-bd40-2455fbfd0fa3')==before_session and inventory(path.parent.parent)==before_run
 applied=migrate(tmp_path,path,True);assert applied['status']=='applied';backup=Path(applied['backup'])
 assert inventory(backup/'session')==before_session;assert inventory(backup/'evidence-run')==before_run
 assert inventory(tmp_path/'cef8ed07-0894-4f4e-bd40-2455fbfd0fa3')==before_session
 original=json.loads((backup/'evidence-run'/path.relative_to(path.parent.parent)).read_text());new=json.loads(path.read_text());assert {**original,'contract_version':CONTRACT_VERSION}==new
 assert json.loads(Path(applied['audit']).read_text())['status']=='applied'
 assert migrate(tmp_path,path,True)['status']=='current';assert len(list((tmp_path/'migration-backups').iterdir()))==1

def test_old_snapshot_is_typed_nonretryable_and_apply_restores_exact_input(tmp_path):
 path,payload,pending=fixture(tmp_path)
 class Program:
  def __init__(self,sid,request):
   self.model_route=request.model_route;self.finished=False
   self.evidence=EvidenceLake(tmp_path/'evidence-lake',request.task_id);self.binding=self.evidence.bind_session(sid)
  def advance(self,response):self.finished=True;return None
 registry=SessionRegistry(lambda sid,r:Program(sid,r),state_dir=tmp_path)
 client=TestClient(create_app(registry,service_token='test'),headers={'X-Interview-Engine-Token':'test'})
 old=client.get('/v1/sessions/cef8ed07-0894-4f4e-bd40-2455fbfd0fa3');assert old.status_code==422
 detail=old.json()['detail'];assert 'Session needs binding migration' in detail['detail'] and detail['retryable'] is False
 assert '0.19.0' in detail['detail'] and CONTRACT_VERSION in detail['detail']
 assert json.loads(path.read_text())['contract_version']=='0.19.0'
 migrate(tmp_path,path,True)
 restored=client.get('/v1/sessions/cef8ed07-0894-4f4e-bd40-2455fbfd0fa3');assert restored.status_code==200
 assert restored.json()['sequence']==7 and restored.json()['open_input']['input_id']==str(pending.input_id)
 reply=InputResponse(input_id=pending.input_id,expected_sequence=7,value='Founders',actor_id='owner')
 accepted=client.post('/v1/sessions/cef8ed07-0894-4f4e-bd40-2455fbfd0fa3/inputs/'+str(pending.input_id),json=reply.model_dump(mode='json'))
 assert accepted.status_code==200 and accepted.json()['snapshot']['process_status']==ProcessStatus.COMPLETED

def test_real_schema_drift_and_identity_mismatch_are_blocked_without_backup(tmp_path):
 path,payload,pending=fixture(tmp_path);body=json.loads(path.read_text());body['schema_version']='2.0.0';path.write_text(json.dumps(body))
 with pytest.raises(ValueError,match='identity drift'):migrate(tmp_path,path,True)
 body['schema_version']='1.0.0';path.write_text(json.dumps(body));state=tmp_path/'cef8ed07-0894-4f4e-bd40-2455fbfd0fa3/interview/http-session.json';payload['task_id']='other';state.write_text(json.dumps(payload))
 with pytest.raises(ValueError,match='identity mismatch'):migrate(tmp_path,path,True)
 assert not (tmp_path/'migration-backups').exists()

def test_scan_dry_run_and_apply_reports_every_session(tmp_path,monkeypatch,capsys):
 fixture(tmp_path);fixture(tmp_path,'7b0c0da6-c340-4cd2-8a10-0a803104192f','task-two')
 import sys
 monkeypatch.setattr(module,'CONTRACT_VERSION','0.20.0')  # the one-time script refuses other contract versions on purpose
 monkeypatch.setattr(sys,'argv',['migration','--artifacts-root',str(tmp_path),'--all']);assert main()==0
 preview=json.loads(capsys.readouterr().out);assert len(preview['results'])==2 and preview['mode']=='dry-run'
 monkeypatch.setattr(sys,'argv',['migration','--artifacts-root',str(tmp_path),'--all','--apply'])
 with pytest.raises(SystemExit):main()
 monkeypatch.setattr(sys,'argv',['migration','--artifacts-root',str(tmp_path),'--all','--apply','--services-stopped']);assert main()==0
 assert {x['status'] for x in json.loads(capsys.readouterr().out)['results']}=={'applied'}

def test_interrupted_backup_never_rewrites_binding(tmp_path,monkeypatch):
 path,_,_=fixture(tmp_path);original=path.read_bytes()
 monkeypatch.setattr(module.shutil,'copytree',lambda *a,**k:(_ for _ in ()).throw(OSError('disk full')))
 with pytest.raises(OSError):migrate(tmp_path,path,True)
 assert path.read_bytes()==original

def test_missing_checkpoint_unsupported_stamp_and_symlink_are_blocked(tmp_path):
 path,_,_=fixture(tmp_path);body=json.loads(path.read_text());body['contract_version']='0.18.0';path.write_text(json.dumps(body))
 with pytest.raises(ValueError,match='Only the reviewed'):migrate(tmp_path,path,True)
 body['contract_version']='0.19.0';path.write_text(json.dumps(body));db=tmp_path/body['session_id']/'interview/checkpoints.sqlite';db.unlink()
 with pytest.raises(ValueError,match='checkpoint'):migrate(tmp_path,path,True)
 assert not (tmp_path/'migration-backups').exists()

def test_changed_source_during_backup_blocks_before_stamp_write(tmp_path,monkeypatch):
 path,_,_=fixture(tmp_path);original=path.read_bytes();real=module.shutil.copytree
 def changed(source,dest,*args,**kwargs):
  result=real(source,dest,*args,**kwargs)
  if Path(dest).name=='evidence-run':(path.parent.parent/'evidence/events.jsonl').write_text('concurrent write')
  return result
 monkeypatch.setattr(module.shutil,'copytree',changed)
 with pytest.raises(ValueError,match='Sources changed'):migrate(tmp_path,path,True)
 assert path.read_bytes()==original

def test_atomic_interruption_after_stamp_preserves_prepared_audit_and_backup(tmp_path,monkeypatch):
 path,_,_=fixture(tmp_path);before=inventory(tmp_path/'cef8ed07-0894-4f4e-bd40-2455fbfd0fa3');real=module.atomic
 def crash(target,value):
  if target.name=='migration.json' and value.get('status')=='applied':raise OSError('interrupted audit finalize')
  return real(target,value)
 monkeypatch.setattr(module,'atomic',crash)
 with pytest.raises(OSError):migrate(tmp_path,path,True)
 assert json.loads(path.read_text())['contract_version']==CONTRACT_VERSION
 journal=next((tmp_path/'migration-backups').glob('*/migration.json'));assert json.loads(journal.read_text())['status']=='prepared'
 assert inventory(tmp_path/'cef8ed07-0894-4f4e-bd40-2455fbfd0fa3')==before
 assert migrate(tmp_path,path,True)['status']=='current'

def test_true_binding_drift_is_readable_and_never_overridden(tmp_path):
 from interview_engine.binding_compat import BindingIdentityMismatch
 path,_,_=fixture(tmp_path);body=json.loads(path.read_text());body['schema_version']='2.0.0';path.write_text(json.dumps(body));before=path.read_bytes()
 with pytest.raises(BindingIdentityMismatch) as exc:EvidenceLake(tmp_path/'evidence-lake','task-one').bind_session(body['session_id'])
 assert 'Do not run' in exc.value.render();assert path.read_bytes()==before
