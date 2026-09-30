"""Offline, explicit 0.19 -> 0.20 binding migration. Never rewrite on reads."""
from __future__ import annotations
import argparse,hashlib,json,os,re,shutil,sqlite3,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
from vmodel_contracts import CONTRACT_VERSION,AppArchetype
from vmodel_contracts.facets import COVERAGE_SCHEMA,SessionSchemaBinding
from vmodel_contracts.interactive import InputRequest,InputResponse,StartSessionRequest
from interview_engine.binding_compat import identity


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def inventory(folder):
    if folder.is_symlink() or not folder.is_dir():raise ValueError('Missing or unsafe source directory: '+str(folder))
    files={}
    for p in folder.rglob('*'):
        if p.is_symlink():raise ValueError('Symlink source is not migratable: '+str(p))
        if p.is_file():files[str(p.relative_to(folder))]=digest(p)
    return files

def validate(root,path):
    for candidate in (root/'evidence-lake',path.parent.parent,path.parent,path):
        if candidate.is_symlink():raise ValueError('Unsafe binding path')
    if path.resolve().parent.parent.parent!=root/'evidence-lake':raise ValueError('Binding is outside the evidence lake')
    raw=json.loads(path.read_text());saved=SessionSchemaBinding.model_validate(raw)
    sid=saved.session_id
    if not re.fullmatch(r'[A-Za-z0-9._-]+',sid) or sid in ('.','..'):raise ValueError('Invalid session identity')
    if path.parent.name!='session_id='+sid:raise ValueError('Binding path/session mismatch')
    run=path.parent.parent;task=run.name.removeprefix('run_id=')
    expected=SessionSchemaBinding(session_id=sid,schema_id=COVERAGE_SCHEMA.schema_id,schema_version=COVERAGE_SCHEMA.schema_version)
    if identity(saved)!=identity(expected):raise ValueError('Coverage identity drift requires a separate reviewed schema migration')
    if saved.contract_version not in ('0.19.0',CONTRACT_VERSION):raise ValueError('Only the reviewed 0.19.0 -> '+CONTRACT_VERSION+' stamp migration is supported')
    session=root/sid
    if session.is_symlink() or (session/'interview').is_symlink():raise ValueError('Unsafe session path')
    state=session/'interview/http-session.json';db=session/'interview/checkpoints.sqlite'
    if state.is_symlink() or db.is_symlink() or not state.is_file() or not db.is_file():raise ValueError('Missing or unsafe durable HTTP session/checkpoint')
    payload=json.loads(state.read_text());request=StartSessionRequest.model_validate(payload['request'])
    if payload['session_id']!=sid or payload['task_id']!=task or request.task_id!=task:raise ValueError('HTTP session/run identity mismatch')
    if request.session_id is not None and str(request.session_id)!=sid:raise ValueError('Request session identity mismatch')
    pending=InputRequest.model_validate(payload['open_input']) if payload.get('open_input') else None
    for response in payload.get('responses',[]):InputResponse.model_validate(response)
    if pending and any(str(x['input_id'])==str(pending.input_id) for x in payload.get('responses',[])):raise ValueError('Open input is already recorded as accepted')
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    # Validate a temporary checkpoint copy: even a read-only SQLite WAL read can
    # create sidecars on the source, so dry-run must never open the original.
    with tempfile.TemporaryDirectory(prefix='binding-preflight-') as directory:
        copy=Path(directory)/db.name
        for source in (db,db.with_name(db.name+'-wal'),db.with_name(db.name+'-shm')):
            if source.exists():
                if source.is_symlink():raise ValueError('Unsafe checkpoint sidecar')
                shutil.copy2(source,Path(directory)/source.name)
        conn=sqlite3.connect(str(copy))
        saver=SqliteSaver(conn,serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]));saver.is_setup=True
        try:checkpoint=saver.get_tuple({'configurable':{'thread_id':sid}})
        finally:conn.close()
    if not checkpoint:raise ValueError('No checkpoint for this session thread')
    values=checkpoint.checkpoint.get('channel_values',{})
    submission=values.get('submission') or {}
    if str(submission.get('submission_id'))!=sid:raise ValueError('Checkpoint submission identity mismatch')
    for key in ('intent_model','conops'):
        if values.get(key) and values[key].get('session_id')!=sid:raise ValueError('Checkpoint artifact session mismatch')
    if pending:
        context=pending.context
        if context.get('session_id') and context['session_id']!=sid:raise ValueError('Open input evidence session mismatch')
        purpose=context.get('decision_purpose')
        if purpose=='semantic_review' and context.get('evidence',{}).get('semantic_review')!=values.get('semantic_review'):raise ValueError('Open semantic evidence differs from checkpoint')
        if purpose in ('conops_concept','interview_approval'):
            evidence=context.get('evidence',{})
            keys=('conops_draft',) if purpose=='conops_concept' else ('intent_model','conops')
            if any(evidence.get(k)!=values.get(k) for k in keys):raise ValueError('Open decision evidence differs from checkpoint')
    return saved,expected,session,run

def migrate(root,path,apply=False):
    saved,current,session,run=validate(root,path)
    result={'session_id':saved.session_id,'task_id':run.name.removeprefix('run_id='),'binding':str(path),'saved_version':saved.contract_version,'target_version':current.contract_version,'status':'current' if saved.contract_version==current.contract_version else 'ready','changes':['schema-binding.json: contract_version only'] if saved.contract_version!=current.contract_version else []}
    if not apply or result['status']=='current':return result
    before={'session':inventory(session),'evidence_run':inventory(run)}
    backup=root/'migration-backups'/('binding-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex)
    backup.mkdir(parents=True);shutil.copytree(session,backup/'session');shutil.copytree(run,backup/'evidence-run')
    if inventory(backup/'session')!=before['session'] or inventory(backup/'evidence-run')!=before['evidence_run']:raise ValueError('Backup checksum verification failed; original binding untouched')
    if inventory(session)!=before['session'] or inventory(run)!=before['evidence_run']:raise ValueError('Sources changed while backing up; stop services and retry')
    audit={**result,'status':'prepared','created_at':datetime.now(timezone.utc).isoformat(),'backup':str(backup),'before_sha256':before,'original_binding':json.loads(path.read_text())}
    journal=backup/'migration.json';atomic(journal,audit)
    # Only the explicit binding stamp changes. All evidence/checkpoint bytes stay intact.
    body=json.loads(path.read_text());body['contract_version']=current.contract_version;atomic(path,body)
    if inventory(session)!=before['session']:raise ValueError('Session changed during migration; inspect backup before continuing')
    after=inventory(run);expected=dict(before['evidence_run']);expected[str(path.relative_to(run))]=digest(path)
    if after!=expected:raise ValueError('Evidence changed during migration; inspect backup before continuing')
    audit['status']='applied';audit['after_binding_sha256']=digest(path);atomic(journal,audit)
    return {**result,'status':'applied','backup':str(backup),'audit':str(journal)}

def atomic(path,value):
    temporary=path.with_name(path.name+'.migration-'+uuid4().hex+'.tmp')
    with temporary.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--artifacts-root',type=Path,required=True)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--session');group.add_argument('--all',action='store_true')
    ap.add_argument('--apply',action='store_true');ap.add_argument('--services-stopped',action='store_true',help='Confirm all services using this artifacts root are stopped before writing')
    args=ap.parse_args()
    if args.apply and not args.services_stopped:ap.error('--apply requires --services-stopped; stop services first')
    if CONTRACT_VERSION!='0.20.0':ap.error('This script is reviewed only for contracts 0.19.0 -> 0.20.0')
    root=args.artifacts_root.expanduser().resolve();paths=sorted((root/'evidence-lake').glob('run_id=*/session_id=*/schema-binding.json'))
    if args.session:paths=[p for p in paths if p.parent.name=='session_id='+args.session]
    if not paths:ap.error('No matching binding files found')
    results=[];failed=False
    import fcntl
    lockfile=None
    try:
        if args.apply:
            lockfile=(root/'.binding-migration.lock').open('a')
            try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:ap.error('Another binding migration is already running')
        for path in paths:
            try:results.append(migrate(root,path,args.apply))
            except Exception as exc:failed=True;results.append({'binding':str(path),'status':'blocked','reason':str(exc)})
    finally:
        if lockfile:lockfile.close()
    print(json.dumps({'mode':'apply' if args.apply else 'dry-run','results':results},indent=2));return 1 if failed else 0

if __name__=='__main__':sys.exit(main())
