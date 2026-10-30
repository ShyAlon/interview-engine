"""Standalone installed-source control. PG journal is authority, never a file."""
import hashlib,json,time
from contextlib import contextmanager
from vmodel_contracts.execution_anchor import ExecutionAnchor

class InstalledControlRouter:
    def __init__(self,pool,base):self.pool,self.base=pool,base

    @contextmanager
    def connection(self):
        from .tenant_worker import _owner,_session
        from .pg_pool import borrow_claimed
        owner,session=_owner.get(),_session.get()
        if not owner or not session:raise PermissionError('verified installed owner/auth scope required')
        with borrow_claimed(self.pool,(owner,session)) as conn,conn.transaction():
            if not conn.execute('SELECT public.requi_session_active() AS active').fetchone()['active']:raise PermissionError('installed owner session revoked')
            yield conn,owner

    def binding(self,conn,owner,sid,*,lock=False):
        lease=conn.execute('SELECT * FROM engine_sessions WHERE session_id=%s',(sid,)).fetchone()
        if lease is None:return None
        if str(lease['owner_id'])!=owner:raise PermissionError('installed owner differs')
        if lease['generation']==1:return None
        task=conn.execute('SELECT * FROM tasks WHERE task_id=%s'+(' FOR UPDATE' if lock else ''),(lease['task_id'],)).fetchone()
        if (not task or task['engine_session_id']!=sid or task.get('deleted_at') is not None
            or task['generation'] not in (lease['generation'],lease['generation']+1)
            or (task['generation']==lease['generation']+1 and task['stop_requested_at'] is None)):
            raise PermissionError('installed task binding changed')
        row=conn.execute('SELECT * FROM installed_execution_bindings WHERE session_id=%s',(sid,)).fetchone()
        if not row:raise PermissionError('installed binding missing')
        plan_row=conn.execute('SELECT * FROM closed_revision_installations WHERE installation_id=%s',(row['installation_id'],)).fetchone()
        event=conn.execute('SELECT * FROM closed_revision_install_events WHERE installation_id=%s ORDER BY sequence DESC LIMIT 1',(row['installation_id'],)).fetchone()
        if not plan_row or not event or event['state']!='installed':raise PermissionError('installed journal unavailable')
        plan=json.loads(plan_row['payload']);payload=json.loads(event['payload'])
        anchor=ExecutionAnchor(str(row['owner_id']),row['task_id'],sid,row['semantic_session_id'],row['generation'],row['lineage_sha256'],row['installation_id'])
        def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (anchor.owner_id!=owner or anchor.task_id!=lease['task_id'] or anchor.generation!=lease['generation']
            or digest(plan)!=plan_row['plan_sha256'] or digest(payload)!=event['event_sha256']
            or payload['plan_sha256']!=plan_row['plan_sha256'] or payload['evidence']['execution_session_id']!=sid
            or payload['evidence']['anchor_sha256']!=row['anchor_sha256'] or plan['destination_anchor']!=anchor.document()
            or hashlib.sha256(json.dumps(anchor.document(),sort_keys=True,indent=2).encode()).hexdigest()!=row['anchor_sha256']):
            raise PermissionError('installed journal evidence differs')
        return {'anchor':anchor,'anchor_sha256':row['anchor_sha256']},task

    def resolve(self,name,sid):
        with self.connection() as (conn,owner):found=self.binding(conn,owner,sid)
        if found is None:return None
        captured=found[0]
        def read(owner,sid):
            with self.connection() as (conn,current):
                if owner!=current:raise PermissionError('installed owner changed')
                found=self.binding(conn,owner,sid)
                if not found or found[0]!=captured:raise PermissionError('installed binding changed')
                return found[0]
        from .installed_execution import InstalledExecutionRegistry
        router=self
        class Fence:
            generation=captured['anchor'].generation
            @contextmanager
            def marker_guard(self):
                # request_stop takes its own task row lock and revalidates there.
                yield
            @property
            def requested(self):
                with router.connection() as (conn,owner):return router.binding(conn,owner,sid)[1]['stop_requested_at'] is not None
            def writer_quiescent(self):
                with router.connection() as (conn,owner):
                    router.binding(conn,owner,sid)
                    lease=conn.execute('SELECT lease_expires_at FROM engine_sessions WHERE session_id=%s',(sid,)).fetchone()
                    if lease['lease_expires_at']>time.time():return False
                    if conn.execute("SELECT 1 FROM worker_jobs WHERE task_id=%s AND generation=%s AND state='running' LIMIT 1",(captured['anchor'].task_id,self.generation)).fetchone():return False
                    key='pg-execution-writer:'+owner+':'+sid
                    held=conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS held',(key,)).fetchone()['held']
                    if held:conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(key,))
                    return held
            @property
            def state(self):return ('STOPPED' if self.writer_quiescent() else 'STOP_REQUESTED') if self.requested else None
            def request_stop(self):
                with router.connection() as (conn,owner):
                    found=router.binding(conn,owner,sid,lock=True)
                    if not found or found[0]!=captured:raise PermissionError('installed binding changed before Stop')
                    if found[1]['stop_requested_at'] is None:
                        conn.execute("UPDATE tasks SET state='stop_requested',generation=generation+1,stop_requested_at=%s,updated_at=%s WHERE task_id=%s",(time.time(),time.time(),captured['anchor'].task_id))
                return self.state
            def confirm_quiescent_after_recovery(self,*,no_live_worker):
                from .stop_fence import StopRequested
                if no_live_worker is not True or not self.writer_quiescent():raise StopRequested('installed worker quiescence unverified')
                if not self.requested:raise ValueError('durable Stop required')
                return 'STOPPED'
        registry=InstalledExecutionRegistry(self.base,read_binding=read,fence_factory=lambda root,anchor:Fence())
        registry.get(sid).validate()
        def snapshot(sid):
            from vmodel_contracts.interactive import EngineSnapshot,ProcessStatus,NextStep
            status=registry.get(sid).fence.state
            return EngineSnapshot(session_id=sid,task_id=captured['anchor'].task_id,sequence=0,
                process_status=ProcessStatus.STOPPED if status=='STOPPED' else ProcessStatus.STOP_REQUESTED if status else ProcessStatus.COMPLETED,
                next_step=NextStep.STOP if status else NextStep.NONE)
        def mutation(sid,*args,**kwargs):
            from .http_service import ProtocolFailure
            from vmodel_contracts.interactive import ProtocolErrorCode
            registry.get(sid).validate()
            raise ProtocolFailure(ProtocolErrorCode.INVALID_RESPONSE,'lineage execution preserves approved interview source; new interview input and route changes are unavailable')
        def stop(sid):registry.stop(sid);return snapshot(sid)
        def reconcile(sid):
            from .stop_fence import StopRequested
            try:registry.reconcile_stopped(sid)
            except StopRequested:pass
            return snapshot(sid)
        def progress(sid,*args,**kwargs):registry.get(sid).validate();raise KeyError(sid)
        return {'get':lambda sid,**kwargs:registry.get(sid),'snapshot':snapshot,'respond':mutation,'switch_route':mutation,'stop':stop,'reconcile_stopped':reconcile,'progress':progress}[name]


class ScopedInstalledRegistry:
    """Single-owner standalone adapter, with scope from configured auth identity."""
    def __init__(self,registry,pool,base,owner,auth_session):
        self.registry=registry;self.router=InstalledControlRouter(pool,base)
        self.owner,self.auth_session=owner,auth_session

    def __getattr__(self,name):
        attribute=getattr(self.registry,name)
        if not callable(attribute):return attribute
        def scoped(*args,**kwargs):
            from .tenant_worker import _owner,_session
            ot=_owner.set(self.owner);st=_session.set(self.auth_session)
            try:
                if name in ('get','snapshot','respond','switch_route','stop','reconcile_stopped','progress'):
                    sid=args[0] if args else kwargs.get('session_id')
                    target=self.router.resolve(name,sid)
                    if target is not None:return target(*args,**kwargs)
                return attribute(*args,**kwargs)
            finally:_session.reset(st);_owner.reset(ot)
        return scoped
