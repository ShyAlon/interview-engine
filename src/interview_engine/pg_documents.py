"""Exact-byte workflow documents under the interview's current PG lease.

No filesystem import. Original claim, owner, task and live auth session fence
both reads and writes. Stop and takeover serialize against the same rows.
"""
import hashlib,re,time
from contextlib import contextmanager

class InterviewDocuments:
    def __init__(self,store,claim,task_id):
        if not store.claims or store.claims[0]!=claim.owner_id:raise PermissionError('real interview owner/auth scope required')
        self.store,self.claim,self.task_id=store,claim,task_id

    @staticmethod
    def key(key):
        if not isinstance(key,str) or len(key)>500 or not key or any(not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',p) or p in ('.','..') for p in key.split('/')):raise ValueError('canonical interview document key required')
        return key

    @contextmanager
    def transaction(self):
        from .stop_fence import StopRequested
        c=self.claim
        with self.store._borrow() as conn,conn.transaction():
            lease=conn.execute("""SELECT t.generation AS task_generation,t.stop_requested_at,
                e.generation,e.lease_epoch,e.lease_expires_at,e.placement_instance,e.control
                FROM tasks t JOIN engine_sessions e ON e.session_id=t.engine_session_id AND e.task_id=t.task_id
                WHERE t.task_id=%s AND e.session_id=%s AND e.owner_id=%s::uuid FOR UPDATE OF t,e""",
                (self.task_id,c.session_id,c.owner_id)).fetchone()
            if (not lease or lease['task_generation']!=c.generation or lease['stop_requested_at'] is not None
                or lease['generation']!=c.generation or lease['lease_epoch']!=c.epoch
                or lease['lease_expires_at']<=time.time() or lease['placement_instance']!=c.instance or (lease['control'] or {}).get('stopped')):
                raise StopRequested('interview document authority lost')
            # Installed executions preserve approved source and never construct a new interview.
            if c.generation!=1:raise StopRequested('installed interview source is read-only')
            yield conn

    def _read(self,conn,key):
        rows=conn.execute('SELECT * FROM public.requi_document_parts(%s,%s,%s)',(self.claim.session_id,self.claim.generation,key)).fetchall()
        if not rows:return None
        if len(rows)!=rows[0]['total']+1 or any(r['ordinal']!=i for i,r in enumerate(rows)):raise ValueError('interview chunk sequence differs')
        parts=[]
        for row in rows:
            if row['payload'] is None:
                if row['ordinal']==0 and row['total']==0:return None
                if row['ordinal']==0:continue
                raise ValueError('interview chunk payload missing')
            raw=bytes(row['payload'])
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('interview document hash differs')
            parts.append(raw)
        return b''.join(parts)

    def read(self,key):
        key=self.key(key)
        with self.transaction() as conn:return self._read(conn,key)

    def _put(self,conn,key,raw):
        c=self.claim
        if raw is not None and (type(raw) is not bytes or len(raw)>64*1024*1024):raise ValueError('bounded interview document bytes required')
        sha=hashlib.sha256(raw).hexdigest() if raw is not None else None
        conn.execute('INSERT INTO generator_documents(execution_session_id,owner_id,auth_session_id,generation,document_key,payload,sha256,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(execution_session_id,document_key) DO UPDATE SET auth_session_id=EXCLUDED.auth_session_id,payload=EXCLUDED.payload,sha256=EXCLUDED.sha256,updated_at=EXCLUDED.updated_at',
            (c.session_id,c.owner_id,self.store.claims[1],c.generation,key,raw,sha,time.time()))

    def write_many(self,values):
        values={self.key(k):v for k,v in values.items()}
        with self.transaction() as conn:
            for key,raw in values.items():
                prior=self._read(conn,key);self._put(conn,key,raw)
                if key.endswith('.json') and prior!=raw:self._put(conn,key[:-5]+'.md',None)

    def write(self,key,raw):return self.write_many({key:raw})

    def append(self,key,raw):
        key=self.key(key)
        if type(raw) is not bytes:raise ValueError('interview append bytes required')
        from .document_diagnostics import queue
        if key.startswith('interview/') and queue(self,'append',key,raw):return
        c=self.claim
        with self.transaction() as conn:conn.execute('SELECT public.requi_document_append(%s,%s,%s,%s,%s,%s,%s)',(c.session_id,c.generation,key,raw,c.owner_id,self.store.claims[1],time.time()))

    def bind(self,key,raw):
        key=self.key(key)
        with self.transaction() as conn:
            prior=self._read(conn,key)
            if prior is not None and prior!=raw:raise ValueError('interview schema binding changed')
            if prior is None:self._put(conn,key,raw)

    def diagnostic_many(self,operations):
        """One final fenced transaction for best-effort answer diagnostics."""
        operations=[(kind,self.key(key),raw) for kind,key,raw in operations]
        if any(kind not in ('append','write') or not key.startswith('interview/') or type(raw) is not bytes for kind,key,raw in operations):
            raise ValueError('interview diagnostic operations required')
        c=self.claim
        with self.transaction() as conn:
            # One pipeline for the payload writes. The single transaction guard
            # remains the same owner/task/Stop/lease check as normal documents.
            with conn.pipeline():
                for kind,key,raw in operations:
                    if kind=='append':conn.execute('SELECT public.requi_document_append(%s,%s,%s,%s,%s,%s,%s)',
                        (c.session_id,c.generation,key,raw,c.owner_id,self.store.claims[1],time.time()))
                    else:self._put(conn,key,raw)
