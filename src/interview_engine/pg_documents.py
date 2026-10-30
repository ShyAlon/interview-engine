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
            task=conn.execute('SELECT engine_session_id,generation,stop_requested_at FROM tasks WHERE task_id=%s FOR UPDATE',(self.task_id,)).fetchone()
            lease=conn.execute('SELECT task_id,generation,lease_epoch,lease_expires_at,placement_instance,control FROM engine_sessions WHERE session_id=%s AND owner_id=%s::uuid FOR UPDATE',(c.session_id,c.owner_id)).fetchone()
            if (not task or task['engine_session_id']!=c.session_id or task['generation']!=c.generation or task['stop_requested_at'] is not None
                or not lease or lease['task_id']!=self.task_id or lease['generation']!=c.generation or lease['lease_epoch']!=c.epoch
                or lease['lease_expires_at']<=time.time() or lease['placement_instance']!=c.instance or (lease['control'] or {}).get('stopped')):
                raise StopRequested('interview document authority lost')
            # Installed executions preserve approved source and never construct a new interview.
            if c.generation!=1:raise StopRequested('installed interview source is read-only')
            yield conn

    def _read(self,conn,key):
        row=conn.execute('SELECT payload,sha256 FROM generator_documents WHERE execution_session_id=%s AND generation=%s AND document_key=%s',(self.claim.session_id,self.claim.generation,key)).fetchone()
        if row is None or row['payload'] is None:return None
        raw=bytes(row['payload'])
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('interview document hash differs')
        return raw

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
        with self.transaction() as conn:self._put(conn,key,(self._read(conn,key) or b'')+raw)

    def bind(self,key,raw):
        key=self.key(key)
        with self.transaction() as conn:
            prior=self._read(conn,key)
            if prior is not None and prior!=raw:raise ValueError('interview schema binding changed')
            if prior is None:self._put(conn,key,raw)
