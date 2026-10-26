"""Explicit trusted control-plane owner+session scope for private worker storage.

Service-token authentication is still mandatory. Header owner and session scope
are supplied only by the authorized control plane, never by a public
founder/browser request. Cloud mode rejects missing owner or session scope
rather than sharing the legacy workspace or minting synthetic session
authority: the database placement policies require a live auth session whose id
and user match the connection claims, so a synthetic value can only fail loud.
"""
from contextvars import ContextVar
from pathlib import Path
from uuid import UUID
from threading import RLock
import logging
import os
import time

log=logging.getLogger("interview_engine.tenant_worker")

_owner=ContextVar('interview_worker_owner',default=None)
_session=ContextVar('interview_worker_session',default=None)
_root=ContextVar('interview_worker_root',default=None)


def current_owner():return _owner.get()



def worker_root(default:Path)->Path:
    return _root.get() or default


def pg_session_store(pool,owner:str,session_id:str,*,close_pool:bool=False):
    """A database-backed session store over the process connection pool.

    Fails clear at startup when the database is unreachable or the delivery-api
    placement migration has not run. The store pins NO connection: every
    operation borrows from the pool (pg_pool).

    Tenant claims ride every transaction (pg_claims), never the conninfo:
    poolers drop libpq startup-option GUCs, so claims in the startup packet
    never reach the backend there. Every engine operation - serving, heartbeat
    and saver - applies the same (owner, session) claims as the first
    statement of each transaction, so the placement tables' row security
    admits exactly this owner's rows, and only while the named auth session
    stays live."""
    from .pg_session_store import PgSessionStore
    from .pg_pool import borrow_claimed
    try:
        with borrow_claimed(pool,(owner,session_id)) as conn:
            conn.execute("SELECT 1 FROM engine_sessions LIMIT 1")
            conn.commit()
    except Exception as exc:
        from .pg_diagnostics import tag_stage
        import psycopg.errors
        if isinstance(exc, psycopg.errors.UndefinedTable):
            err=RuntimeError('database session persistence configured but the delivery-api placement migration '
                             '20261022090000_session_placement_leases.sql has not run: engine_sessions is missing')
        else:
            err=RuntimeError('database session persistence configured but the database is unreachable: '+type(exc).__name__)
        raise tag_stage(err,'startup connect',None) from exc
    return PgSessionStore(pool,claims=(owner,session_id),close_pool=close_pool)


class TenantRegistryRouter:
    """One registry per (owner, auth session): parallel logins coexist and
    share ONE process pool, because claims ride each checked-out transaction
    (pg_claims) instead of pinned connections. Idle registries cost no
    connections, but each still runs a heartbeat loop while it holds leases,
    so the map stays bounded: a registry becomes evictable only when it owns
    no active work (no held lease and no turn in flight) and has been idle
    past the TTL; eviction waits out in-flight turns and releases leases
    through the registry's own close(). At the cap with nothing evictable,
    new scope requests are rejected rather than closing anyone's active
    interview."""

    def __init__(self,registry_type,factory,base:Path,*,session_dsn:str|None=None,allow_file_sessions:bool=False,pool=None):
        self.registry_type,self.factory,self.base=registry_type,factory,base.resolve()
        self._session_dsn=session_dsn
        # One pool per process/DSN shared by every tenant registry: claims are
        # applied per checkout (pg_claims), so one pool serves all tenants
        # without cross-tenant bleed and the connection budget is one number.
        # Hosted in-process (delivery-api) the host INJECTS its process pool;
        # standalone the router builds its own from session_dsn and owns it.
        self._pool=pool
        self._owns_pool=pool is None
        self._allow_file_sessions=allow_file_sessions
        self._registries={}
        self._used={}
        self._inflight={}
        self._lock=RLock()
        lease_ttl=float(os.environ.get('INTERVIEW_ENGINE_LEASE_TTL','150.0'))
        # Defaults disclose what they protect: the idle TTL retires registries
        # unused for ten lease periods; the cap bounds concurrent tenant
        # scopes, not connections (those are bounded by the pool).
        self._idle_ttl=float(os.environ.get('INTERVIEW_ENGINE_REGISTRY_IDLE_TTL',str(10*lease_ttl)))
        self._cap=int(os.environ.get('INTERVIEW_ENGINE_REGISTRY_CAP','100'))

    @staticmethod
    def _evictable(registry)->bool:
        # Active work = a live lease heartbeat or a turn holding a session
        # write lock. Anything else is terminal or never-started state that a
        # later request reloads from the store on demand.
        for session in list(registry._sessions.values()):
            # Active work = a held lease (beaten by the registry's shared loop)
            # or a turn holding the session write lock.
            if session.pg_claim is not None and not session.pg_stopped:return False
            if not session.write_lock.acquire(blocking=False):return False
            session.write_lock.release()
        return True

    def _evict_idle(self,now:float)->None:
        # LRU order; callers hold the lock. Idle past the TTL first, then any
        # evictable registry when the cap demands room. Active work is never
        # closed here.
        victims=sorted(self._registries,key=lambda k:self._used.get(k,0.0))
        for key in victims:
            registry=self._registries.get(key)
            if registry is None:continue
            if len(self._registries)<self._cap and now-self._used.get(key,now)<self._idle_ttl:continue
            if self._inflight.get(key,0)>0:continue
            if not self._evictable(registry):continue
            del self._registries[key];self._used.pop(key,None);self._inflight.pop(key,None)
            try:registry.close()
            except Exception:pass

    def _registry(self):
        owner=_owner.get()
        if owner is None:raise PermissionError('worker owner scope required')
        session=None
        if self._session_dsn is not None:
            session=_session.get()
            if session is None:raise PermissionError('worker session scope required')
        key=(owner,session)
        root=worker_root(self.base)
        with self._lock:
            self._evict_idle(time.monotonic())
            if key not in self._registries:
                if len(self._registries)>=self._cap:
                    raise RuntimeError('tenant registry capacity reached (%d active scopes); no idle registry can be retired - retry after active interviews settle'%self._cap)
                if self._session_dsn is None:
                    if not self._allow_file_sessions:
                        raise RuntimeError('database session persistence is the default: configure the sessions DSN, or explicitly choose file persistence')
                    self._registries[key]=self.registry_type(self.factory,state_dir=root)
                else:
                    # Database session persistence opts into provider dispatch:
                    # staged responses (PR B) landed, so the documented
                    # constructor block is lifted for exactly this wiring.
                    ttl=float(os.environ.get('INTERVIEW_ENGINE_LEASE_TTL','150.0'))
                    if self._pool is None:
                        from .pg_pool import build_pool
                        self._pool=build_pool(self._session_dsn)  # standalone: router-owned
                    self._registries[key]=self.registry_type(self.factory,
                        session_store=pg_session_store(self._pool,owner,session),owner_id=owner,
                        lease_ttl=ttl,
                        allow_db_provider_dispatch=True)
                    log.info("registry.constructed owner=%s session=%s mode=pg lease_ttl=%s",owner,session,ttl)
                self._inflight[key]=0
            self._used[key]=time.monotonic()
            return self._registries[key]

    def close(self)->None:
        # Shutdown path: close every per-owner registry so held leases are
        # released at committed boundaries instead of expiring, then close the
        # shared pool this router owns.
        with self._lock:
            registries=list(self._registries.values())
            self._registries={};self._used={};self._inflight={}
        for registry in registries:
            registry.close()
        if self._pool is not None and self._owns_pool:
            self._pool.close()
        self._pool=None

    def __getattr__(self,name):
        registry=self._registry()
        key=(_owner.get(),_session.get() if self._session_dsn is not None else None)
        attr=getattr(registry,name)
        if not callable(attr):return attr
        def guarded(*args,**kwargs):
            with self._lock:self._inflight[key]=self._inflight.get(key,0)+1
            try:return attr(*args,**kwargs)
            finally:
                with self._lock:
                    self._inflight[key]=max(self._inflight.get(key,1)-1,0)
                    self._used[key]=time.monotonic()
        return guarded


class TenantWorkerMiddleware:
    def __init__(self,app,*,base,service_token=None,require_session=False):
        self.app,self.base=app,Path(base).resolve()
        self.service_token=service_token or os.environ.get('INTERVIEW_ENGINE_TOKEN','local-interview-engine-token')
        self.require_session=require_session

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        from starlette.responses import JSONResponse
        headers=dict(scope.get('headers',[]))
        import hmac
        supplied=headers.get(b'x-interview-engine-token',b'').decode('latin1')
        if not hmac.compare_digest(supplied,self.service_token):
            return await JSONResponse({'detail':'invalid worker service token'},status_code=401)(scope,receive,send)
        try:
            owner=str(UUID(headers.get(b'x-requi-owner',b'').decode('ascii')))
        except (ValueError,UnicodeError):
            return await JSONResponse({'detail':'worker owner scope required'},status_code=401)(scope,receive,send)
        session=headers.get(b'x-requi-session-id',b'').decode('ascii')
        if session:
            try:session=str(UUID(session))
            except (ValueError,UnicodeError):
                return await JSONResponse({'detail':'worker session scope invalid'},status_code=401)(scope,receive,send)
        elif self.require_session:
            return await JSONResponse({'detail':'worker session scope required'},status_code=401)(scope,receive,send)
        else:
            session=None
        root=self.base/'tenants'/owner/'sessions'
        if root.resolve()!=root:
            return await JSONResponse({'detail':'unsafe tenant workspace'},status_code=403)(scope,receive,send)
        # Reject symlinked persisted files before legacy readers open them.
        if root.exists():
            for directory,dirs,files in os.walk(root,followlinks=False):
                if any((Path(directory)/name).is_symlink() for name in dirs+files):
                    return await JSONResponse({'detail':'unsafe tenant workspace'},status_code=403)(scope,receive,send)
        ot=_owner.set(owner);st=_session.set(session);rt=_root.set(root)
        try:return await self.app(scope,receive,send)
        finally:_root.reset(rt);_session.reset(st);_owner.reset(ot)
