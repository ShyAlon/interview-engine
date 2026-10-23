"""Explicit trusted control-plane owner scope for private worker storage.

Service-token authentication is still mandatory. Header owner scope is supplied
only by the authorized control plane, never by a public founder/browser request.
Cloud mode rejects missing owner rather than sharing the legacy workspace.
"""
from contextvars import ContextVar
from pathlib import Path
from uuid import UUID
from threading import RLock
import os

_owner=ContextVar('interview_worker_owner',default=None)
_root=ContextVar('interview_worker_root',default=None)


def current_owner():return _owner.get()


def worker_root(default:Path)->Path:
    return _root.get() or default


def owner_scoped_dsn(dsn:str,owner:str)->str:
    """Tenant claims travel in the conninfo: every engine connection - serving,
    heartbeat and saver-spawned - runs under the same tenant boundary, so the
    placement tables' row security admits exactly this owner's rows. URI and
    keyword conninfo forms both normalize through libpq."""
    from psycopg.conninfo import conninfo_to_dict,make_conninfo
    parts=conninfo_to_dict(dsn)
    options="-c request.jwt.claim.sub=%s -c request.jwt.claim.session_id=%s"%(owner,owner)
    parts['options']=((parts.get('options') or '')+' '+options).strip()
    return make_conninfo(**parts)


def pg_session_store(dsn:str):
    """One serving connection for a database-backed registry; fails clear at
    startup when the database is unreachable or the delivery-api placement
    migration has not run. The store spawns its own heartbeat connections."""
    import psycopg
    import psycopg.rows
    from .pg_session_store import PgSessionStore
    try:
        conn=psycopg.connect(dsn,autocommit=False,row_factory=psycopg.rows.dict_row)
    except Exception as exc:
        raise RuntimeError('database session persistence configured but the database is unreachable: '+type(exc).__name__) from exc
    try:
        conn.execute('SELECT session_id FROM engine_sessions LIMIT 0')
    except psycopg.errors.UndefinedTable:
        conn.close()
        raise RuntimeError('database session persistence configured but the delivery-api migration 20261022090000_session_placement_leases.sql has not run: engine_sessions is missing') from None
    return PgSessionStore(conn)


class TenantRegistryRouter:
    def __init__(self,registry_type,factory,base:Path,*,session_dsn:str|None=None):
        self.registry_type,self.factory,self.base=registry_type,factory,base.resolve()
        self._session_dsn=session_dsn
        self._registries={}
        self._lock=RLock()

    def _registry(self):
        owner=_owner.get()
        if owner is None:raise PermissionError('worker owner scope required')
        root=worker_root(self.base)
        with self._lock:
            if owner not in self._registries:
                if self._session_dsn is None:
                    self._registries[owner]=self.registry_type(self.factory,state_dir=root)
                else:
                    # Database session persistence opts into provider dispatch:
                    # staged responses (PR B) landed, so the documented
                    # constructor block is lifted for exactly this wiring.
                    self._registries[owner]=self.registry_type(self.factory,
                        session_store=pg_session_store(owner_scoped_dsn(self._session_dsn,owner)),owner_id=owner,
                        lease_ttl=float(os.environ.get('INTERVIEW_ENGINE_LEASE_TTL','30.0')),
                        allow_db_provider_dispatch=True)
            return self._registries[owner]

    def close(self)->None:
        # Shutdown path: close every per-owner registry so held leases are
        # released at committed boundaries instead of expiring.
        with self._lock:
            registries=list(self._registries.values())
            self._registries={}
        for registry in registries:
            registry.close()

    def __getattr__(self,name):return getattr(self._registry(),name)


class TenantWorkerMiddleware:
    def __init__(self,app,*,base,service_token=None):
        self.app,self.base=app,Path(base).resolve()
        self.service_token=service_token or os.environ.get('INTERVIEW_ENGINE_TOKEN','local-interview-engine-token')

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
        root=self.base/'tenants'/owner/'sessions'
        if root.resolve()!=root:
            return await JSONResponse({'detail':'unsafe tenant workspace'},status_code=403)(scope,receive,send)
        # Reject symlinked persisted files before legacy readers open them.
        if root.exists():
            for directory,dirs,files in os.walk(root,followlinks=False):
                if any((Path(directory)/name).is_symlink() for name in dirs+files):
                    return await JSONResponse({'detail':'unsafe tenant workspace'},status_code=403)(scope,receive,send)
        ot=_owner.set(owner);rt=_root.set(root)
        try:return await self.app(scope,receive,send)
        finally:_root.reset(rt);_owner.reset(ot)
