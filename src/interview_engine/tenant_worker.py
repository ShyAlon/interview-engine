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


class TenantRegistryRouter:
    def __init__(self,registry_type,factory,base:Path):
        self.registry_type,self.factory,self.base=registry_type,factory,base.resolve()
        self._registries={}
        self._lock=RLock()

    def _registry(self):
        owner=_owner.get()
        if owner is None:raise PermissionError('worker owner scope required')
        root=worker_root(self.base)
        with self._lock:
            if owner not in self._registries:
                self._registries[owner]=self.registry_type(self.factory,state_dir=root)
            return self._registries[owner]

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
