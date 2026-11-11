"""Session-bound diagnostic documents, never inferred from a path or flag.

The document adapter owns its original lease/auth/Stop transaction. Do not hold
an external publication transaction while it borrows its own connection.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re

_bound=ContextVar('interview_diagnostic_documents',default=None)

@contextmanager
def bind(documents,root):
    root=Path(root)
    if documents is not None and (not root.is_absolute() or root.resolve()!=root):raise ValueError('owned canonical diagnostic root required')
    token=_bound.set((documents,root) if documents is not None else None)
    try:yield
    finally:_bound.reset(token)

def configured():return _bound.get() is not None

def _target(path):
    bound=_bound.get()
    if bound is None:raise ValueError('bound diagnostic documents required')
    documents,root=bound;path=Path(path)
    if path.resolve()!=path or not path.is_relative_to(root):raise ValueError('diagnostic outside owned session')
    key=path.relative_to(root).as_posix()
    if not key or len(key)>500 or any(not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',p) for p in key.split('/')):
        raise ValueError('canonical diagnostic key required')
    return documents,key

def write(path,text):
    documents,key=_target(path);raw=text.encode('utf-8')
    if not queue(documents,'write',key,raw):return documents.write(key,raw)

def append(path,text):
    documents,key=_target(path);raw=text.encode('utf-8')
    if not queue(documents,'append',key,raw):return documents.append(key,raw)

def read(path):
    documents,key=_target(path)
    batch=_answer.get()
    found,raw=batch.read_overlay(documents,key) if batch is not None else (False,None)
    if not found:raw=documents.read(key)
    if raw is None:raise FileNotFoundError(str(path))
    return raw.decode('utf-8')


# Explicit answer-only seam. Outside it, existing document semantics stay unchanged.
_answer=ContextVar('answer_diagnostic_batch',default=None)

class _AnswerBatch:
    def __init__(self,recovery_root,identity):
        import hashlib,json
        self.identity=dict(identity);self.documents=None;self.operations=[]
        self.root=Path(recovery_root)
        if not self.root.is_absolute() or self.root.resolve()!=self.root:raise ValueError('canonical diagnostic recovery root required')
        self.path=self.root/(hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()+'.json')

    def queue(self,documents,kind,key,raw):
        if self.documents is not None and self.documents is not documents:
            raise ValueError('one owned document adapter per answer batch')
        self.documents=documents
        if self.operations and kind=='append' and self.operations[-1][:2]==(kind,key):
            self.operations[-1]=(kind,key,self.operations[-1][2]+raw)
        else:self.operations.append((kind,key,raw))
        self.spool()

    def spool(self):
        import base64,json,os,tempfile,logging
        temporary=None
        try:
            self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
            if self.root.resolve()!=self.root:raise ValueError('recovery root changed')
            fd,temporary=tempfile.mkstemp(prefix='.diagnostic-',dir=self.root)
            with os.fdopen(fd,'w') as stream:
                json.dump({'version':1,'identity':self.identity,'kind':'diagnostic-only-not-execution-replay',
                    'operations':[[kind,key,base64.b64encode(raw).decode('ascii')] for kind,key,raw in self.operations]},stream)
            os.replace(temporary,self.path)
        except Exception:
            logging.getLogger(__name__).warning('diagnostic.spool_failed',exc_info=False)
        finally:
            if temporary is not None:
                try:os.unlink(temporary)
                except OSError:pass

    def read_overlay(self,documents,key):
        result=None;found=False
        for kind,name,raw in self.operations:
            if name!=key:continue
            if kind=='write':result=raw;found=True
            else:
                if not found:result=documents.read(key) or b'';found=True
                result+=raw
        return found,result

    def flush(self):
        import logging
        if not self.operations:return
        try:
            self.documents.diagnostic_many(self.operations)
            self.path.unlink(missing_ok=True)
        except Exception:
            # Diagnostics must never turn success into failure, or mask Stop.
            # The local spool is not permission for paid replay or stale writes.
            logging.getLogger(__name__).warning('diagnostic.flush_failed recovery=local',exc_info=False)

@contextmanager
def answer_batch(recovery_root,identity):
    if _answer.get() is not None:raise ValueError('nested answer diagnostic batch')
    batch=_AnswerBatch(recovery_root,identity);token=_answer.set(batch)
    try:yield batch
    finally:
        _answer.reset(token)
        batch.flush()


def queue(documents,kind,key,raw):
    batch=_answer.get()
    if batch is None:return False
    batch.queue(documents,kind,key,raw);return True
