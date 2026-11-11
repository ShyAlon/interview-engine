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

# Private best-effort copies only. Never evict prior crash evidence to make room.
SPOOL_FILE_BYTES=4*1024*1024
SPOOL_TOTAL_BYTES=32*1024*1024
SPOOL_TOTAL_FILES=128


def _recovery_files(base,*,include_temporary=False):
    import stat
    for folder in base.iterdir():
        if not re.fullmatch('[0-9a-f]{64}',folder.name) or folder.is_symlink() or not folder.is_dir():continue
        for path in folder.iterdir():
            if not (re.fullmatch(r'[0-9a-f]{64}\.json',path.name) or (include_temporary and path.name.startswith('.diagnostic-'))):continue
            info=path.lstat()
            if stat.S_ISREG(info.st_mode):yield path,info.st_size


def recovery_export(artifacts,*,max_bytes=120000):
    """Read-only diagnostic export, never replay or re-import authority.

    Returns decoded text so the owner's debug collector can redact credentials.
    File names and roots are fixed by the application, not external input.
    """
    import base64,json,os,stat
    root=Path(artifacts).resolve();records=[];omissions=[];used=0
    for base in root.glob('tenants/*/sessions/.diagnostic-recovery'):
        if base.resolve()!=base or not base.is_dir():continue
        for path,size in _recovery_files(base):
            if size>SPOOL_FILE_BYTES:
                omissions.append('oversize diagnostic spool');continue
            fd=None
            try:
                fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
                if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('nonregular spool')
                with os.fdopen(fd,'rb') as stream:
                    fd=None;raw=stream.read(SPOOL_FILE_BYTES+1)
                if len(raw)>SPOOL_FILE_BYTES:raise ValueError('oversize spool')
                value=json.loads(raw)
                if value.get('version')!=1 or value.get('kind')!='diagnostic-only-not-execution-replay':raise ValueError('unknown spool format')
                operations=[]
                for kind,key,encoded in value['operations']:
                    if kind not in ('write','append') or not isinstance(key,str):raise ValueError('invalid diagnostic operation')
                    operations.append({'kind':kind,'key':key,'text':base64.b64decode(encoded,validate=True).decode('utf-8',errors='replace')})
                record={'identity':value['identity'],'diagnostic_operations':operations}
                amount=len(json.dumps(record).encode())
                if used+amount>max_bytes:
                    omissions.append('diagnostic export size limit');continue
                used+=amount;records.append(record)
            except (OSError,ValueError,KeyError,TypeError):omissions.append('unreadable or malformed diagnostic spool')
            finally:
                if fd is not None:os.close(fd)
    return {'version':1,'kind':'diagnostic-only-not-execution-replay','records':records,
        'omissions':omissions[:32],'omission_count':len(omissions),'retention':'no automatic expiry or eviction; capacity exhaustion stops new recovery copies',
        'fidelity':'last successfully spooled snapshot; may be incomplete after a failed or full write'}


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
        import base64,json,os,tempfile,logging,fcntl
        temporary=None
        try:
            self.root.mkdir(mode=0o700,parents=True,exist_ok=True)
            if self.root.resolve()!=self.root:raise ValueError('recovery root changed')
            raw=json.dumps({'version':1,'identity':self.identity,'kind':'diagnostic-only-not-execution-replay',
                'operations':[[kind,key,base64.b64encode(data).decode('ascii')] for kind,key,data in self.operations]}).encode()
            if len(raw)>SPOOL_FILE_BYTES:raise ValueError('diagnostic spool file ceiling')
            # Same-owner children share the ceiling. The lock spans capacity
            # accounting and replacement, never any database/provider work.
            fd=os.open(self.root.parent/'.capacity.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                sizes=[size for path,size in _recovery_files(self.root.parent,include_temporary=True)]
                if len(sizes)>=SPOOL_TOTAL_FILES or sum(sizes)+len(raw)>SPOOL_TOTAL_BYTES:
                    raise ValueError('diagnostic spool owner ceiling')
                fd,temporary=tempfile.mkstemp(prefix='.diagnostic-',dir=self.root)
                with os.fdopen(fd,'wb') as stream:stream.write(raw)
                os.replace(temporary,self.path)
        except Exception:
            logging.getLogger(__name__).warning('diagnostic.spool_failed_or_full recovery_copy=not_updated',exc_info=False)
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
