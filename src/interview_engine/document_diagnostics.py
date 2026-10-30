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
    documents,key=_target(path);return documents.write(key,text.encode('utf-8'))

def append(path,text):
    documents,key=_target(path);return documents.append(key,text.encode('utf-8'))

def read(path):
    documents,key=_target(path);raw=documents.read(key)
    if raw is None:raise FileNotFoundError(str(path))
    return raw.decode('utf-8')
