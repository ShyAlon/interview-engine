"""Session-owned call budget and timing; shared HTTP sessions never share them."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
from threading import RLock
from vmodel_contracts.runtime import AdaptiveCallWatchdog

_current=ContextVar('interview_session_runtime',default=None)

@dataclass
class SessionRuntime:
    session_id:str=''
    calls:int=0
    lock:object=field(default_factory=RLock)
    watchdog:object=field(default_factory=lambda:AdaptiveCallWatchdog(multiplier=8.0,floor_s=240.0))

@contextmanager
def use(runtime):
    if not isinstance(runtime,SessionRuntime):raise TypeError('session runtime required')
    token=_current.set(runtime)
    try:yield
    finally:_current.reset(token)


def current():return _current.get()
