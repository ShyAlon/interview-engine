"""Call identity from the actual published graph anchor, node, step and item.

One identity per SDK invocation, not per transport retry. SDK retries are off
for broker models. A missing graph context refuses rather than inventing IDs.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib,json

_call=ContextVar('child_logical_call',default=None)

@contextmanager
def broker_invocation(stage,item=None,*,metadata=None):
    from .model_config import _response_staging,StagingContext
    if metadata is None and isinstance(item,str):
        import re
        if re.fullmatch(r'[A-Za-z0-9 _./():-]{1,200}',item):metadata={'item':item,'fields':[]}
    staging=_response_staging.get()
    if staging is None:
        token=_call.set(None)
    else:
        node=StagingContext.current_node();key,ordinal=staging.next_cursor(node,stage,item or '')
        staging.cursors[key]=ordinal+1
        identity=[staging.session_id,staging.generation,*key,ordinal]
        token=_call.set((stage,'call:'+hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest(),metadata))
    try:yield
    finally:_call.reset(token)


def broker_call_id(stage):
    call=_call.get()
    if call is None or call[0]!=stage:raise PermissionError('durable graph call cursor required')
    return call[1]


def broker_call_metadata(stage):
    call=_call.get()
    return call[2] if call is not None and call[0]==stage else None
