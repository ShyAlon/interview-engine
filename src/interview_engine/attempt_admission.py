"""Private pre-provider seam. Cloud mode has no permissive fallback.

The gateway must be bound to a verified durable job outside producer code.
No prices, owner limits or token bounds are inferred here.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import os
import uuid

_socket_gateway=None
_socket_fd=None
_gateway=ContextVar('attempt_admission_gateway',default=None)
_required=ContextVar('attempt_admission_required',default=False)

class AdmissionUnavailable(RuntimeError):pass

@contextmanager
def admission_gateway(gateway,*,required=False):
    token=_gateway.set(gateway);required_token=_required.set(required)
    try:yield
    finally:
        _required.reset(required_token);_gateway.reset(token)

def _cloud():
    return _required.get() or os.environ.get('VMODEL_TENANT_OWNER') is not None or os.environ.get('INTERVIEW_ENGINE_TENANTS')=='1'

def _reserve(stage,provider,model,messages,max_output_tokens):
    gateway=_gateway.get()
    # A multi-owner HTTP service must never use an ambient process-wide FD.
    # Its authority must be installed for this exact request/job context.
    if gateway is None and os.environ.get('INTERVIEW_ENGINE_TENANTS') == '1':
        raise AdmissionUnavailable('tenant service requires request-bound job admission')
    if gateway is None and os.environ.get('VMODEL_ADMISSION_FD'):
        global _socket_gateway,_socket_fd
        if _socket_gateway is not None and _socket_fd!=os.environ['VMODEL_ADMISSION_FD']:
            raise AdmissionUnavailable('admission descriptor changed during worker lifetime')
        if _socket_gateway is None:
            from .admission_client import SocketAdmissionGateway
            try:
                _socket_gateway=SocketAdmissionGateway(int(os.environ['VMODEL_ADMISSION_FD']))
                _socket_fd=os.environ['VMODEL_ADMISSION_FD']
            except Exception as exc:raise AdmissionUnavailable('invalid inherited admission descriptor') from exc
        gateway=_socket_gateway
    if gateway is None:raise AdmissionUnavailable('cloud provider admission is unavailable')
    # Gateway computes/rejects exact serialized input bound and pinned price.
    attempt=gateway.reserve(attempt_id=uuid.uuid4().hex,stage=stage,provider=provider,
        model=model,messages=messages,max_output_tokens=max_output_tokens)
    if not attempt:raise AdmissionUnavailable('cloud provider attempt was not admitted')
    gateway.start(attempt)
    return gateway,attempt

# Bound for one staged raw provider payload; exceeding it fails loudly (the attempt
# stays on the ledger for reconciliation) instead of silently dropping a paid result.
MAX_STAGED_RESPONSE_BYTES = 4 * 1024 * 1024


def admitted_invoke(call,*,stage,provider,model,messages,max_output_tokens,item=None):
    from .stop_fence import check_execution
    check_execution()
    if not _cloud():
        response=call()
        try:check_execution()
        except __import__('interview_engine.stop_fence',fromlist=['StopRequested']).StopRequested:
            from .stop_fence import record_stopped_usage
            record_stopped_usage(response,stage=stage,provider=provider,model=model)
            raise
        return response
    from . import model_config
    staging=model_config._response_staging.get()
    if staging is not None:
        key=(stage,item or '')
        log=staging.store.staged_log(staging.session_id,staging.generation,stage=stage,item=key[1])
        idx=staging.cursors.get(key,0)
        if idx<len(log):
            staging.cursors[key]=idx+1
            row=log[idx]
            from .errors import UserFacingError
            fingerprint=model_config._staging_request_sha(
                type('Cfg',(),{'provider':provider,'model':model,'max_tokens':max_output_tokens})(),messages,stage,item)
            if row['request_sha256']!=fingerprint:
                # The deterministic replay diverged (e.g. code changed mid-session):
                # never mix old staged results into a new execution path.
                raise UserFacingError(f"stage '{stage}': the staged provider result does not match this request.",
                    "the session's execution path changed between the crash and the resume.",
                    "Resume the session with the original build or start a new session.")
            if row['response'] is not None:
                from langchain_core.load import loads
                model_config._log_llm({"event":"llm_call_staged_replay","provider":provider,"model":model,"stage":stage,**( {"item":item} if item else {})})
                return loads(__import__('json').dumps(row['response']))
            # Dispatched but never completed: the provider may have been charged.
            # Unknown outcome - pause for reconciliation, never blind-rerun.
            raise UserFacingError(f"stage '{stage}': a provider call was dispatched but its result is missing.",
                "the process crashed after dispatch and before the result was staged; the call may have been billed.",
                "Reconcile attempt " + str(row['attempt_id'] or '(unbound)') + " against the provider ledger before resuming.")
        staging.cursors[key]=idx+1 if idx<len(log) else len(log)+1
        ordinal=len(log)
        request_sha=model_config._staging_request_sha(
            type('Cfg',(),{'provider':provider,'model':model,'max_tokens':max_output_tokens})(),messages,stage,item)
        staging.store.record_dispatch(staging.claim,stage=stage,item=key[1],ordinal=ordinal,
            request_sha256=request_sha,provider=provider,model=model)
        gateway,attempt=_reserve(stage,provider,model,messages,max_output_tokens)
        staging.store.bind_attempt(staging.claim,stage=stage,item=key[1],ordinal=ordinal,
            attempt_id=getattr(attempt,'attempt_id',None) or str(attempt))
        try:
            response=call()
            from langchain_core.load import dumpd
            payload=dumpd(response)
            if len(__import__('json').dumps(payload))>MAX_STAGED_RESPONSE_BYTES:
                raise UserFacingError(f"stage '{stage}': provider response exceeds the staged-size bound.",
                    "a paid result larger than the staging bound cannot be made durable.",
                    "Reconcile the billed attempt before retrying.")
            # Raw result + measured usage durable BEFORE the settle handoff and any
            # interpretation: crash windows after this point replay, never rebill.
            staging.store.complete_staged(staging.claim,stage=stage,item=key[1],ordinal=ordinal,
                response=payload,usage=model_config._usage_of(response))
            # Missing usage is the gateway's responsibility: hold, never assume zero.
            gateway.settle(attempt,response)
            check_execution()
            return response
        except BaseException:
            try:gateway.uncertain(attempt)
            except Exception:pass  # Never turn a ledger outage into a refund or hide provider failure.
            raise
    gateway,attempt=_reserve(stage,provider,model,messages,max_output_tokens)
    try:
        response=call()
        # Missing usage is the gateway's responsibility: hold, never assume zero.
        gateway.settle(attempt,response)
        check_execution()
        return response
    except BaseException:
        try:gateway.uncertain(attempt)
        except Exception:pass  # Never turn a ledger outage into a refund or hide provider failure.
        raise

def admitted_stream(call,*,stage,provider,model,messages,max_output_tokens):
    if not _cloud():
        yield from call()
        return
    gateway,attempt=_reserve(stage,provider,model,messages,max_output_tokens)
    try:
        yield from call()
    finally:
        # Streaming aggregation/usage is not wired yet. Preserve the hold.
        gateway.uncertain(attempt)

def checked_model_invoke(chat_model,messages,*,stage,provider,model,max_output_tokens,item=None):
    from .wire_binding import wire_model
    if wire_model(chat_model,stage=stage,provider=provider,model=model):
        from .stop_fence import check_execution
        check_execution()
        try:response=chat_model.invoke(messages)
        except Exception as exc:
            # OpenAI wraps transport control exceptions as connection errors.
            # Preserve Stop identity rather than presenting a provider outage.
            from .stop_fence import StopRequested
            cause=exc;seen=set()
            while cause is not None and id(cause) not in seen:
                if isinstance(cause,StopRequested):raise cause
                seen.add(id(cause));cause=cause.__cause__ or cause.__context__
            raise
        check_execution()
        return response
    return admitted_invoke(lambda:chat_model.invoke(messages),stage=stage,provider=provider,
        model=model,messages=messages,max_output_tokens=max_output_tokens,item=item)
