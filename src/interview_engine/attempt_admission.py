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

def admitted_invoke(call,*,stage,provider,model,messages,max_output_tokens):
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

def checked_model_invoke(chat_model,messages,*,stage,provider,model,max_output_tokens):
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
        model=model,messages=messages,max_output_tokens=max_output_tokens)
