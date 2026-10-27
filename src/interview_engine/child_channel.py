"""Private one-operation receiver, not an installed service or production CLI.

Trusted launch wiring supplies the binding and store separately from received
bytes. Only explicit fake program/action factories can exercise this seam in
phase 2; application start/answer and provider dispatch are not enabled here.
"""
from contextlib import contextmanager
from threading import Event,Thread
from requi_child_contract import decode_operation,require_binding,LifecycleEvent,encode_event
from .child_step import restored_child_step


class ParentGone(RuntimeError):pass


def serve_restored(channel,*,trusted_binding,store,program_factory,action):
    """Restore one operation, invoke an isolated action, close/release and exit.

    action receives a cooperative stop predicate, not a raw socket. The parent
    owns escalation if the action ignores it. EOF is stop, never permission to
    continue a provider call. No automatic retry, payload result or billing.
    """
    reader=channel.makefile('rb');writer=channel.makefile('wb',buffering=0)
    stop=Event();gone=Event();sequence=0;thread=None
    def event(state):
        nonlocal sequence
        if gone.is_set():raise ParentGone('parent descriptor closed')
        sequence+=1
        writer.write(encode_event(LifecycleEvent(version=1,operation_id=operation.operation_id,sequence=sequence,state=state)))
    def watch():
        try:line=reader.readline(6)
        except OSError:line=b''
        if line!=b'STOP\n':gone.set()
        stop.set()
    try:
        raw=reader.readline(64*1024+1)
        operation=decode_operation(raw);require_binding(operation,trusted_binding)
        event('ready')
        thread=Thread(target=watch,name='child-parent-watch',daemon=True);thread.start()
        with restored_child_step(operation,trusted_binding=trusted_binding,store=store,program_factory=program_factory) as step:
            if stop.is_set():
                if gone.is_set():raise ParentGone('parent descriptor closed')
                event('stopped');return
            event('running')
            action(step,stop.is_set)
            if gone.is_set():raise ParentGone('parent descriptor closed')
        # Terminal success only after lease release and heartbeat shutdown.
        event('stopped' if stop.is_set() else 'completed')
    finally:
        import socket
        try:channel.shutdown(socket.SHUT_RDWR)
        except OSError:pass
        if thread is not None:thread.join(timeout=1)
        reader.close();writer.close();channel.close()
