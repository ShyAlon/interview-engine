"""Same-host private FD handoff prototype, not a public HTTP/header interface.

The control connection must come from the trusted supervisor. Expected identity
comes from its request binding, never the received envelope. No listener or
ambient process-global gateway is installed by this module.
"""
import array
import json
import os
import socket
from contextlib import contextmanager
from .attempt_admission import AdmissionUnavailable, admission_gateway
from .admission_client import SocketAdmissionGateway


@contextmanager
def received_job_gateway(control, *, owner_id, job_id, session_id):
    if control.family != socket.AF_UNIX or control.type != socket.SOCK_SEQPACKET:
        raise ValueError('private handoff requires a Unix sequenced-packet connection')
    descriptors = []
    gateway = None
    try:
        raw, ancillary, flags, _ = control.recvmsg(4096, socket.CMSG_SPACE(16 * array.array('i').itemsize))
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds = array.array('i')
                fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
                descriptors.extend(fds)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(descriptors) != 1:
            raise ValueError('invalid descriptor handoff')
        envelope = json.loads(raw)
        if envelope != {'version': 1, 'owner_id': owner_id, 'job_id': job_id, 'session_id': session_id}:
            raise ValueError('handoff does not match trusted job binding')
        gateway = SocketAdmissionGateway(descriptors[0])
        with admission_gateway(gateway):
            yield gateway
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise AdmissionUnavailable('private job admission handoff unavailable') from exc
    finally:
        if gateway is not None:
            gateway.close()
        for fd in descriptors:
            os.close(fd)
