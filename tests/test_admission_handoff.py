import array
import json
import socket
import pytest
from interview_engine.admission_handoff import received_job_gateway
from interview_engine.attempt_admission import AdmissionUnavailable


def _sequenced_packets_supported():
    try:
        pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    except (AttributeError, OSError):
        return False
    for end in pair:
        end.close()
    return True


needs_seqpacket = pytest.mark.skipif(
    not _sequenced_packets_supported(),
    reason="this platform has no Unix sequenced-packet sockets (macOS); the handoff is used on the Linux server, where this runs")


def test_stream_connection_is_not_a_handoff_transport():
    left,right=socket.socketpair()
    try:
        with pytest.raises(ValueError):
            with received_job_gateway(right,owner_id='a',job_id='j',session_id='s'):pytest.fail('stream accepted')
    finally:left.close();right.close()


@needs_seqpacket
@pytest.mark.parametrize('count,payload',[(0,{}),(2,{}),(1,{'version':1,'owner_id':'a','job_id':'j','session_id':'s','extra':'spoof'}),(1,{'version':1,'owner_id':'a','job_id':'j','session_id':'wrong'})])
def test_missing_extra_or_mismatched_handoff_rejected(count,payload):
    left,right=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
    channels=[socket.socketpair() for _ in range(count)]
    try:
        ancillary=[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[pair[0].fileno() for pair in channels]))] if count else []
        left.sendmsg([json.dumps(payload).encode()],ancillary)
        with pytest.raises(AdmissionUnavailable):
            with received_job_gateway(right,owner_id='a',job_id='j',session_id='s'):pytest.fail('invalid authority accepted')
    finally:
        left.close();right.close()
        for pair in channels:
            for channel in pair:channel.close()
