"""Fixed local-only subprocess target. Never imported by installed runtime."""
import json
import os
import socket
import sys
from requi_child_contract import JobBinding
from interview_engine.child_channel import serve_restored
from interview_engine.pg_pool import build_pool
from interview_engine.pg_session_store import PgSessionStore

binding=JobBinding.model_validate(json.loads(sys.argv[2]))
pool=build_pool(sys.argv[1])
store=PgSessionStore(pool,claims=(str(binding.owner_id),str(binding.auth_session_id)))

class FakeProgram:
    finished=False
    def advance(self,*a):raise AssertionError('phase 2 must not advance')

def action(step,stopped):
    assert step.session.task_id==binding.task_id
    assert step.session.pg_claim.epoch==binding.lease_epoch+1

try:
    serve_restored(socket.socket(fileno=int(os.environ['REQUI_CHILD_FD'])),trusted_binding=binding,
        store=store,program_factory=lambda *a:FakeProgram(),action=action)
finally:pool.close()
