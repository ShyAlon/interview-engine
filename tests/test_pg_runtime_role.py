"""Runtime uses SET-only membership, not inherited table privileges."""
from contextlib import contextmanager
from interview_engine.pg_claims import apply_claims,CLAIM_SQL


def test_transaction_enters_fixed_runtime_role_before_tenant_claims():
    class Connection:
        def __init__(self):self.calls=[];self.in_pipeline=False
        @contextmanager
        def pipeline(self):
            self.in_pipeline=True
            try:yield
            finally:self.in_pipeline=False
        def execute(self,*args):
            assert self.in_pipeline
            self.calls.append(args)
    conn=Connection();apply_claims(conn,('owner','auth-session'))
    assert conn.calls==[('SET LOCAL ROLE requi_backend',),(CLAIM_SQL,('owner','auth-session'))]
