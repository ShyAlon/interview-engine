"""New compare-and-claim method against delivery's real migration and RLS."""
import time
import pytest
from requi_child_contract import JobBinding,InterviewAdvance
from test_pg_claim_transport import migrated,_STATE,A,SESSION_A,DELIVERY

pytestmark=pytest.mark.skipif(not DELIVERY,reason='requires explicit delivery migration source')


def test_child_claim_uses_original_auth_and_revocation_rls():
    import psycopg
    from conftest import managed_pool
    from interview_engine.pg_session_store import PgSessionStore
    from interview_engine.child_step import published_digest
    pool=managed_pool(_STATE['dsn']+' user=delivery_app')
    store=PgSessionStore(pool,claims=(A,SESSION_A))
    sid='child-rls-'+str(time.time_ns())
    store.create(sid,'ict',A)
    claim=store.claim(sid,A,'parent',60)
    store.publish_step(claim,{'sequence':2},published_thread_id=sid)
    assert store.release(claim)
    row=store.load_published(sid,A)
    binding=JobBinding(owner_id=A,auth_session_id=SESSION_A,task_id='ict',session_id=sid,
        job_id='job-1',generation=claim.generation,lease_epoch=claim.epoch,provider='ollama',model='fake')
    operation=InterviewAdvance(kind='interview.advance',checkpoint_sha256=published_digest(row),expected_sequence=2)
    child,_=store.claim_published_expected(binding,operation,'child',60)
    assert child.epoch==claim.epoch+1 and store.release(child)
    # Same owner but nonexistent/revoked Auth session must see no claimable row.
    revoked=PgSessionStore(pool,claims=(A,'00000000-0000-0000-0000-000000000099'))
    binding=binding.model_copy(update={'lease_epoch':child.epoch})
    with pytest.raises(PermissionError):revoked.claim_published_expected(binding,operation,'revoked',60)
    foreign=PgSessionStore(pool,claims=('00000000-0000-0000-0000-000000000002','00000000-0000-0000-0000-0000000000a2'))
    with pytest.raises(PermissionError):foreign.claim_published_expected(binding,operation,'foreign',60)
