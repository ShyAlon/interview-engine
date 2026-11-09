"""Warm claimed-session reuse on disposable Postgres; fake programs only.

Fail-first evidence: on origin/main these tests fail at import/collection -
interview_engine.warm_child_step does not exist before this change.
"""
from uuid import uuid4

import pytest
from requi_child_contract import decode_operation
import json

from interview_engine.child_step import published_digest
from interview_engine.http_service import default_program_factory
from interview_engine.pg_session_store import PgSessionStore
from interview_engine.warm_child_step import WarmChildStep
from test_child_step import operation, setup, OWNER


class Program:
    finished=False
    def advance(self,response=None):raise AssertionError('warm validation must not advance')


def warm(tmp_path,kind='interview.answer'):
    conn,store,sid,row=setup(tmp_path)
    op=operation(sid,row,kind=kind)
    step=WarmChildStep(op,trusted_binding=op.binding,store=store,program_factory=lambda *a:Program())
    return conn,store,sid,row,op,step


def test_warm_claims_once_and_holds_lease_across_checks(tmp_path):
    conn,store,sid,row,op,step=warm(tmp_path)
    assert step.session.pg_claim.epoch==2
    # A follow-up operation built from the same published state validates
    # against the live session without a second claim or any row re-read.
    again=operation(sid,row,kind='interview.answer')
    step.check(again,trusted_binding=again.binding)
    held=conn.execute('SELECT lease_epoch,placement_instance FROM engine_sessions WHERE session_id=%s',(sid,)).fetchone()
    assert held['lease_epoch']==2 and held['placement_instance']==step.registry._instance
    step.close()
    lease=conn.execute('SELECT lease_expires_at FROM engine_sessions WHERE session_id=%s',(sid,)).fetchone()
    assert lease['lease_expires_at']<=__import__('time').time()
    step.close()  # idempotent: releasing twice must not raise


@pytest.mark.parametrize('change',['sequence','wait','review','generation','stop'])
def test_warm_rejects_diverged_followup_evidence(tmp_path,change):
    conn,store,sid,row,op,step=warm(tmp_path)
    follow=operation(sid,row,kind='interview.answer');value=follow.model_dump(mode='json')
    if change=='sequence':value['operation']['expected_sequence']=99
    if change=='wait':value['operation']['wait_id']=str(uuid4())
    if change=='review':value['operation']['review_revision']='engine:'+sid+':99'
    if change=='generation':
        value['binding']['generation']=2;value['binding']['execution_anchor_sha256']='d'*64
    if change=='stop':step.session.pg_stopped=True
    follow=decode_operation(json.dumps(value).encode()+b'\n')
    with pytest.raises(PermissionError):
        step.check(follow,trusted_binding=follow.binding)
    step.close()
    # A refused follow-up never costs the lease: a fresh child can still claim.
    assert store.claim(sid,OWNER,'after-warm',60) is not None


def test_warm_setup_failure_releases_claim(tmp_path):
    conn,store,sid,row=setup(tmp_path)
    op=operation(sid,row,kind='interview.answer')
    def fail(*a):raise RuntimeError('fake warm restore failed')
    with pytest.raises(RuntimeError,match='fake warm restore failed'):
        WarmChildStep(op,trusted_binding=op.binding,store=store,program_factory=fail)
    assert store.claim(sid,OWNER,'after-error',60) is not None


def test_warm_production_factory_refused_before_database_or_program(tmp_path):
    conn,store,sid,row=setup(tmp_path)
    op=operation(sid,row,kind='interview.answer')
    with pytest.raises(PermissionError,match='money boundary'):
        WarmChildStep(op,trusted_binding=op.binding,store=store,program_factory=default_program_factory)
