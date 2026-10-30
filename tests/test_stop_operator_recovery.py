import importlib.util
from pathlib import Path
import pytest
from test_pg_session_store import make_store,OWNER_A,OWNER_B

spec=importlib.util.spec_from_file_location('stop_recovery',Path(__file__).parents[1]/'scripts/confirm-stopped-writer-exit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_recovery_is_dry_run_and_exact_stopped_identity_only(tmp_path):
    store=make_store(tmp_path);store.create('s','task',OWNER_A)
    claim=store.claim('s',OWNER_A,'old-process',60)
    args=dict(session_id='s',owner_id=OWNER_A,generation=2,epoch=claim.epoch,instance=claim.instance)
    with pytest.raises(ValueError):module.recover(store,**args)
    store.stop('s',OWNER_A)
    before=store.load_published('s',OWNER_A)
    assert module.recover(store,**args)['mode']=='dry-run'
    assert store.load_published('s',OWNER_A)==before
    with pytest.raises(ValueError,match='operator confirmation'):module.recover(store,**args,apply=True)
    for change in ({'owner_id':OWNER_B},{'generation':3},{'epoch':2},{'instance':'new-process'}):
        with pytest.raises(ValueError,match='identity'):module.recover(store,**(args|change),apply=True,writers_terminated=True)
    assert store.load_published('s',OWNER_A)==before
    result=module.recover(store,**args,apply=True,writers_terminated=True)
    assert result['process_status']=='STOPPED'
    after=store.load_published('s',OWNER_A)
    assert after['generation']==before['generation']
    assert after['control']['stop_quiescent']
    assert after['control']['stop_quiescence_source']=='operator-confirmed-writer-exit'
    assert module.recover(store,**args,apply=True,writers_terminated=True)['already_quiescent']


def test_existing_debug_projection_exposes_unproven_stop_status():
    """Debug's existing protocol projection does not hide conservative recovery."""
    import os,sys
    from types import SimpleNamespace
    from vmodel_contracts.interactive import ProcessStatus,NextStep
    source=os.getenv('DELIVERY_TEST_DELIVERY_SOURCE')
    if not source:pytest.skip('requires DELIVERY_TEST_DELIVERY_SOURCE')
    sys.path.insert(0,str(Path(source)/'src'))
    try:
        from delivery_api.task_diagnostics import engine_summary
        state=SimpleNamespace(session_id='s',process_status=ProcessStatus.STOP_REQUESTED,
            next_step=NextStep.STOP,sequence=7,open_input=None)
        shown=engine_summary(state)
        assert shown['process_status_name']=='STOP_REQUESTED' and shown['process_status']==70
        state.process_status=ProcessStatus.STOPPED
        assert engine_summary(state)['process_status_name']=='STOPPED'
    finally:sys.path.pop(0)
