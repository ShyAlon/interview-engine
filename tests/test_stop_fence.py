from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from interview_engine.stop_fence import StopFence,StopRequested,check_execution


def test_stop_does_not_wait_on_call_and_late_result_is_fenced(tmp_path):
    fence=StopFence(tmp_path/'stop.json');entered=threading.Event();release=threading.Event();calls=[];published=[]
    def work():
        with fence.running() as generation:
            calls.append('accepted');entered.set();release.wait(2)
            calls.append('usage recorded')
            fence.check(generation);published.append('late result')
    with ThreadPoolExecutor() as pool:
        future=pool.submit(work);assert entered.wait(1)
        assert fence.request_stop()=='STOP_REQUESTED'
        assert fence.state=='STOP_REQUESTED' and fence.active==1
        assert StopFence(fence.path).state=='STOP_REQUESTED'  # same-host writer remains active
        release.set()
        with pytest.raises(StopRequested):future.result()
    assert calls==['accepted','usage recorded'] and published==[] and fence.state=='STOPPED'
    with pytest.raises(StopRequested):
        with fence.running():pass


def test_stop_replay_does_not_increment_generation_and_no_auto_resume(tmp_path):
    f=StopFence(tmp_path/'stop.json');assert f.request_stop()=='STOPPED'
    generation=f.generation;f.request_stop();assert f.generation==generation
    restored=StopFence(f.path)
    assert restored.requested and restored.generation==generation
    with pytest.raises(StopRequested):
        with restored.running():pass


def test_next_provider_check_blocks_but_settlement_can_finish(tmp_path):
    f=StopFence(tmp_path/'stop.json')
    with f.running():
        check_execution();f.request_stop()
        # Existing accepted response usage settlement does not call this check.
        with pytest.raises(StopRequested):check_execution()


def test_fenced_publish_after_stop_never_writes(tmp_path):
    from interview_engine.stop_fence import guarded_publish
    f=StopFence(tmp_path/'stop.json');out=tmp_path/'result.json'
    with f.running():
        guarded_publish(lambda:out.write_text('old approved'))
        f.request_stop()
        with pytest.raises(StopRequested):guarded_publish(lambda:out.write_text('late replacement'))
    assert out.read_text()=='old approved'


def test_timeout_does_not_unlock_delete_while_provider_thread_lives(tmp_path):
    from interview_engine.model_config import _run_with_deadline
    f=StopFence(tmp_path/'stop.json');started=threading.Event();release=threading.Event();finished=threading.Event()
    def provider():
        started.set();release.wait(2);finished.set()
    with f.running():
        with pytest.raises(TimeoutError):_run_with_deadline(provider,.01,'fake')
        assert started.is_set();f.request_stop()
    assert f.state=='STOP_REQUESTED' and f.active==1
    release.set();assert finished.wait(1)
    import time
    for _ in range(100):
        if f.state=='STOPPED':break
        time.sleep(.001)
    assert f.state=='STOPPED'


def test_local_accepted_response_usage_recorded_before_stop_discards_result(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from interview_engine.attempt_admission import admitted_invoke
    from interview_engine.stop_fence import _stopped_usage
    monkeypatch.delenv('VMODEL_TENANT_OWNER',raising=False);monkeypatch.delenv('INTERVIEW_ENGINE_TENANTS',raising=False)
    f=StopFence(tmp_path/'stop.json');recorded=[]
    response=SimpleNamespace(usage_metadata={'input_tokens':2,'output_tokens':3})
    def call():f.request_stop();return response
    token=_stopped_usage.set(lambda response,**kw:recorded.append((response.usage_metadata,kw)))
    try:
        with f.running(),pytest.raises(StopRequested):
            admitted_invoke(call,stage='needs',provider='fake',model='fake',messages=[],max_output_tokens=10)
    finally:_stopped_usage.reset(token)
    assert len(recorded)==1 and recorded[0][0]=={'input_tokens':2,'output_tokens':3}
    assert f.state=='STOPPED'


def test_stopped_result_sinks_do_not_publish_but_existing_evidence_remains(tmp_path):
    from interview_engine.raw_output import RawOutputLog
    from interview_engine.verification import VerificationLog
    from interview_engine.evidence import EvidenceLake
    from interview_engine.progress_journal import ProgressJournal
    from uuid import uuid4
    sid=str(uuid4());f=StopFence(tmp_path/'stop.json')
    raw=RawOutputLog(tmp_path/'raw');verification=VerificationLog(tmp_path/'verification.jsonl')
    lake=EvidenceLake(tmp_path/'lake','fake');journal=ProgressJournal(tmp_path,sid,'fake')
    baseline=raw.write('fake','fake','saved');before=baseline.read_bytes()
    with f.running():
        f.request_stop()
        for publish in [lambda:raw.write('fake','fake','late'),lambda:verification.write({'fake':'late'}),
                        lambda:lake._append('evidence',{'fake':'late'}),lambda:journal.append(phase_key='intent',stage='fake',label='fake',status='completed')]:
            with pytest.raises(StopRequested):publish()
    assert baseline.read_bytes()==before and len(list((tmp_path/'raw').glob('*')))==1
    assert not (tmp_path/'verification.jsonl').exists()
    assert not (lake.paths.run/'evidence').exists()
    assert not journal.read().events


def test_preexisting_fence_observes_external_marker_and_never_claims_quiescence(tmp_path):
    from interview_engine.stop_fence import guarded_publish
    first=StopFence(tmp_path/'stop.json');other=StopFence(first.path);result=tmp_path/'late'
    with first.running():
        other.request_stop()
        with pytest.raises(StopRequested):check_execution()
        with pytest.raises(StopRequested):guarded_publish(lambda:result.write_text('late'))
    assert first.state=='STOP_REQUESTED' and not result.exists()
    assert first.request_stop()=='STOP_REQUESTED'
    assert first.generation==other.generation


def test_cross_process_stop_is_observed_by_existing_execution(tmp_path):
    import subprocess,sys
    from interview_engine.stop_fence import guarded_publish
    fence=StopFence(tmp_path/'stop.json');out=tmp_path/'late'
    with fence.running():
        subprocess.run([sys.executable,'-c','from interview_engine.stop_fence import StopFence;import sys;StopFence(sys.argv[1]).request_stop()',str(fence.path)],check=True)
        with pytest.raises(StopRequested):guarded_publish(lambda:out.write_text('late'))
    assert not out.exists() and fence.state=='STOP_REQUESTED'


def test_external_stop_cannot_claim_stopped_while_another_writer_holds_lock(tmp_path):
    first=StopFence(tmp_path/'stop.json');other=StopFence(first.path)
    with first.running():
        assert other.request_stop()=='STOP_REQUESTED'
        with pytest.raises(StopRequested):other.confirm_quiescent_after_recovery(no_live_worker=True)
        with pytest.raises(StopRequested):
            with StopFence(first.path).running():pass
    assert other.confirm_quiescent_after_recovery(no_live_worker=True)=='STOPPED'


def test_duplicate_writer_before_stop_is_denied(tmp_path):
    first=StopFence(tmp_path/'stop.json');other=StopFence(first.path)
    with first.running():
        with pytest.raises(__import__('interview_engine.stop_fence',fromlist=['SessionWriterBusy']).SessionWriterBusy,match='another session writer'):
            with other.running():pass
    with other.running():pass
