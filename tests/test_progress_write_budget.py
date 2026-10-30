"""Ephemeral model signals do not become durable source journal rows."""
from test_llm_call_progress import _app
from interview_engine.progress import ProgressEvent


def test_application_does_not_journal_heartbeat_or_call_summary(tmp_path):
    app=_app(tmp_path)
    app._heartbeat_signal('probing','Probing',40)
    app._call_end_signal({'stage':'probing','latency_s':1.2})
    assert app.progress_journal.rows==[]


def test_graph_signal_filter_preserves_failure_and_boundaries():
    from interview_engine.application import durable_progress_signal
    rows=[ProgressEvent('probe','Probing',status,0) for status in ('started','heartbeat','completed','failed','item_started','item_completed')]
    assert [r.status for r in rows if durable_progress_signal(r)]==['started','completed','failed','item_started','item_completed']
    assert durable_progress_signal(ProgressEvent('llm_call:probe','Failure','failed',0))
