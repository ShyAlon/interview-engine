"""Application runtime binding is independent of an optional pinned model map."""
from types import SimpleNamespace
import pytest
from interview_engine.application import InterviewApplication
from interview_engine.session_runtime import SessionRuntime,current
from interview_engine import session_models

@pytest.mark.parametrize('pinned',[None,{'intent':'fake'}])
def test_application_always_binds_private_runtime_with_or_without_models(tmp_path,monkeypatch,pinned):
    from contextlib import contextmanager
    entered=[]
    @contextmanager
    def selected(value):
        entered.append(value)
        yield
    monkeypatch.setattr(session_models,'use',selected)
    application=InterviewApplication.__new__(InterviewApplication)
    application.runtime=SessionRuntime()
    application.model_map=pinned
    application.outdir=tmp_path
    application._heartbeat_signal=lambda *args:None
    application._call_end_signal=lambda *args:None
    application.usage_log=SimpleNamespace(record=lambda *args:None)
    assert current() is None
    with application._runtime_context():
        assert current() is application.runtime
    assert current() is None
    assert entered==([] if pinned is None else [pinned])
