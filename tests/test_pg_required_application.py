"""Retired file mode must fail before side effects or model routing."""
import pytest


def test_application_requires_pg_before_local_paths_or_provider_setup(tmp_path,monkeypatch):
    from interview_engine.application import InterviewApplication
    from interview_engine import session_models
    calls=[]
    monkeypatch.setattr(session_models,'resolve',lambda *a,**k:calls.append(1))
    with pytest.raises(ValueError,match='PG'):
        InterviewApplication('retired','task','idea',artifact_root=str(tmp_path),model_route='ollama')
    assert calls==[] and not list(tmp_path.iterdir())
