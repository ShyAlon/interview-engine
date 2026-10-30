"""PG document transactions own Stop serialization, never nest a second lock."""
from types import SimpleNamespace
import pytest

def test_pg_final_outputs_do_not_enter_external_publication_transaction(monkeypatch):
    from interview_engine.application import InterviewApplication
    from interview_engine import stop_fence
    app=InterviewApplication.__new__(InterviewApplication);app.documents=object();calls=[]
    app._finish_outputs=lambda:calls.append('atomic documents')
    def blocked(call):raise AssertionError('outer task lock would deadlock separate interview connection')
    monkeypatch.setattr(stop_fence,'guarded_publish',blocked)
    app._finish()
    assert calls==['atomic documents']

def test_pg_evidence_append_owns_its_publication_fence(monkeypatch):
    from interview_engine.evidence import DatabaseEvidenceLake
    from interview_engine import stop_fence
    calls=[]
    docs=SimpleNamespace(append=lambda key,raw:calls.append((key,raw)))
    lake=DatabaseEvidenceLake(docs,'run')
    def blocked(call):raise AssertionError('nested task publication lock')
    monkeypatch.setattr(stop_fence,'guarded_publish',blocked)
    lake._append('turns',{'answer':'yes'})
    assert calls==[('evidence/turns.jsonl',b'{"answer":"yes"}\n')]
