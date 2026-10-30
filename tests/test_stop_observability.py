"""The engine-side minimum markers for the stop, stop/reconcile and progress routes:
request, outcome and refusal each leave one line in the interview_engine.http log."""
import logging
from uuid import uuid4

from fastapi.testclient import TestClient
from vmodel_contracts.interactive import InputKind, InputRequest, StartSessionRequest

from interview_engine.http_service import SessionRegistry, create_app

TOKEN = 'test-service-token'

class Program:
    finished = False
    def advance(self, response=None):
        return InputRequest(input_id=uuid4(), kind=InputKind.TEXT, prompt='fake')

def _client(pg_registry):
    registry = pg_registry(lambda *a: Program())
    return TestClient(create_app(registry, service_token=TOKEN), headers={'X-Interview-Engine-Token': TOKEN}), registry

def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == 'interview_engine.http']

def test_stop_marks_request_and_recorded(pg_registry, caplog):
    client, registry = _client(pg_registry)
    session, _ = registry.start(StartSessionRequest(task_id='fake', idea='fake', model_route='local'))
    with caplog.at_level(logging.INFO, logger='interview_engine.http'):
        response = client.post(f'/v1/sessions/{session.session_id}/stop')
    assert response.status_code == 200
    messages = _lines(caplog)
    assert any(m.startswith('session.stop.request ') for m in messages)
    assert any(m.startswith('session.stop.recorded ') and 'STOP' in m for m in messages)

def test_stop_marks_a_not_found_refusal(pg_registry, caplog):
    client, _ = _client(pg_registry)
    with caplog.at_level(logging.INFO, logger='interview_engine.http'):
        response = client.post(f'/v1/sessions/{uuid4()}/stop')
    assert response.status_code == 404
    messages = _lines(caplog)
    assert any(m.startswith('session.stop.request ') for m in messages)
    assert any(m.startswith('session.stop.refused ') and 'not-found' in m for m in messages)

def test_reconcile_marks_request_and_recorded(pg_registry, caplog):
    client, registry = _client(pg_registry)
    session, _ = registry.start(StartSessionRequest(task_id='fake', idea='fake', model_route='local'))
    registry.stop(session.session_id)
    with caplog.at_level(logging.INFO, logger='interview_engine.http'):
        response = client.post(f'/v1/sessions/{session.session_id}/stop/reconcile')
    assert response.status_code == 200
    messages = _lines(caplog)
    assert any(m.startswith('session.stop.reconcile.request ') for m in messages)
    assert any(m.startswith('session.stop.reconcile.recorded ') for m in messages)

def test_progress_marks_a_missing_journal(pg_registry, caplog):
    client, _ = _client(pg_registry)
    with caplog.at_level(logging.INFO, logger='interview_engine.http'):
        response = client.get(f'/v1/sessions/{uuid4()}/progress')
    assert response.status_code == 404
    assert any(m.startswith('session.progress.no_journal ') for m in _lines(caplog))
