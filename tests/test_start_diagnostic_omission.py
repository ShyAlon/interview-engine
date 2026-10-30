import pytest
from fastapi.testclient import TestClient
from interview_engine.http_service import SessionRegistry,create_app
from interview_engine import pg_diagnostics


@pytest.mark.parametrize('mode,reason',[('oversize','oversize'),('broken','build_failed')])
def test_start_diagnostic_omission_is_safe_visible_and_keeps_minimal_503(pg_registry,monkeypatch,caplog,mode,reason):
    registry=pg_registry()
    def start(request):raise RuntimeError('original')
    monkeypatch.setattr(registry,'start',start)
    def block(error):
        if mode=='broken':raise ValueError('private-diagnostic-text')
        return {'message':'private-diagnostic-text'*200}
    monkeypatch.setattr(pg_diagnostics,'build_operator_block',block)
    client=TestClient(create_app(registry,service_token='token'),headers={'X-Interview-Engine-Token':'token'})
    response=client.post('/v1/sessions',json={'protocol_version':1,'task_id':'t','idea':'i','model_route':'m'})
    assert response.status_code==503 and 'x-engine-operator-diagnostics' not in response.headers
    records=[r for r in caplog.records if 'operator_diagnostic_omitted' in r.getMessage()]
    assert len(records)==1 and 'reason='+reason in records[0].getMessage()
    assert 'private-diagnostic-text' not in records[0].getMessage() and not records[0].exc_info
    assert response.json()['detail']['detail']=='engine program failed during start: RuntimeError'
