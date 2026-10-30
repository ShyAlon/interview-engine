import pytest
from fastapi.testclient import TestClient
from interview_engine import question_config as qc
from interview_engine.http_service import SessionRegistry, create_app

TOKEN = "t"


@pytest.fixture(autouse=True)
def _isolate():
    old = qc.store()
    qc.set_store(qc.ConfigStore())
    yield
    qc.set_store(old)


def test_defaults_are_complete_and_valid():
    assert qc.validate(qc.defaults()) == qc.defaults()
    d = qc.defaults()
    assert d["retrieval"]["max_words"] == 90 and d["retrieval"]["max_chars"] == 600
    assert d["verification"]["retry_enabled"] is False


@pytest.mark.parametrize("bad", [
    {"nope": {}}, {"retrieval": {"max_words": 0}}, {"retrieval": {"max_words": True}},
    {"retrieval": {"top_k": "4"}}, {"retrieval": {"bogus": 1}}, {"verification": {"retry_enabled": "yes"}},
    {"integrations": {"external_system_verbs": ["ok", ""]}}, {"area_cues": {"data": "from"}}, [],
])
def test_invalid_config_is_rejected_naming_the_key(bad):
    with pytest.raises(qc.ConfigError):
        qc.validate(bad)


def test_partial_edit_merges_over_defaults():
    out = qc.validate({"retrieval": {"max_words": 60}})
    assert out["retrieval"]["max_words"] == 60 and out["retrieval"]["max_chars"] == 600


def test_file_store_is_read_per_call_and_survives_corruption(tmp_path):
    path = tmp_path / "q.json"
    st = qc.ConfigStore(path)
    assert st.get() == qc.defaults()
    st.put({"retrieval": {"top_k": 2}})
    assert st.get()["retrieval"]["top_k"] == 2
    other = qc.ConfigStore(path)  # another process/session sees the edit without restart
    assert other.get()["retrieval"]["top_k"] == 2
    path.write_text("{not json")
    assert st.get() == qc.defaults()


def test_http_get_put_roundtrip_and_422(tmp_path,pg_registry):
    client = TestClient(create_app(pg_registry(), service_token=TOKEN,quality_store=qc.ConfigStore(tmp_path/"question_quality.json")),
                        headers={"X-Interview-Engine-Token": TOKEN})
    got = client.get("/v1/question-quality-config").json()
    assert got["config"] == got["defaults"] == qc.defaults()
    r = client.put("/v1/question-quality-config", json={"config": {"verification": {"retry_enabled": True}}})
    assert r.status_code == 200 and r.json()["config"]["verification"]["retry_enabled"] is True
    assert client.get("/v1/question-quality-config").json()["config"]["verification"]["retry_enabled"] is True
    assert (tmp_path / "question_quality.json").exists()
    bad = client.put("/v1/question-quality-config", json={"config": {"retrieval": {"max_words": 9999}}})
    assert bad.status_code == 422 and "retrieval.max_words" in bad.json()["detail"]


def test_http_requires_token(pg_registry):
    client = TestClient(create_app(pg_registry(),service_token=TOKEN))
    assert client.get("/v1/question-quality-config").status_code == 401
    assert client.put("/v1/question-quality-config", json={}).status_code == 401


def test_corrupt_saved_config_exposes_safe_fallback_once_and_recovers(tmp_path,caplog):
    from interview_engine.question_config import ConfigStore,defaults
    path=tmp_path/'saved.json';store=ConfigStore(path)
    assert store.get()==defaults() and store.status()['source']=='defaults_missing'
    path.write_text('{fake-secret')
    assert store.get()==defaults() and store.get()==defaults()
    assert store.status()=={'source':'defaults_invalid','category':'JSONDecodeError','fallback':True}
    assert caplog.text.count('quality_config_fallback')==1
    assert 'fake-secret' not in caplog.text and str(path) not in caplog.text
    store.put({'retrieval':{'top_k':2}})
    assert store.get()['retrieval']['top_k']==2 and not store.status()['fallback']
