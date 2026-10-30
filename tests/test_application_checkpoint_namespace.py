"""The root checkpoint namespace must survive LangGraph's direct state writes."""
from interview_engine.application import InterviewApplication
from test_pg_application_wiring import fake_models,fake_ollama,make_conn,PgSessionStore,OWNER
from conftest import managed_pool


def test_root_config_supports_direct_graph_state_update(tmp_path,fake_models,fake_ollama):
    store=PgSessionStore(managed_pool(make_conn(tmp_path).info.dsn))
    store.create('namespace-fixture','task-fixture',OWNER)
    claim=store.claim('namespace-fixture',OWNER,'namespace-worker',ttl=60)
    app=InterviewApplication('namespace-fixture','task-fixture','idea',
        model_route='ollama',initial_model='phi4',artifact_root=str(tmp_path),pg_store=store,pg_claim=claim)
    try:
        app.advance()
        app.graph.update_state(app.cfg,{'messages':[]})
        assert app.cfg['configurable']['checkpoint_ns']==''
        snapshot=app.graph.get_state(app.cfg)
        assert snapshot.config['configurable']['checkpoint_ns']==''
        assert snapshot.config['configurable']['thread_id']=='namespace-fixture'
    finally:
        pass  # host pool is cleaned up by conftest
