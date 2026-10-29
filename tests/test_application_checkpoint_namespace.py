"""The root checkpoint namespace must survive LangGraph's direct state writes."""
from interview_engine.application import InterviewApplication
from test_pg_application_wiring import fake_models,fake_ollama


def test_root_config_supports_direct_graph_state_update(tmp_path,fake_models,fake_ollama):
    app=InterviewApplication('namespace-fixture','task-fixture','idea',
        model_route='ollama',initial_model='phi4',artifact_root=str(tmp_path))
    try:
        app.advance()
        app.graph.update_state(app.cfg,{'messages':[]})
        assert app.cfg['configurable']['checkpoint_ns']==''
        snapshot=app.graph.get_state(app.cfg)
        assert snapshot.config['configurable']['checkpoint_ns']==''
        assert snapshot.config['configurable']['thread_id']=='namespace-fixture'
    finally:
        app.checkpointer.conn.close()
