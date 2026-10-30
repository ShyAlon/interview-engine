"""A finished model call lands in the session progress journal: which call
type, how long, what it produced - the live answer to 'what is it doing'."""
import interview_engine.model_config as mc
from interview_engine.application import InterviewApplication
from test_pg_application_wiring import make_conn,PgSessionStore,OWNER
from conftest import managed_pool


class _StubJournal:
    def __init__(self):
        self.rows = []

    def append(self, **row):
        self.rows.append(row)


def _app(tmp_path):
    store=PgSessionStore(managed_pool(make_conn(tmp_path).info.dsn))
    store.create('s-call-end','task-1',OWNER)
    claim=store.claim('s-call-end',OWNER,'progress-worker',ttl=60)
    app = InterviewApplication('s-call-end', 'task-1', 'idea', artifact_root=str(tmp_path),pg_store=store,pg_claim=claim)
    app.progress_journal = _StubJournal()
    return app


def test_call_end_journal_entry_for_intent_call(tmp_path):
    app = _app(tmp_path)
    app._call_end_signal({"stage": "probing", "item": "IT-1", "provider": "ollama", "model": "phi4",
                          "latency_s": 14.2,
                          "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                          "keep_alive": "30m", "tokens_per_s": 18.1})
    assert app.progress_journal.rows == [
        {"phase_key": "intent", "stage": "llm_call:probing",
         "label": "Model call 'probing' [IT-1] finished", "status": "completed",
         "detail": "14.2s · 10 in / 5 out tokens · 18.1 tok/s"}]


def test_call_end_journal_entry_maps_conops_stage_and_native_timing(tmp_path):
    app = _app(tmp_path)
    app._call_end_signal({"stage": "conops_draft", "item": None, "provider": "ollama", "model": "phi4",
                          "latency_s": 66.7, "usage": {"input_tokens": 900, "output_tokens": 400, "total_tokens": 1300},
                          "timing": {"load_ms": 800, "prompt_eval_ms": 2100, "eval_ms": 11300}})
    (row,) = app.progress_journal.rows
    assert row["phase_key"] == "conops" and row["stage"] == "llm_call:conops_draft"
    assert row["label"] == "Model call 'conops_draft' finished"
    assert "load 800ms, prompt 2100ms, generate 11300ms" in row["detail"]


def test_runtime_context_wires_and_resets_the_sink(tmp_path):
    app = _app(tmp_path)
    assert mc._progress_call_end.get() is None
    with app._runtime_context():
        assert mc._progress_call_end.get() is app._call_end_signal
    assert mc._progress_call_end.get() is None
