import pytest

import interview_engine.model_config as mc
@pytest.fixture(autouse=True)
def _reset_llm_call_budget(monkeypatch):
    """Each test is one run: the run-wide LLM call budget starts clean."""
    monkeypatch.setattr(mc, "_CALLS_USED", 0)


@pytest.fixture(autouse=True)
def _no_dotenv_files(monkeypatch):
    """No test reads a developer's .env files: model routes come from the code defaults and the test's own settings.
    A test that needs a .env builds its own and resets mc._DOTENV itself."""
    monkeypatch.setattr(mc, "_DOTENV", {})


_PG_SERVERS = []


def managed_pg_server(path):
    """pgserver.get_server tracked for per-test shutdown. Each test's tmp_path
    yields a NEW postgres instance; with only atexit cleanup a suite run kept
    every instance alive concurrently and OOMed the host (~100 orphaned
    postmasters). Per-test cleanup keeps concurrent instances bounded to one
    test's own servers."""
    import pgserver
    server = pgserver.get_server(str(path))
    _PG_SERVERS.append(server)
    return server


@pytest.fixture(autouse=True)
def _cleanup_pg_servers():
    yield
    while _PG_SERVERS:
        server = _PG_SERVERS.pop()
        try:
            server.cleanup()
        except Exception:
            pass
