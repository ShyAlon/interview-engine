"""2026-09-12 crash: logging a phi4 scope_tiebreak call raised
KeyError 'no price for phi4' - langchain reports provider 'chatollama',
which an exact-match ollama check missed. Local calls log $0, always."""
from interview_engine.telemetry import UsageLog, price_for


def test_chatollama_phi4_call_logs_zero_cost():
    log = UsageLog("s1")
    rec = log.record("scope_tiebreak", "chatollama", "phi4",
                     {"input_tokens": 100, "output_tokens": 20})
    assert rec.price_per_mtok_prompt_usd == 0.0
    assert rec.price_per_mtok_completion_usd == 0.0
    assert rec.cost_usd == 0.0


def test_price_table_comes_from_contracts():
    from vmodel_contracts import price_for as shared
    assert price_for is shared  # one table, one function (shared-module doctrine)
