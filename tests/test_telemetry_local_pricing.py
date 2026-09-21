"""2026-09-12 crash: logging a phi4 scope_tiebreak call raised
KeyError 'no price for phi4' - langchain reports provider 'chatollama',
which an exact-match ollama check missed. Local calls log $0, always."""
from interview_engine.telemetry import UsageLog, price_for
import pytest


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


def test_unpriced_remote_usage_is_metered_without_becoming_free():
    log=UsageLog("s")
    rec=log.record_unpriced("categorization_scope","typesafe","jev",{"input_tokens":12,"output_tokens":3},.25)
    assert rec.cost_usd is None and log.durations==[.25]
    stage=log.summary()["stages"]["categorization_scope"]
    assert stage["calls"]==1 and stage["unpriced_calls"]==1 and stage["cost_usd"]==0.0


def test_estimated_rate_records_measured_cost_without_price_table():
    log=UsageLog('s');rec=log.record_estimated('categorization_scope','typesafe','jev',{'input_tokens':9000,'output_tokens':300},.0322580645,.1)
    assert rec.cost_usd == pytest.approx(.0003)
