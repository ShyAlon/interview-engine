import pytest
from langchain_core.messages import HumanMessage

from interview_engine import question_config as qc
from interview_engine import session_retrieval as sr
from interview_engine.coverage import AREA_KEYS, status_block
from interview_engine.graph import build_graph
from interview_engine.prompts import INTERVIEWER_SYSTEM, STRUCTURED_TURN_FORMAT
from interview_engine.telemetry import UsageLog
from tests.test_coverage import FakeModel, fake_models, structured

TURNS = [(0, "a kids music app with songs and star animations"),
         (2, "We write every song ourselves. Parents sign in to see progress."),
         (4, "It must work offline on a tablet.")]


def test_spans_split_on_sentences_with_turn_ids():
    spans = sr.build_spans(TURNS)
    assert {"id": "t2", "uid": "t2.0", "turn": 2, "text": "We write every song ourselves."} in spans
    assert all(s["id"] == f't{s["turn"]}' for s in spans)


def test_repeated_sentences_are_one_span():
    spans = sr.build_spans([(1, "Songs come from our store. Songs come from our store."), (3, "songs come from our STORE.")])
    assert len(spans) == 1


def test_ranking_uses_area_cues_then_recency_and_drops_no_hit_spans():
    ranked = sr.rank_spans(sr.build_spans(TURNS), "data")
    assert ranked and all(s["score"] > 0 for s in ranked)
    assert ranked[0]["text"].startswith(("We write every song", "a kids music app"))
    none = sr.rank_spans(sr.build_spans([(0, "Hello there friend, nice day")]), "data")
    assert none == []


def test_block_quotes_spans_with_ids_and_empty_when_nothing_relevant():
    block = sr.retrieval_block(TURNS, "data")
    assert "SESSION FACTS" in block and '[t2] "We write every song ourselves."' in block
    assert sr.retrieval_block([(0, "Hello there friend, nice day")], "data") == ""
    assert sr.retrieval_block(TURNS, None) == ""


def test_block_never_exceeds_word_or_char_cap_and_drops_lowest_ranked_first():
    long_turns = [(i, f"song {i} data " + "word " * 30 + "from our own store.") for i in range(1, 9)]
    block = sr.retrieval_block(long_turns, "data")
    assert len(block.split()) <= 90 and len(block) <= 600
    cfg = qc.validate({"retrieval": {"max_words": 30, "max_chars": 200}})
    small = sr.retrieval_block(long_turns, "data", cfg)
    assert 0 < len(small.split()) <= 30 and len(small) <= 200
    one = sr.retrieval_block([(1, "We write every song ourselves " + "and then " * 60)], "data")
    assert one and len(one.split()) <= 90 and len(one) <= 600


def test_top_k_and_cues_come_from_config():
    cfg = qc.validate({"retrieval": {"top_k": 1}, "area_cues": {"data": ["zebra"]}})
    spans = sr.build_spans([(1, "The zebra list lives in a file."), (2, "We write every song ourselves.")])
    ranked = sr.rank_spans(spans, "data", cfg)
    assert [s["id"] for s in ranked] == ["t1"]


def test_unknown_area_in_cue_config_is_rejected():
    with pytest.raises(qc.ConfigError):
        qc.validate({"area_cues": {"bogus": ["x"]}})


def test_static_prompt_budget_pin_is_untouched_by_the_dynamic_block():
    static = INTERVIEWER_SYSTEM + "\n\n" + status_block({k: "uncovered" for k in AREA_KEYS}) + "\n\n" + STRUCTURED_TURN_FORMAT
    assert len(static) <= 3_200 and len(static.split()) <= 460


def test_full_dynamic_system_message_stays_bounded(monkeypatch):
    seen = {}
    class Spy(FakeModel):
        def invoke(self, messages):
            seen["system"] = messages[0].content
            return super().invoke(messages)
    cov = {k: "covered" for k in AREA_KEYS}; cov["data"] = "uncovered"
    fake_models(monkeypatch, Spy([structured("What does it keep?")]))
    idea = "a kids music app. We write every song ourselves. " + "More songs and images from our own store. " * 20 + "Kids hear tunes from the bundled library."
    build_graph(UsageLog("sr-bound")).invoke({
        "messages": [HumanMessage(content=idea)],
        "submission": {"free_text": idea, "founder_id": "t", "submission_id": "sr-bound"},
        "verdict": {"accepted": True}, "coverage": cov, "coverage_target": "data",
    }, {"configurable": {"thread_id": "sr-bound"}})
    system = seen["system"]
    assert "SESSION FACTS" in system and '[t0] "We write every song ourselves."' in system
    assert system.count("More songs and images from our own store.") == 1
    facts = system[system.index("SESSION FACTS"):].split("\n\n")[0]
    assert len(facts.split()) <= 90 and len(facts) + 2 <= 600
    base = INTERVIEWER_SYSTEM + "\n\n" + status_block(cov) + "\n\n" + STRUCTURED_TURN_FORMAT
    assert len(system) <= len(base) + 600
