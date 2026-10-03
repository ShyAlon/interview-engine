"""A reply that only says the question was not understood never advances coverage
(JEV-D12 3.3.3; session 42b71e2e closed data on "I don't understand the question")."""
import pytest
from langchain_core.messages import HumanMessage

from interview_engine.graph import build_graph
from interview_engine.non_answer import is_non_answer
from interview_engine.telemetry import UsageLog
from tests.test_coverage import fake_models, FakeModel, start, structured
from tests.test_partial_drill import Cat

NON_ANSWERS = ["I don't understand the question", "I don't understand the question.", "i dont understand", "what?",
               "?", "??", "huh", "sorry?", "I'm not sure what you mean", "not sure what you mean",
               "What do you mean?", "can you rephrase?", "Please clarify", "the question is unclear",
               "I do not get it", "I don't get the question", "  ", "", "no idea what you mean", "unclear"]
REAL_ANSWERS = ["The developer.", "Kids learn songs from a bundled catalog.", "I don't understand why we need accounts, "
                "but kids log in with a parent's email", "What matters is speed", "not sure, maybe offline",
                "stars animation, for example", "no", "yes", "I don't know the budget yet, around 5k",
                "we don't need logins"]


@pytest.mark.parametrize("text", NON_ANSWERS)
def test_non_answers_are_detected(text):
    assert is_non_answer(text)


@pytest.mark.parametrize("text", REAL_ANSWERS)
def test_real_answers_are_not_non_answers(text):
    assert not is_non_answer(text)


def _drive(g, thread, answer):
    return g.invoke({"messages": [HumanMessage(content=answer)]}, {"configurable": {"thread_id": thread}})


def test_non_answer_never_closes_the_area_even_if_jev_says_covered(monkeypatch):
    probing = FakeModel([structured(f"question {i}?", covered=["goal"]) for i in range(6)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-nonanswer"), categorizer=Cat([("addressed", "goal")]))
    r0 = start(g, thread="na1")
    assert r0["coverage_target"] == "goal"
    r1 = _drive(g, "na1", "I don't understand the question")
    assert r1["coverage"]["goal"] == "uncovered"          # Jev said answered/covered; code ignores it
    assert r1["coverage_target"] == "goal"                # same area is re-asked
    asked = r1["messages"][-1].content
    assert asked.startswith("Sorry, that question was not clear.") and "that use" not in asked


def test_non_answer_without_jev_ignores_the_models_covered_claim(monkeypatch):
    probing = FakeModel([structured("first?"), structured("again?", covered=["goal"])])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-nonanswer2"))
    start(g, thread="na2")
    r1 = _drive(g, "na2", "what?")
    assert r1["coverage"]["goal"] != "covered"
    assert r1["messages"][-1].content.startswith("Sorry, that question was not clear.")


def test_real_answer_still_advances(monkeypatch):
    probing = FakeModel([structured(f"question {i}?") for i in range(4)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-realanswer"), categorizer=Cat([("addressed", "goal")]))
    start(g, thread="na3")
    r1 = _drive(g, "na3", "Fewer missed calls for the garage.")
    assert r1["coverage"]["goal"] == "covered"


def test_two_non_answers_hit_the_existing_two_attempt_closeout(monkeypatch):
    probing = FakeModel([structured(f"question {i}?") for i in range(8)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-nonanswer4"), categorizer=Cat([("dodged", None), ("dodged", None)]))
    start(g, thread="na4")
    r1 = _drive(g, "na4", "huh?")
    r2 = _drive(g, "na4", "what?")
    assert r1["coverage"]["goal"] == "uncovered"
    assert r2["coverage"]["goal"] == "delegated" and any("after two interview attempts" in d["statement"]
                                                       for d in r2["delegated_decisions"])


def test_data_gate_runs_on_the_jev_path(monkeypatch):
    """Jev marks data covered while the founder named content ('songs') but no source."""
    probing = FakeModel([structured("what data?"), structured("next?")])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-gate-jev"), categorizer=Cat([("addressed", "data")]))
    start(g, "a kids music PWA with songs and star animations", thread="gj1")
    r1 = _drive(g, "gj1", "Kids play the songs and get stars.")
    assert r1["coverage"]["data"] == "thin"          # was 'covered' before the gate ran on this path


class SourceAwareModel(FakeModel):
    """Reports a verified provenance quote only on the turn that follows the founder's answer."""
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        humans = [m.content for m in messages if getattr(m, "type", "") == "human"]
        if any("we write every song ourselves" in h for h in humans[:-1] or humans):
            return AIMessage(content='{"question":"next?","covered":[],"thin":[],"skipped":[],"complete":false,'
                                     '"data_source":"founder_supplied","source_quote":"we write every song ourselves"}')
        return super().invoke(messages)


def test_data_gate_lets_verified_provenance_close_on_the_jev_path(monkeypatch):
    probing = SourceAwareModel([structured("what data?") for _ in range(4)])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-gate-jev2"), categorizer=Cat([("addressed", "data")]))
    start(g, "a kids music PWA with songs and star animations", thread="gj2")
    r1 = _drive(g, "gj2", "we write every song ourselves")
    assert r1["coverage"]["data"] == "covered"
