"""2026-09-23 smoke finding: the founder pasted one canned answer at several
consecutive distinct questions and the interview accepted it as an ordinary
answer turn after turn. The repeat guard detects a repeated non-answer
deterministically: the first repeat freezes coverage (re-ask), the next trips
the existing two-attempts closeout (escalate to a visible delegation)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from interview_engine import model_config
from interview_engine.graph import build_graph
from interview_engine.repeat_guard import answers_match, founder_turns, repeat_streak
from interview_engine.telemetry import UsageLog
from tests.test_coverage import FakeModel, fake_models, start, structured

CANNED = ("Two adults use it weekly at home. Success means every chore has one "
          "owner and due date, and both people can see completion history.")


# --- unit ---

def test_exact_and_near_duplicate_match():
    assert answers_match(CANNED, CANNED)
    edited = CANNED.replace("one owner and due date", "one owner and a due date")
    assert answers_match(CANNED, edited)
    reordered = ("Success means every chore has one owner and due date, and both "
                 "people can see completion history. Two adults use it weekly at home.")
    assert answers_match(CANNED, reordered)


def test_different_answers_do_not_match():
    other = "The main flow is creating a chore, assigning it to someone, and marking it done."
    assert not answers_match(CANNED, other)


def test_short_repeats_are_conversation_not_dodges():
    assert not answers_match("yes", "yes")
    assert not answers_match("no thanks", "no thanks")
    assert not answers_match("I don't know", "I don't know")


def test_founder_turns_pairs_answers_with_questions_and_skips_intake():
    msgs = [HumanMessage(content="a chore app for couples"),
            AIMessage(content="Who uses it?"), HumanMessage(content=CANNED),
            AIMessage(content="What data?"), HumanMessage(content="Names and owners of chores.")]
    turns = founder_turns(msgs)
    assert turns == [("Who uses it?", CANNED), ("What data?", "Names and owners of chores.")]


def test_repeat_streak_counts_trailing_repeats():
    msgs = [HumanMessage(content="a chore app for couples"),
            AIMessage(content="Who uses it?"), HumanMessage(content=CANNED)]
    assert repeat_streak(msgs) == 0
    msgs += [AIMessage(content="What data matters?"), HumanMessage(content=CANNED)]
    assert repeat_streak(msgs) == 1
    msgs += [AIMessage(content="Which risks worry you?"), HumanMessage(content=CANNED)]
    assert repeat_streak(msgs) == 2
    msgs += [AIMessage(content="Anything else?"), HumanMessage(content="Something entirely new about integrations.")]
    assert repeat_streak(msgs) == 0


# --- graph: the exact smoke scenario ---

def _drive(g, thread, answer):
    return g.invoke({"messages": [HumanMessage(content=answer)]},
                    {"configurable": {"thread_id": thread}})


def test_first_repeat_freezes_coverage_by_turn_two(monkeypatch):
    # The model WOULD advance coverage from the canned answer (it mentions
    # users); the guard must refuse the advance on the repeated turn.
    probing = FakeModel([
        structured("Who uses the product day to day?", covered=["goal"]),
        structured("What information does a chore need?", covered=["users"]),
        structured("Which existing tools must it connect with?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-repeat"))
    r = start(g, thread="rep1")
    assert r["coverage"]["goal"] == "covered"

    r = _drive(g, "rep1", CANNED)          # first use: ordinary answer, advances
    assert r["coverage"]["users"] == "covered"
    assert r["repeat_strikes"] == 0

    r = _drive(g, "rep1", CANNED)          # same answer, new question: dodge
    assert r["repeat_strikes"] == 1
    assert r["coverage"]["flows"] == "uncovered"    # frozen despite model claim
    assert not r["delegated_decisions"]             # re-ask, not yet escalation


def test_second_repeat_escalates_through_existing_closeout(monkeypatch):
    probing = FakeModel([
        structured("Who uses the product day to day?", covered=["goal"]),
        structured("What information does a chore need?", covered=["users"]),
        structured("What else should we understand about data?"),   # frozen, no effect
        structured("Which existing tools must it connect with?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-repeat2"))
    start(g, thread="rep2")
    _drive(g, "rep2", CANNED)              # paste 1: ordinary answer, advances
    r = _drive(g, "rep2", CANNED)          # paste 2: dodge - freeze + re-ask
    assert r["repeat_strikes"] == 1
    assert r["coverage"]["flows"] == "uncovered"
    r = _drive(g, "rep2", CANNED)          # paste 3: second dodge - area escalates
    assert r["repeat_strikes"] == 2
    assert r["coverage"]["flows"] == "delegated"
    delegation = r["delegated_decisions"][-1]["statement"]
    assert "FOUNDER-DELEGATED" in delegation and "operational scenarios" in delegation  # _LABELS["flows"]
    assert "repeated the same non-responsive answer" in delegation


def test_fresh_answers_advance_normally(monkeypatch):
    probing = FakeModel([
        structured("Who uses the product day to day?", covered=["goal"]),
        structured("What information does a chore need?", covered=["users"]),
        structured("Which risks worry you?", covered=["data"]),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-fresh"))
    start(g, thread="fresh1")
    r = _drive(g, "fresh1", CANNED)
    r = _drive(g, "fresh1", "Chores have a title, an owner, a due date, and a completion log.")
    r = _drive(g, "fresh1", "The big risk is that one partner ignores the notifications.")
    assert r["repeat_strikes"] == 0
    assert r["coverage"]["users"] == "covered"
    assert r["coverage"]["data"] == "covered"


def test_repeated_steer_keeps_terminal_steer_handling(monkeypatch):
    # A repeated explicit steer is not a dodge: the steer path parks the topic
    # and records the delegation, and the guard stays out of it.
    probing = FakeModel([
        structured("Who uses the product day to day?", covered=["goal"]),
        structured("What specific encryption standards should protect user data?"),
        structured("Apart from parked details, what information does a chore need?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-steer"))
    start(g, thread="steer-rep")
    r = _drive(g, "steer-rep", "No. You do it.")
    assert r["repeat_strikes"] == 0
    r = _drive(g, "steer-rep", "No. You do it.")
    assert r["repeat_strikes"] == 0
    assert r["delegated_decisions"]


def test_repeat_freezes_the_categorizer_path_too(monkeypatch):
    # With the Jev/LLM categorizer active, coverage comes from classifier
    # verdicts; the freeze must gate that application as well.
    from interview_engine.categorization import ClassificationAnswer, ProviderResult

    class StubCategorizer:
        def active(self, slot):
            return slot in ("coverage", "intent")

        def classify(self, request):
            if request.slot == "scope":
                return ProviderResult({
                    "accepted": ClassificationAnswer("accepted", 0.99),
                    "archetype": ClassificationAnswer("saas_crud", 0.99)}, "stub", "stub")
            if request.slot == "coverage":
                answers = {"relevance": ClassificationAnswer("addressed", 0.9)}
                answers["covers_flows"] = ClassificationAnswer(0.9, 0.9)
                return ProviderResult(answers, "stub", "stub")
            return ProviderResult({"intent": ClassificationAnswer("answer", 0.9)}, "stub", "stub")

    probing = FakeModel([
        structured("Who uses the product day to day?", covered=["goal"]),
        structured("Please walk through one important real-world use."),
        structured("What information does a chore need?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t-cat"), categorizer=StubCategorizer())
    start(g, thread="cat1")
    r = _drive(g, "cat1", CANNED)          # first use: classifier advances flows
    assert r["coverage"]["flows"] == "covered"
    r = _drive(g, "cat1", CANNED)          # repeat: classifier verdict frozen out
    assert r["repeat_strikes"] == 1
    assert r["coverage"]["data"] == "uncovered"
