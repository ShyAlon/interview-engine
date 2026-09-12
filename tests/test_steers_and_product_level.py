"""2026-09-12 transcript (session 20260912-201837-bd17): the interviewer
drilled into UI frameworks and re-asked after the founder said 'No. You do
it.' and 'Can you move on from the UI?'. Steers are terminal; implementation
questions are suppressed; delegations land in the intent model."""
import json

from langchain_core.messages import AIMessage, HumanMessage

from interview_engine import model_config
from interview_engine.coverage import founder_steer, guard_question, implementation_question
from interview_engine.graph import build_graph
from interview_engine.telemetry import UsageLog
from tests.test_coverage import NEEDS, SCOPE, FakeModel, fake_models, start, structured


# --- unit: steer detection ---

def test_steer_detection_matches_the_real_transcript():
    assert founder_steer("No. You do it.") == "delegate"
    assert founder_steer("You do it") == "delegate"
    assert founder_steer("Can you move on from the UI?") == "dismiss"
    assert founder_steer("next topic") == "dismiss"
    assert founder_steer("not up to me") == "delegate"
    # ordinary answers are NOT steers
    assert founder_steer("Walking 10000 steps daily, meditating") is None
    assert founder_steer("The user can assign points to each item") is None


def test_implementation_question_guard():
    assert implementation_question("Which UI framework do you envision for the dashboard?")
    assert implementation_question("What JavaScript library should handle charts?")
    assert implementation_question("Which database engine should store the points?")
    # product-level technology questions stay legal (the constraints area)
    assert not implementation_question("Any technology constraints - language/ecosystem, cloud affinity?")
    assert not implementation_question("What existing systems must it integrate with?")


def test_guard_question_reroutes_implementation_to_product_level():
    q = guard_question("Which UI framework do you envision?", [], "flows")
    assert "framework" not in q.lower()
    assert q  # a product-level fallback for the target area


# --- transcript level: a steer closes the topic, no re-ask ---

def test_delegate_records_delegation_and_never_reasks(monkeypatch):
    probing = FakeModel([
        structured("What specific encryption standards should protect user data?", thin=["nfrs"]),
        structured("Which encryption protocol, such as TLS 1.3, should be used?"),  # must be suppressed
        structured("Apart from parked details, what uptime target matters most?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="steer1")
    r = g.invoke({"messages": [HumanMessage(content="No. You do it.")]},
                 {"configurable": {"thread_id": "steer1"}})
    # delegation recorded for the intent model, topic parked
    assert r["delegated_decisions"]
    d = r["delegated_decisions"][0]
    assert d["statement"].startswith("FOUNDER-DELEGATED: security/encryption detail")
    assert "No. You do it." in d["statement"]
    assert d["source_turns"]
    assert "security/encryption detail" in r["parked_topics"]
    # the immediate re-ask attempt was suppressed by the parked-topic guard
    assert "encryption" not in r["messages"][-1].content.lower()
    # the area stays honestly thin - delegated is not covered
    assert r["coverage"]["nfrs"] == "thin"


def test_dismiss_with_topic_parks_but_keeps_area(monkeypatch):
    probing = FakeModel([
        structured("What specific encryption standards should be used?", thin=["nfrs"]),
        structured("Apart from parked details, what uptime target matters most?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="steer2")
    r = g.invoke({"messages": [HumanMessage(content="Can you move on from the encryption?")]},
                 {"configurable": {"thread_id": "steer2"}})
    assert "security/encryption detail" in r["parked_topics"]
    assert r["delegated_decisions"]
    assert r["coverage"]["nfrs"] == "thin"      # topic closed, area not falsely skipped
    assert "encryption" not in r["messages"][-1].content.lower()


def test_dismiss_without_topic_skips_the_area(monkeypatch):
    probing = FakeModel([
        structured("What outcome would make this product worth building?", thin=["goal"]),
        structured("Who besides the primary user is affected?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="steer2b")
    r = g.invoke({"messages": [HumanMessage(content="Can we move on?")]},
                 {"configurable": {"thread_id": "steer2b"}})
    assert r["coverage"]["goal"] == "skipped"
    assert r["messages"][-1].content == "Who besides the primary user is affected?"


def test_interviewer_implementation_question_is_replaced(monkeypatch):
    # even with NO founder steer, an implementation-choice question never
    # reaches the founder - it is swapped for the product-level fallback
    probing = FakeModel([structured("Which programming language should the backend use?")])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    r = start(g, thread="steer3")
    assert "language" not in r["messages"][-1].content.lower() or "framework" not in r["messages"][-1].content.lower()
    assert r["messages"][-1].content != "Which programming language should the backend use?"


def test_delegation_reaches_the_intent_model(monkeypatch):
    # end-to-end: delegate, finish, and the intent model carries the record
    probing = FakeModel([
        structured("What specific encryption standards should be used?", thin=["nfrs"]),
        structured(
            "closing", covered=["goal", "users", "flows", "data", "integrations",
                                "failures", "constraints", "risks", "done", "glossary"],
            skipped=["nfrs"],  # delegated detail - the quantified-NFR guard holds it out of covered
            complete=True),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="steer4")
    g.invoke({"messages": [HumanMessage(content="You do it.")]},
             {"configurable": {"thread_id": "steer4"}})
    r = g.invoke({"messages": [HumanMessage(content="nothing else")]},
                 {"configurable": {"thread_id": "steer4"}})
    im = r.get("intent_model") or {}
    stmts = [a.get("statement", "") for a in im.get("assumption_items", [])]
    assert any(s.startswith("FOUNDER-DELEGATED:") for s in stmts), stmts
