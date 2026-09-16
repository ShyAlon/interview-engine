"""2026-09-12 transcript (session 20260912-201837-bd17): the interviewer
drilled into UI frameworks and re-asked after the founder said 'No. You do
it.' and 'Can you move on from the UI?'. Steers are terminal; implementation
questions are suppressed; delegations land in the intent model."""
import json

import pytest
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
    assert r["coverage"]["goal"] == "delegated"
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


# --- the exact turn 21-23 exchange from session 20260912-201837-bd17 ---

def test_storage_drilling_is_suppressed():
    assert implementation_question(
        "What data storage solution do you envision for persisting user points across devices?")
    assert implementation_question(
        "What specific storage solution do you envision that is both simple and cost-effective "
        "for persisting user data?")


def test_persistence_deferral_parks_and_stops_the_drill(monkeypatch):
    # turn 21: behavior-flavored question, founder delegates; turns 22/23 must
    # NOT re-ask storage - the parked topic + the widened impl guard both fire
    probing = FakeModel([
        structured("How will the app handle data persistence to ensure points are saved "
                   "across devices and over time?", thin=["data"]),
        structured("What data storage solution do you envision for persisting user points "
                   "across devices?"),
        structured("What specific storage solution do you envision that is both simple "
                   "and cost-effective for persisting user data?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="steer5")
    r = g.invoke({"messages": [HumanMessage(content="You decide")]},
                 {"configurable": {"thread_id": "steer5"}})
    assert "persistence/storage detail" in r["parked_topics"]
    assert r["delegated_decisions"][0]["statement"].startswith(
        "FOUNDER-DELEGATED: persistence/storage detail")
    # turn 22's storage question is suppressed before it reaches the founder
    q22 = r["messages"][-1].content.lower()
    assert "storage" not in q22 and "persist" not in q22
    # and turn 23 after a non-steer answer is suppressed too
    r2 = g.invoke({"messages": [HumanMessage(content="Simple and cheap")]},
                  {"configurable": {"thread_id": "steer5"}})
    q23 = r2["messages"][-1].content.lower()
    assert "storage" not in q23 and "persist" not in q23


# --- session 20260912-201837-bd17, turns 24-27: saturated but never ends ---

def test_saturation_terminates_without_model_cooperation(monkeypatch):
    # turn 25: coverage hit 11/11, phi4 never set complete=true, and the
    # interview kept drilling for THREE more turns. Termination is code-owned.
    probing = FakeModel([
        structured("What fixed deadline, budget, team, platform, or compliance constraint applies?"),
        # founder answers; model covers everything but forgets complete=true
        structured("What specific free resources, like Supabase, are you considering for backend services?",
                   covered=["goal", "users", "flows", "data", "integrations", "nfrs",
                            "failures", "constraints", "risks", "done", "glossary"],
                   complete=False),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("t1"))
    start(g, thread="sat1")
    r = g.invoke({"messages": [HumanMessage(
        content="The application should use free resources (like supabase) wherever possible. "
                "There are no timeline constraints. Expect maybe 10000 users.")]},
        {"configurable": {"thread_id": "sat1"}})
    assert "[INTERVIEW COMPLETE]" in r["messages"][-1].content
    assert probing.calls == 2                      # no third drilling call
    assert r.get("intent_model") is not None       # distillation ran


def test_cloud_infrastructure_drilling_is_suppressed():
    assert implementation_question(
        "What specific free resources, like Supabase, are you considering for backend services, "
        "and what are their roles in the application?")
    assert implementation_question(
        "What specific cloud services or infrastructure components are you considering for the "
        "backend, and how do they fit into the application's architecture?")


def test_leading_steer_inside_a_longer_answer():
    assert founder_steer("You suggest. I'm good with any cloud infrastructure.") == "delegate"
    assert founder_steer("This is implementation. I just want the app to be accessible and to "
                         "persist the users' data.") == "dismiss"

# --- session 20260914-210005-68fa: historical-data question repeated 4x ---

def test_historical_data_deflection_forms_are_terminal():
    assert founder_steer("You decide") == "delegate"
    assert founder_steer("The data science team will approve the final selection") == "delegate"
    assert founder_steer("Leave the historical data") == "dismiss"


def test_all_three_live_deflections_resolve_the_current_area(monkeypatch):
    replies=["You decide", "The data science team will approve the final selection", "Leave the historical data"]
    expected=["delegated", "delegated", "delegated"]
    for i,(answer,status) in enumerate(zip(replies,expected)):
        probing=FakeModel([structured("Which historical datasets should train the model?"),
                           structured("Which historical data sources should train the model?")])
        fake_models(monkeypatch,probing); g=build_graph(UsageLog(f"live-deflect-{i}"))
        cfg={"configurable":{"thread_id":f"live-deflect-{i}"}}
        cov={k:"covered" for k,_,_ in __import__('interview_engine.coverage',fromlist=['AREAS']).AREAS}
        cov["data"]="uncovered"; cov["risks"]="thin"
        g.invoke({"messages":[HumanMessage(content="weather and port feeds")],"submission":{"free_text":"web supply chain predictor","founder_id":"t","submission_id":f"live-deflect-{i}"},"verdict":{"accepted":True},"coverage":cov,"coverage_target":"data"},cfg)
        r=g.invoke({"messages":[HumanMessage(content=answer)]},cfg)
        assert r["coverage"]["data"]==status
        assert r["delegated_decisions"]
        assert "historical" not in r["messages"][-1].content.lower()


def test_exact_historical_data_turn_sequence_resolves_and_never_reasks(monkeypatch):
    probing = FakeModel([
        structured("What specific historical data sources will train the predictive model?"),
        structured("Which historical data providers will the data science team approve?"),
        structured("What historical training datasets will be selected?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("historical-repeat"))
    cfg={"configurable":{"thread_id":"historical-repeat"}}
    cov={k:"covered" for k,_,_ in __import__('interview_engine.coverage',fromlist=['AREAS']).AREAS}
    cov["data"]="uncovered"; cov["risks"]="thin"
    initial={"messages":[HumanMessage(content="Weather, geopolitical events, port congestion and customs data")],
             "submission":{"free_text":"a web supply chain predictor","founder_id":"t","submission_id":"historical-repeat"},
             "verdict":{"accepted":True},"coverage":cov,"coverage_target":"data"}
    first=g.invoke(initial,cfg)
    assert "historical data sources" in first["messages"][-1].content.lower()
    r=g.invoke({"messages":[HumanMessage(content="You decide")]},cfg)
    assert r["coverage"]["data"]=="delegated"
    assert r["coverage_target"]=="risks"
    assert r["delegated_decisions"] and "historical-data source selection" in r["parked_topics"]
    assert "historical" not in r["messages"][-1].content.lower()
    r2=g.invoke({"messages":[HumanMessage(content="The data science team will approve the final selection")]},cfg)
    assert "historical" not in r2["messages"][-1].content.lower()


def test_coverage_credit_and_question_selector_cannot_drift(monkeypatch):
    probing = FakeModel([
        structured("What exact outcome makes this worth building?", covered=["goal"]),
        structured("Which outcome would make this product worthwhile?"),
    ])
    fake_models(monkeypatch, probing)
    g=build_graph(UsageLog("credit-retarget")); cfg={"configurable":{"thread_id":"credit-retarget"}}
    first=start(g,thread="credit-retarget")
    assert first["coverage"]["goal"]=="covered" and first["coverage_target"]=="goal"
    r=g.invoke({"messages":[HumanMessage(content="Operations managers") ]},cfg)
    assert "outcome" not in r["messages"][-1].content.lower()


def test_same_substance_question_is_reframed_after_two_attempts():
    from interview_engine.coverage import same_question_substance, unique_question
    a="What specific historical data sources will train the predictive model?"
    b="Which historical data sources will be used to train the prediction model?"
    assert same_question_substance(a,b)
    q=unique_question(b,[{"area":"data","question":a}],"data")
    assert not same_question_substance(q,a)


def test_two_failed_attempts_become_visible_assumption_and_move_on(monkeypatch):
    probing=FakeModel([
        structured("What exact business risk remains?", thin=["risks"]),
        structured("Which assumption could kill the product?", thin=["risks"]),
        structured("What result proves the product is ready?"),
    ])
    fake_models(monkeypatch,probing); g=build_graph(UsageLog("two-attempts")); cfg={"configurable":{"thread_id":"two-attempts"}}
    cov={k:"covered" for k,_,_ in __import__('interview_engine.coverage',fromlist=['AREAS']).AREAS}
    cov["risks"]="uncovered"; cov["done"]="thin"
    initial={"messages":[HumanMessage(content="web product")],"submission":{"free_text":"web product","founder_id":"t","submission_id":"two-attempts"},"verdict":{"accepted":True},"coverage":cov,"coverage_target":"risks"}
    g.invoke(initial,cfg)
    g.invoke({"messages":[HumanMessage(content="It is unclear today")]},cfg)
    r=g.invoke({"messages":[HumanMessage(content="We still need evidence")]},cfg)
    assert r["coverage"]["risks"]=="delegated"
    assumption=next(x for x in r["delegated_decisions"] if "after two interview attempts" in x["statement"])
    assert assumption["source_turns"] and "attempted_questions" not in assumption
    assert r["coverage_target"]=="done"
    assert "ready" in r["messages"][-1].content.lower()
    # End to end: the same delegated assumption must survive intent assembly
    # and validate against the current vmodel-contracts schema.
    done=g.invoke({"messages":[HumanMessage(content="enough")]},cfg)
    from vmodel_contracts import IntentModel
    intent=IntentModel.model_validate(done["intent_model"])
    auto=next(a for a in intent.assumption_items if "after two interview attempts" in a.statement)
    assert auto.source_turns


def test_resume_repairs_bb5b8d4_invalid_delegation_checkpoint(monkeypatch):
    """The user's held session already contains the old invalid record."""
    probing=FakeModel([structured("closing", covered=[k for k,_,_ in __import__('interview_engine.coverage',fromlist=['AREAS']).AREAS], complete=True)])
    fake_models(monkeypatch,probing); g=build_graph(UsageLog("legacy-checkpoint")); cfg={"configurable":{"thread_id":"legacy-checkpoint"}}
    messages=[HumanMessage(content="web supply chain predictor"),
              AIMessage(content="What exact risk remains?"), HumanMessage(content="It is unclear"),
              AIMessage(content="Which assumption could kill it?"), HumanMessage(content="We need evidence")]
    old={"statement":"FOUNDER-DELEGATED: risks - after two interview attempts, the remaining detail is left open.",
         "source_turns":[],"attempted_questions":["What exact risk remains?","Which assumption could kill it?"]}
    cov={k:"covered" for k,_,_ in __import__('interview_engine.coverage',fromlist=['AREAS']).AREAS}
    g.invoke({"messages":messages,"submission":{"free_text":"web supply chain predictor","founder_id":"t","submission_id":"legacy-checkpoint"},
              "verdict":{"accepted":True},"coverage":cov,"coverage_target":"risks","delegated_decisions":[old]},cfg)
    r=g.invoke({"messages":[HumanMessage(content="enough")]},cfg)
    from vmodel_contracts import IntentModel
    intent=IntentModel.model_validate(r["intent_model"])
    fixed=next(a for a in intent.assumption_items if "after two interview attempts" in a.statement)
    assert fixed.source_turns==[4]
    dumped=fixed.model_dump(mode="json")
    assert "attempted_questions" not in dumped

def test_external_fact_and_quality_method_questions_are_never_sent_to_founder():
    assert implementation_question("Which APIs exist for real-time port data?")
    assert implementation_question("How will model quality be verified?")
    assert implementation_question("What quality metrics should verify the predictive model?")
    for question in ("Which APIs exist for real-time port data?", "How will model quality be verified?"):
        assert guard_question(question, [], "data") != question


def test_delegation_carries_grounded_research_brief(monkeypatch):
    probing=FakeModel([structured("Which APIs exist for real-time port data?"), structured("What data must arrive in real time?")])
    fake_models(monkeypatch,probing); g=build_graph(UsageLog("research-brief")); cfg={"configurable":{"thread_id":"research-brief"}}
    start(g,thread="research-brief")
    r=g.invoke({"messages":[HumanMessage(content="You decide")]},cfg)
    assert r["delegated_decisions"][0]["research_brief"]
    assert "current" in r["delegated_decisions"][0]["research_brief"]


@pytest.mark.parametrize("answer", [
    "That's for the subject matter experts to answer, but it's structured and well defined.",
    "It doesn't need to validate the registration number.",
    "Not sure - the insurance experts should define it.",
])
def test_expert_deflection_forms_delegate_and_close_thread(monkeypatch, answer):
    probing = FakeModel([
        structured("Which registration-number rules must the claim form validate?"),
        structured("What operational scenario should happen first?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("expert-deflection"))
    cfg = {"configurable": {"thread_id": "expert-deflection-" + str(abs(hash(answer)))}}
    cov = {k: "covered" for k, _, _ in __import__('interview_engine.coverage', fromlist=['AREAS']).AREAS}
    cov["data"] = "uncovered"
    cov["flows"] = "thin"
    initial = {"messages": [HumanMessage(content="garage insurance claims")],
               "submission": {"free_text": "garage claims", "founder_id": "t", "submission_id": "expert-deflection"},
               "verdict": {"accepted": True}, "coverage": cov, "coverage_target": "data"}
    g.invoke(initial, cfg)
    result = g.invoke({"messages": [HumanMessage(content=answer)]}, cfg)
    assert result["coverage"]["data"] == "delegated"
    assert result["coverage_target"] == "flows"
    assert result["delegated_decisions"][0]["research_brief"]
    assert "registration" not in result["messages"][-1].content.lower()


def test_direct_move_on_closes_current_subject_not_whole_interview(monkeypatch):
    probing = FakeModel([
        structured("Which registration-number rules must be checked?"),
        structured("What is the main claim workflow?"),
    ])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("move-on-subject"))
    cfg = {"configurable": {"thread_id": "move-on-subject"}}
    cov = {k: "covered" for k, _, _ in __import__('interview_engine.coverage', fromlist=['AREAS']).AREAS}
    cov["data"] = "uncovered"
    cov["flows"] = "thin"
    initial = {"messages": [HumanMessage(content="garage claims")],
               "submission": {"free_text": "garage claims", "founder_id": "t", "submission_id": "move-on-subject"},
               "verdict": {"accepted": True}, "coverage": cov, "coverage_target": "data"}
    g.invoke(initial, cfg)
    result = g.invoke({"messages": [HumanMessage(content="Move on.")]}, cfg)
    assert "[INTERVIEW COMPLETE]" not in result["messages"][-1].content
    assert result["coverage"]["data"] == "delegated"
    assert result["coverage_target"] == "flows"


def test_followup_cap_counts_only_consecutive_same_area(monkeypatch):
    probing = FakeModel([structured("What is still missing?")])
    fake_models(monkeypatch, probing)
    g = build_graph(UsageLog("consecutive-cap"))
    cfg = {"configurable": {"thread_id": "consecutive-cap"}}
    cov = {k: "covered" for k, _, _ in __import__('interview_engine.coverage', fromlist=['AREAS']).AREAS}
    cov["data"] = "thin"
    state = {"messages": [HumanMessage(content="A useful answer")],
             "submission": {"free_text": "garage claims", "founder_id": "t", "submission_id": "consecutive-cap"},
             "verdict": {"accepted": True}, "coverage": cov, "coverage_target": "data",
             "question_history": [
                 {"area": "data", "question": "Old data question?"},
                 {"area": "risks", "question": "Intervening risk question?"},
                 {"area": "data", "question": "Current data question?"},
             ]}
    result = g.invoke(state, cfg)
    assert result["coverage"]["data"] == "thin"
    assert not any("after two interview attempts" in d["statement"]
                   for d in result.get("delegated_decisions", []))
