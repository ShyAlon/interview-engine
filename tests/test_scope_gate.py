"""Deterministic gate behavior: no LLM in these tests - tiebreaker is mocked."""
from vmodel_contracts import IntakeSubmission
from interview_engine.scope_gate import verdict_deterministic
from vmodel_contracts import AppArchetype


def sub(text: str) -> IntakeSubmission:
    return IntakeSubmission(free_text=text, founder_id="t", submission_id="t1")


def test_clear_web_app_accepted():
    v = verdict_deterministic(sub("A SaaS web app where surf schools manage bookings and customers log in"))
    assert v and v.accepted and v.archetype == AppArchetype.SAAS_CRUD
    assert not v.llm_tiebreak_used


def test_native_mobile_declined_politely():
    v = verdict_deterministic(sub("A native iOS app distributed through the app store"))
    assert v and not v.accepted
    assert v.archetype == AppArchetype.NOT_WEB_APP
    assert v.decline_message and "web applications" in v.decline_message


def test_ambiguous_goes_to_tiebreaker():
    assert verdict_deterministic(sub("Something new for surfers")) is None


def test_hardware_declined():
    v = verdict_deterministic(sub("An embedded firmware controller for drone sensors"))
    assert v and not v.accepted
