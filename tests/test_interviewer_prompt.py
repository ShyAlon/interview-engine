"""Behavior and budget pins for the founder-facing interviewer prompt."""
from interview_engine.coverage import initial_coverage, status_block
from interview_engine.prompts import INTERVIEWER_SYSTEM, STRUCTURED_TURN_FORMAT
from vmodel_contracts import DELEGATION_DOCTRINE, PRODUCT_SCOPE_DEFINITION


def assembled():
    return INTERVIEWER_SYSTEM + "\n\n" + status_block(initial_coverage()) + "\n\n" + STRUCTURED_TURN_FORMAT


def test_interviewer_prompt_keeps_behavioral_obligations():
    prompt = INTERVIEWER_SYSTEM.lower()
    obligations = (
        "one short, concrete", "next target", "load-bearing answer is vague",
        "actors, flows, rules", "never implementation choices",
        "numeric latency/throughput/uptime", "business impact",
        "stakeholder classes", "data/privacy", "failure/recovery", "security",
        "press once", "record an assumption", '"skip"', '"enough"',
        '"you decide"', '"move on"', "complete=true", "never invent facts",
    )
    for obligation in obligations:
        assert obligation in prompt, obligation
    assert DELEGATION_DOCTRINE in INTERVIEWER_SYSTEM
    assert PRODUCT_SCOPE_DEFINITION in INTERVIEWER_SYSTEM


def test_interviewer_output_contract_keeps_exact_shape_and_all_areas():
    assert "Reply ONLY with JSON" in STRUCTURED_TURN_FORMAT
    for field in ("question", "covered", "thin", "skipped", "complete"):
        assert f'"{field}"' in STRUCTURED_TURN_FORMAT
    for area in initial_coverage():
        assert area in STRUCTURED_TURN_FORMAT


def test_interviewer_prompt_has_a_hard_runtime_budget():
    # Before compression the assembled system prompt was 7,885 chars / ~1,182
    # whitespace tokens. Keep enough headroom for clarity without regressing.
    prompt = assembled()
    assert len(prompt) <= 3_200
    assert len(prompt.split()) <= 450
