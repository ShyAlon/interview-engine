"""Run-3 defect regression tests: the mechanical guards must catch what prompts didn't."""
from interview_engine.distill_qa import (content_words, dedup_constraints,
                                         ground_assumptions, is_solution_language)

TRANSCRIPT = [
    "[0] human: A supply chain optimization web application",
    "[1] ai: who uses it?",
    "[2] human: SC managers and logistics execs",
    "[3] human: external APIs can stop transmitting - we must operate with partial data",
    "[4] human: the founder said the mobile app is for later, not this product",
]

PROMPTS = ["assumptions[]: things the founder ASSERTED without evidence "
           "('these sources can't conflict', 'users will always have connectivity')"]


def test_prompt_echo_assumption_dropped():
    items = [{"statement": "these sources can't conflict with each other", "source_turns": [2]}]
    kept, dropped = ground_assumptions(items, TRANSCRIPT, PROMPTS)
    assert kept == [] and dropped


def test_contradicted_assumption_dropped():
    items = [{"statement": "users will always have connectivity", "source_turns": [3]}]
    kept, dropped = ground_assumptions(items, TRANSCRIPT, PROMPTS)
    assert kept == [] and dropped


def test_grounded_assumption_kept():
    items = [{"statement": "external APIs can stop transmitting", "source_turns": [3]}]
    kept, dropped = ground_assumptions(items, TRANSCRIPT, PROMPTS)
    assert len(kept) == 1 and dropped == []


def test_out_of_range_turns_dropped():
    items = [{"statement": "external APIs can stop transmitting", "source_turns": [42]}]
    kept, dropped = ground_assumptions(items, TRANSCRIPT, PROMPTS)
    assert kept == []


def test_constraint_dedup_run3_examples():
    needs = [
        "SSO and MFA must be enforced for all enterprise users",
        "no PII may be persisted anywhere in the system",
        "the app must be simple for non-technical users to operate",
        "easy scale-out as customers grow",
        "see stock levels per warehouse in one dashboard",
    ]
    constraints = [
        "SSO with MFA mandated", "no PII may be stored", "keep the UI simple",
        "must scale out easily", "$0 infrastructure budget during the demo",
    ]
    kept, dropped = dedup_constraints(constraints, needs)
    assert kept == ["$0 infrastructure budget during the demo"]
    assert len(dropped) == 4


def test_solution_language_detection():
    assert is_solution_language("GET /dashboard returns a JSON response with 'status' field")
    assert is_solution_language("POST /login returns 401 Unauthorized")
    assert not is_solution_language("a seeded conflict produces exactly one alert within the refresh cycle")
    assert not is_solution_language("data older than 24h is flagged stale on screen")


def test_renderer_splits_explicit_and_candidate_exclusions():
    from interview_engine.rendering import render_intent_markdown
    from vmodel_contracts import IntentModel
    intent = IntentModel(
        session_id="t", problem_statement="help managers predict and preempt supply issues",
        target_users=["managers"], core_value_proposition="predict and preempt",
        stakeholder_needs=[{"need_id": "SN-1", "statement": "see stock", "kind": "functional"}],
        in_scope=["dashboard"], out_of_scope=["legacy flat"],
        scope_exclusions=[
            {"item": "mobile apps", "source_turns": [3], "explicit": True},
            {"item": "predictive analytics beyond basic monitoring", "explicit": False},
        ],
        source_turns={})
    md = render_intent_markdown(intent)
    assert "**Out of scope**" in md and "mobile apps" in md
    assert "Candidate exclusions - inferred" in md
    assert "predictive analytics" in md and "confirm this is not the core value" in md


def test_renderer_new_sections():
    from interview_engine.rendering import render_intent_markdown
    from vmodel_contracts import IntentModel
    intent = IntentModel(
        session_id="t", problem_statement="p", target_users=["u"], core_value_proposition="v",
        stakeholder_needs=[{"need_id": "SN-1", "statement": "x", "kind": "functional"}],
        in_scope=["a"], out_of_scope=["b"], source_turns={},
        assumption_items=[{"statement": "vendors are always honest", "source_turns": [7]}],
        done_criteria=["customer onboarded unaided"],
        operational_concepts=["support team handles failures"])
    md = render_intent_markdown(intent)
    assert "vendors are always honest (turn 7)" in md
    assert "Done criteria" in md and "Operational concepts" in md


def test_verify_tech_preferences_grounded_recognized():
    from interview_engine.distill_qa import verify_tech_preferences
    kept, dropped, warnings = verify_tech_preferences([
        {"area": "language", "preference": "python", "strength": "required", "source_turns": [4]}])
    assert len(kept) == 1 and not dropped and not warnings


def test_verify_tech_preferences_ungrounded_dropped():
    from interview_engine.distill_qa import verify_tech_preferences
    kept, dropped, _ = verify_tech_preferences([
        {"area": "cloud", "preference": "AWS", "strength": "preferred", "source_turns": []}])
    assert not kept and len(dropped) == 1 and "no source turns" in dropped[0]


def test_verify_tech_preferences_unrecognized_warned_but_kept():
    from interview_engine.distill_qa import verify_tech_preferences
    kept, _, warnings = verify_tech_preferences([
        {"area": "cloud", "preference": "zorbacloud", "strength": "preferred", "source_turns": [2]}])
    assert len(kept) == 1  # niche-but-real exists: keep, warn
    assert warnings and "not a recognized" in warnings[0]


def test_agnostic_is_an_answer_not_silence():
    from vmodel_contracts import TechPreference
    tp = TechPreference(area="language", preference="agnostic", strength="preferred", source_turns=[1])
    assert tp.preference == "agnostic"
