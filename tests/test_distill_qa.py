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


FOUNDER = {2: "I want to build something like Duolingo for piano, working title Keys16", 4: "no competitors yet"}


def test_reference_names_grounded_kept_and_normalized():
    from interview_engine.distill_qa import verify_reference_names
    kept, dropped = verify_reference_names([
        {"name": "Duolingo", "kind": "reference_product", "source_turns": [2]},
        {"name": "keys16", "kind": "working_title", "source_turns": [2]},
        {"name": "DUOLINGO", "kind": "reference_product", "source_turns": [2]},   # case duplicate collapses
    ], FOUNDER)
    assert [k["name"] for k in kept] == ["Duolingo", "keys16"] and dropped == []


def test_reference_names_ungrounded_are_dropped_with_a_reason():
    from interview_engine.distill_qa import verify_reference_names
    kept, dropped = verify_reference_names([
        {"name": "Duolingo", "kind": "competitor", "source_turns": []},              # no turns
        {"name": "Duolingo", "kind": "rival", "source_turns": [2]},                  # bad kind
        {"name": "Simply Piano", "kind": "competitor", "source_turns": [2]},         # not in the cited turn
        {"name": "Duolingo", "kind": "competitor", "source_turns": [3]},             # turn 3 is not a founder turn
        {"name": "Duo", "kind": "competitor", "source_turns": [2]},                  # substring, not a whole word
        {"name": "", "kind": "competitor", "source_turns": [2]},
        "Duolingo",
    ], FOUNDER)
    assert kept == [] and len(dropped) == 7


def test_scope_prompt_asks_for_reference_names_and_forbids_inference():
    from interview_engine.prompts import DISTILL_SCOPE_SYSTEM
    assert "reference_names[]" in DISTILL_SCOPE_SYSTEM and "Never infer or guess a name" in DISTILL_SCOPE_SYSTEM


def test_intent_model_accepts_and_defaults_reference_names():
    from vmodel_contracts import IntentModel, ReferenceName
    assert IntentModel.model_fields["reference_names"].default_factory() == []
    assert ReferenceName(name="Keys16", kind="working_title", source_turns=[2]).kind.value == "working_title"


# A real session's classified needs and constraints (a tabletop-game web app). The rule
# flagged three of five constraints; all three restate a need.
_REAL_NEEDS = [
    ("SN-3", "The Virtual DM works with the SRD 3.5 rules and should stay consistent with the SRD 3.5 rules and the module's text as much as possible."),
    ("SN-4", "The Virtual DM should accept adventure modules written to SRD 3.5 standards as input."),
    ("SN-5", "The web application should support multiple devices."),
    ("SN-6", "The application should provide a playful experience, such as dice rolling, sounds of battles, trumpets when leveling up."),
    ("SN-13", "The SRD will be baked into the code, and so will the audio assets and voice libraries (nothing is exchanged with existing external systems or services)."),
    ("SN-21", "Because the game is turn based, a short delay is acceptable in keeping every player's device in sync."),
    ("SN-22", "The players create their characters by hand, and the application saves the state of the characters as well as the state of the adventure."),
    ("SN-23", "The app should validate the players' characters according to the SRD rules (for example, strength can't be above a threshold for a specific race)."),
]
_REAL_CONSTRAINTS = [
    "The product is a web application that must support multiple devices.",
    "Must work with SRD 3.5 rules and accept adventure modules written in SRD 3.5 standards.",
    "SRD 3.5 rules, audio assets, and voice libraries must be baked into the code rather than supplied by external services.",
    "The experience is turn-based; a short delay in syncing across devices is acceptable, so real-time sync is not a design constraint.",
    "Players build characters by hand; the app's role is validation, not generation.",
]


def test_real_session_rule_is_unchanged_three_flagged_two_kept():
    kept, dropped = dedup_constraints(_REAL_CONSTRAINTS, [s for _, s in _REAL_NEEDS])
    assert dropped == _REAL_CONSTRAINTS[:3] and kept == _REAL_CONSTRAINTS[3:]


def test_flag_text_names_the_matched_need_and_says_why():
    from interview_engine.distill_qa import explain_duplicate_constraint
    texts = [explain_duplicate_constraint(c, _REAL_NEEDS) for c in _REAL_CONSTRAINTS[:3]]
    assert texts[0].startswith('Same fact as need SN-5: "The web application should support multiple devices."')
    assert texts[1].startswith("Same fact as need SN-4:") and "also overlaps need SN-3" in texts[1]
    assert texts[2].startswith("Same fact as need SN-13:")
    for t in texts:
        assert "one place" in t and "drift apart" in t and "Keep one." in t
    assert "also overlaps" not in texts[0]


def test_flag_text_without_need_ids_quotes_the_need_only():
    from interview_engine.distill_qa import explain_duplicate_constraint
    text = explain_duplicate_constraint(_REAL_CONSTRAINTS[0], [("", s) for _, s in _REAL_NEEDS])
    assert text.startswith('Same fact as a need: "The web application should support multiple devices."')


def test_flag_text_falls_back_when_no_need_fires():
    from interview_engine.distill_qa import explain_duplicate_constraint
    assert explain_duplicate_constraint("$0 infrastructure budget", _REAL_NEEDS) == \
        "This design constraint repeats a classified stakeholder need."
