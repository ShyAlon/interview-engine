"""coverage_taxonomy is the single source of the coverage-area list (Step 0 of
the nfrs split): every consumer reads it, and the refactor changed no text."""
import importlib.util
from pathlib import Path

from interview_engine import categorization, coverage, prompts
from interview_engine.coverage_taxonomy import AREA_KEYS, AREAS, AREAS_V1, LABELS, TAXONOMY_VERSION, V1_TAXONOMY_VERSION

ROOT = Path(__file__).resolve().parents[1]
V1 = ["goal", "users", "flows", "data", "integrations", "nfrs", "failures", "constraints", "risks", "done", "glossary"]


def test_v1_list_and_order_unchanged():
    assert V1_TAXONOMY_VERSION == "coverage-v1"
    assert AREA_KEYS == V1 and [k for k, _, _ in AREAS] == V1
    assert LABELS["nfrs"] == "quality needs"


def test_every_consumer_reads_the_same_list():
    assert coverage.AREAS is AREAS and coverage.AREA_KEYS is AREA_KEYS and coverage._LABELS is LABELS
    assert categorization.COVERAGE_AREAS == tuple(V1)
    spec = importlib.util.spec_from_file_location("score", ROOT / "scripts" / "score-rubric-variants.py")
    score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
    assert score.AREAS == tuple(V1)


def test_structured_turn_prompt_is_byte_identical():
    assert prompts.STRUCTURED_TURN_FORMAT == (
        'Reply ONLY with JSON (no prose/fence):\n'
        '{"question":"one short concrete question","covered":[],"thin":[],"skipped":[],"complete":false}\n'
        'Arrays use only: goal, users, flows, data, integrations, nfrs, failures, constraints, risks, done, glossary.\n'
        'covered=concrete; thin=mentioned/vague; skipped=refused. If complete=true, question is a one-line closing summary.')


def test_no_other_hardcoded_area_list():
    needle = '"goal", "users", "flows"'
    hits = [p for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
            if needle in p.read_text()]
    assert hits == [], hits


def test_v1_1_changes_only_the_integrations_closing_text():
    assert TAXONOMY_VERSION == "coverage-v1.1"
    changed = [k for (k, _, a), (_, _, b) in zip(AREAS_V1, AREAS) if a != b]
    assert changed == ["integrations"]
    assert dict((k, t) for k, _, t in AREAS_V1)["integrations"] == "named systems/APIs, not categories"
    assert dict((k, t) for k, _, t in AREAS)["integrations"].endswith("or 'none'")
