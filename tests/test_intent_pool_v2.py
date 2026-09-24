"""intent-pool-v2: v1 plus the enough expansion (Shy 2026-09-24: at least 16)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pool(name):
    return json.loads((ROOT / "benchmarks" / name).read_text())


def test_v2_has_at_least_16_enough_and_keeps_v1_rows():
    v1, v2 = _pool("intent-pool-v1.json"), _pool("intent-pool-v2.json")
    enough = [r for r in v2["responses"] if r["intent"] == ["enough"]]
    assert len(enough) >= 16
    v2_by_id = {r["id"]: r for r in v2["responses"]}
    assert all(v2_by_id[r["id"]] == r for r in v1["responses"])      # v1 rows unchanged
    assert len({r["text"].casefold() for r in v2["responses"]}) == len(v2["responses"])
    assert v2["pair_count"] == len(v2["responses"]) * len(v2["questions"]) - len(v2["pairing_rules"]["exclusions"])


def test_enough_rows_do_not_copy_rubric_examples():
    from interview_engine.intent_taxonomy import INTENT_EXAMPLES
    text = " ".join(INTENT_EXAMPLES.values()).casefold()
    for r in _pool("intent-pool-v2.json")["responses"]:
        assert r["text"].casefold().rstrip(".") not in text


def test_pool_texts_not_copied_into_v1_4_examples():
    from interview_engine.categorization_rubrics import get_variant
    text = " ".join(get_variant("intent-v1.4").questions["intent"].criteria.values()).casefold()
    for r in _pool("intent-pool-v2.json")["responses"]:
        assert r["text"].casefold().rstrip(".") not in text, r["id"]
