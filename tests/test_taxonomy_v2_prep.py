"""Taxonomy v2 (nfrs split) is prepared but not wired: v1 stays the default."""
import importlib.util, json
from pathlib import Path

from interview_engine import categorization, coverage
from interview_engine.categorization_rubrics import COVERS_V3_TAX2, RELEVANCE_VARIANTS
from interview_engine.coverage_taxonomy import (AREA_KEYS, AREA_KEYS_V2, DRILL_PROBES_V2, FALLBACK_QUESTIONS_V2,
                                                MAX_DRILLS_V2, NUMERIC_AREAS_V2)

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ["performance", "reliability", "security", "scalability"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_v2_shape():
    i = AREA_KEYS.index("nfrs")
    assert AREA_KEYS_V2 == AREA_KEYS[:i] + QUALITY + AREA_KEYS[i + 1:] and len(AREA_KEYS_V2) == 14
    assert "usability" not in AREA_KEYS_V2 and MAX_DRILLS_V2 == 8
    assert set(NUMERIC_AREAS_V2) == {"performance", "reliability", "scalability"} == set(DRILL_PROBES_V2)
    assert set(FALLBACK_QUESTIONS_V2) == set(QUALITY)
    assert list(COVERS_V3_TAX2) == AREA_KEYS_V2


def test_v1_still_default():
    assert coverage.AREA_KEYS == AREA_KEYS and "nfrs" in categorization.COVERAGE_AREAS
    assert not any(v.covers is COVERS_V3_TAX2 for v in RELEVANCE_VARIANTS.values())


def test_pool_v3_derivation():
    pool = _load("derive-pool-v3").derive(json.loads((ROOT / "benchmarks" / "intent-pool-v2.json").read_text()))
    assert [q["area"] for q in pool["questions"]] == AREA_KEYS_V2
    assert len(pool["responses"]) == 78  # pool-v2 final (70) + 8 prototypes
    assert not any("nfrs" in r.get("areas", []) for r in pool["responses"])
    rel = _load("build-intent-benchmark").relevance
    by = {r["id"]: r for r in pool["responses"]}
    assert rel(by["ANS-06"], "reliability") == "addressed" and rel(by["ANS-06"], "performance") == "partial"
    assert rel(by["ANS-P3"], "performance") == "dodged"
    assert rel(by["ANS-SCALE-P"], "scalability") == "partial" and rel(by["ANS-SCALE"], "scalability") == "addressed"
