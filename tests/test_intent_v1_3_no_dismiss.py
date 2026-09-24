"""v1.3 (2026-09-24): dismiss removed, skip covers both moving on and declining
the topic. Old labels normalize to skip; v1.2 stays replayable and scorable."""
import csv, json, subprocess, sys
from pathlib import Path

from interview_engine.categorization import ClassificationAnswer, classified_intent
from interview_engine.intent_taxonomy import (INTENT_ORDER, LEGACY_ALIASES, TERMINAL_INTENTS, compatible, normalize,
                                              taxonomy_json)

ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_has_no_dismiss_and_maps_it_to_skip():
    assert INTENT_ORDER == ("answer", "delegate", "defer", "skip", "enough")
    assert "dismiss" not in TERMINAL_INTENTS and normalize("dismiss") == "skip" and normalize("answer") == "answer"
    assert json.loads(taxonomy_json())["legacy_aliases"] == LEGACY_ALIASES == {"dismiss": "skip"}
    assert compatible("dismiss", ["skip"]) and not compatible("dismiss", ["skip"], legacy=True)


def test_engine_treats_a_legacy_dismiss_verdict_as_skip():
    assert classified_intent({"intent": ClassificationAnswer("dismiss", .95)}) == "skip"
    assert classified_intent({"intent": ClassificationAnswer("dismiss", .5)}) == "answer"   # still gated


def test_scorer_normalizes_by_default_and_legacy_flag_keeps_v1_2(tmp_path):
    rows = [{"case_id": "c1", "input_ref": "r1", "slot": "intent", "session_id": "s", "input_state": "Q", "input_context": "{}",
             "rubric": "{}", "rubric_version": "v", "system_a_verdict": '{"intent": "dismiss"}', "system_a_confidence": "{}",
             "system_b_verdict": '{"intent": "answer"}', "system_b_confidence": "{}", "sample_reason": "x",
             "human_verdict": '{"intent": ["skip"]}', "winner": "neither", "reviewer_id": "", "review_notes": ""}]
    lab = tmp_path / "l.csv"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    key = tmp_path / "k.json"; key.write_text(json.dumps([{"case_id": "c1", "system_a": "baseline", "system_b": "jev"}]))
    rd = tmp_path / "r"; rd.mkdir()
    run = lambda *extra: subprocess.run([sys.executable, str(ROOT / "scripts/score-rubric-variants.py"), str(lab), str(key), str(rd), *extra],
                                        capture_output=True, text=True)
    new = run(); old = run("--legacy-dismiss")
    assert new.returncode == 0 and old.returncode == 0, new.stderr + old.stderr
    assert "| baseline (original) | 1 | 1.00" in new.stdout and "dismiss P/R" not in new.stdout
    assert "| baseline (original) | 1 | 0.00" in old.stdout and "dismiss P/R" in old.stdout


def test_benchmark_pool_has_no_dismiss_labels():
    pool = json.loads((ROOT / "benchmarks/intent-pool-v1.json").read_text())
    assert pool["taxonomy"] == "categorization-request-v1.3"
    assert not any("dismiss" in r["intent"] for r in pool["responses"])
    assert sum(r["intent"] == ["skip"] for r in pool["responses"]) == 15
