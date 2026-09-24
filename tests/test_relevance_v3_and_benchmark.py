"""relevance-v3-se (covers_* carried by the variant, option B), per-area covers
scoring, and the synthetic benchmark generator. Hermetic: no model calls."""
import csv, importlib.util, io, json, subprocess, sys
from pathlib import Path

import pytest

from interview_engine.categorization import COVERAGE_AREAS, coverage_request
from interview_engine.categorization_rubrics import (COVERS_V3, COVERS_V3_PLAIN, REL_V3_PREAMBLE, RELEVANCE_VARIANTS,
                                                     configured_variant, get_variant)

ROOT = Path(__file__).resolve().parents[1]
CLAUSE = "in response to the question asked"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_v1_keeps_legacy_covers_template_and_state():
    req = coverage_request("Q", "A", "users", "relevance-v1")
    assert req.state == "Question: Q\nAnswer: A" and req.rubric_version == "categorization-request-v1"
    assert req.questions["covers_goal"].instructions == "The answer supplies concrete product facts for coverage area: goal"


def test_v3_se_option_b_clause_only_on_target_area():
    req = coverage_request("Q", "A", "users", "relevance-v3-se")
    assert req.rubric_version == "relevance-v3-se" and req.state.startswith(REL_V3_PREAMBLE)
    assert req.state.endswith("Question: Q\nAnswer: A")
    assert set(k for k in req.questions if k.startswith("covers_")) == {f"covers_{a}" for a in COVERAGE_AREAS}
    assert req.questions["covers_users"].instructions == COVERS_V3["users"] and CLAUSE in COVERS_V3["users"]
    for a in COVERAGE_AREAS:
        if a != "users":
            assert req.questions[f"covers_{a}"].instructions == COVERS_V3_PLAIN[a] and CLAUSE not in COVERS_V3_PLAIN[a]
    assert "supplies concrete product facts" not in json.dumps({k: q.instructions for k, q in req.questions.items() if k.startswith("covers_")})


def test_v3_se_without_target_uses_plain_everywhere_and_option_a_uses_clause_everywhere():
    b = coverage_request("Q", "A", None, "relevance-v3-se")
    assert all(CLAUSE not in b.questions[f"covers_{a}"].instructions for a in COVERAGE_AREAS)
    a = coverage_request("Q", "A", "users", "relevance-v3-se-a")
    assert all(CLAUSE in a.questions[f"covers_{x}"].instructions for x in COVERAGE_AREAS)


def test_v3_relevance_criteria_match_the_approved_draft():
    q = RELEVANCE_VARIANTS["relevance-v3-se"].questions["relevance"]
    assert q.instructions == "Does the answer address the question asked?"
    assert "substantively" not in json.dumps(q.criteria) and "without delegating" not in json.dumps(q.criteria)
    assert "generic or pasted text" in q.criteria["dodged"]


def test_v3_se_is_env_switchable_and_v1_stays_default(monkeypatch):
    monkeypatch.delenv("INTERVIEW_ENGINE_RUBRIC_RELEVANCE", raising=False)
    assert configured_variant("relevance").id == "relevance-v1"
    monkeypatch.setenv("INTERVIEW_ENGINE_RUBRIC_RELEVANCE", "relevance-v3-se")
    assert configured_variant("relevance").id == "relevance-v3-se"
    assert coverage_request("Q", "A", "goal").rubric_version == "relevance-v3-se"


def test_replay_uses_variant_covers_and_stored_target():
    rp = _load("replay", "scripts/replay-jev-rubric-variants.py")
    stored = {"covers_goal": {"instructions": "old goal"}, "relevance": {"instructions": "x", "criteria": {"addressed": "a"}}}
    case = {"slot": "coverage", "state": "Question: Q\nAnswer: A", "context": {"target": "goal"}, "questions": stored}
    old = rp.build_request(case, get_variant("relevance-v2a-anchored"))
    assert old.questions["covers_goal"].instructions == "old goal" and old.state == "Question: Q\nAnswer: A"
    new = rp.build_request(case, get_variant("relevance-v3-se"))
    assert new.questions["covers_goal"].instructions == COVERS_V3["goal"] and new.state.startswith(REL_V3_PREAMBLE)
    assert new.questions["covers_users"].instructions == COVERS_V3_PLAIN["users"]


def test_covers_truth_formats():
    sc = _load("score", "scripts/score-rubric-variants.py")
    assert sc.covers_truth({"human_verdict": '{"relevance": "partial", "covers": ["users"]}'})["users"] is True
    t = sc.covers_truth({"human_verdict": '{"covers_goal": 0.8, "covers_data": 0}'})
    assert t["goal"] is True and t["data"] is False and t["users"] is False
    t = sc.covers_truth({"human_verdict": "", "winner": "B", "system_a_verdict": '{"covers_goal": 0.9}', "system_b_verdict": '{"covers_goal": 0.1}'})
    assert t == {"goal": False}
    assert sc.covers_truth({"human_verdict": "", "winner": "neither", "system_a_verdict": "{}", "system_b_verdict": "{}"}) is None


POOL = {"questions": [{"id": "Q-goal", "area": "goal", "text": "What problem?"}, {"id": "Q-users", "area": "users", "text": "Who uses it?"}],
        "responses": [{"cls": "answer", "id": "A1", "text": "Missed calls.", "intent": ["answer"], "areas": ["goal"], "full": True},
                      {"cls": "answer", "id": "P1", "text": "Clients, I guess.", "intent": ["answer"], "areas": ["users"], "full": False},
                      {"cls": "delegate", "id": "D1", "text": "Your call.", "intent": ["delegate"], "areas": []},
                      {"cls": "mix", "id": "M1", "text": "Missed calls; rest your call.", "intent": ["answer", "delegate"], "areas": ["goal"], "full": False}],
        "pairing_rules": {"exclusions": ["A1 x Q-users"]}}


def test_benchmark_rows_labels_and_determinism(tmp_path):
    gen = _load("gen", "scripts/build-intent-benchmark.py")
    rows = gen.build_rows(POOL)
    intent = [r for r in rows if r["slot"] == "intent"]; cov = [r for r in rows if r["slot"] == "coverage"]
    assert len(intent) == 7 and len(cov) == 5          # 4x2 minus 1 exclusion; control-only D1 gets no coverage rows
    hv = {r["review_notes"].split(" |")[0] + "/" + r["slot"]: json.loads(r["human_verdict"]) for r in rows}
    assert hv["A1 x Q-goal/coverage"]["relevance"] == "addressed" and hv["P1 x Q-users/coverage"]["relevance"] == "partial"
    assert hv["P1 x Q-goal/coverage"]["relevance"] == "dodged" and hv["M1 x Q-goal/coverage"]["relevance"] == "partial"
    assert hv["M1 x Q-users/intent"]["intent"] == ["answer", "delegate"] and "D1 x Q-goal/coverage" not in hv
    assert len({r["case_id"] for r in rows}) == len(rows) and len({(r["input_ref"], r["slot"]) for r in rows}) == len(rows)
    assert gen.to_csv(rows) == gen.to_csv(gen.build_rows(POOL))
    assert list(csv.DictReader(io.StringIO(gen.to_csv(rows))))[0].keys() == set(gen.FIELDS)
    with pytest.raises(ValueError, match="unknown intent"):
        gen.build_rows({**POOL, "responses": [{**POOL["responses"][0], "intent": ["maybe"]}]})


def test_shipped_pool_builds_and_flows_through_replay_and_scorer(tmp_path):
    out = tmp_path / "bench.csv"
    env = {"PYTHONPATH": ":".join(sys.path), "PATH": ""}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build-intent-benchmark.py"), "-o", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "rows: 779  intent: 560  coverage: 219" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "scripts/replay-jev-rubric-variants.py"), str(out), "-o", str(tmp_path / "o"),
                        "--intent-variants", "intent-v1", "--relevance-variants", "relevance-v1,relevance-v3-se", "--dry-run"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "coverage cases with a stored target area: 219/219" in r.stdout
    empty = tmp_path / "empty"; empty.mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/score-rubric-variants.py"), str(out), "-", str(empty)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "with usable ground truth: 560" in r.stdout and "Coverage rows with a usable covers truth: 219 of 219" in r.stdout


def test_scorer_reports_covers_per_area_for_replayed_variant(tmp_path):
    gen = _load("gen2", "scripts/build-intent-benchmark.py")
    rows = gen.build_rows(POOL)
    lab = tmp_path / "b.csv"; lab.write_text(gen.to_csv(rows))
    rd = tmp_path / "r"; rd.mkdir()
    with (rd / "replay-relevance-v3-se.jsonl").open("w") as f:
        for r in rows:
            if r["slot"] != "coverage": continue
            truth = json.loads(r["human_verdict"])
            verdict = {"relevance": truth["relevance"], **{f"covers_{a}": (1.0 if a in truth["covers"] else 0.0) for a in COVERAGE_AREAS}}
            f.write(json.dumps({"slot": "coverage", "replay": {"case_id": r["case_id"]},
                                "jev": {"status": "ok", "verdict": verdict, "confidence": {"relevance": .9}}}) + "\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/score-rubric-variants.py"), str(lab), "-", str(rd)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "| relevance-v3-se | 5 | 1.00 | 1.00 |" in r.stdout      # relevance table and covers table both perfect


def test_explain_benchmark_misses_groups_by_response(tmp_path):
    gen = _load("gen3", "scripts/build-intent-benchmark.py"); ex = _load("ex", "scripts/explain-benchmark-misses.py")
    rows = gen.build_rows(POOL); lab = tmp_path / "b.csv"; lab.write_text(gen.to_csv(rows))
    rd = tmp_path / "r"; rd.mkdir()
    with (rd / "replay-relevance-v3-se.jsonl").open("w") as f:
        for r in rows:
            if r["slot"] == "coverage":  # predict partial everywhere
                f.write(json.dumps({"slot": "coverage", "replay": {"case_id": r["case_id"]},
                                    "jev": {"status": "ok", "verdict": {"relevance": "partial"}, "confidence": {"relevance": .6}}}) + "\n")
    text = ex.report(str(lab), str(rd))
    assert "relevance-v3-se (coverage) - 3 misses" in text      # 1 addressed + 2 dodged mislabeled partial
    assert "| dodged |  | 2 |  |" in text and "- A1 (" in text and "Missed calls" in text


def test_bench2_variants_registered_replay_only():
    from interview_engine.categorization_rubrics import get_variant, COVERS_V3
    r = get_variant("relevance-v3.1-se"); i = get_variant("intent-v1.4")
    assert r.covers is COVERS_V3 and not r.covers_clause_all and not r.production_ok
    assert "answers at least one part" in r.questions["relevance"].criteria["partial"]
    assert not i.production_ok and "Stop the questions" in i.questions["intent"].criteria["enough"]
    # production defaults untouched
    assert get_variant("intent-v1").production_ok and get_variant("relevance-v3-se").production_ok


def test_pool_v2_final_shape():
    import json
    from pathlib import Path
    pool = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "intent-pool-v2.json").read_text())
    ids = [r["id"] for r in pool["responses"]]
    assert len(ids) == 70 == len(set(ids))
    parts = [r for r in pool["responses"] if r["cls"] == "answer" and "answer" in r["intent"] and not r.get("full", True) and len(r["areas"]) == 1 and not r["id"].startswith("MIX")]
    assert {r["areas"][0] for r in parts} >= {"goal", "data", "integrations", "failures", "risks", "done", "glossary"}
