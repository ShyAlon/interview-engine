#!/usr/bin/env python3
"""Build the synthetic intent + relevance benchmark from the response pool.

  python scripts/build-intent-benchmark.py [benchmarks/intent-pool-v1.json] -o bench.csv

Every response is crossed with every question (minus the pool's exclusions).
Ground truth is known by construction, so each row's human_verdict is filled:

- intent row per pair: human_verdict {"intent": [the response's label set]}.
- coverage row per pair, only when the response contains an answer (pure
  control moves - delegate/defer/dismiss/skip/enough alone - get no relevance
  score): relevance = addressed if the question's area is one the response
  covers fully, partial if it covers it partly, dodged otherwise; plus
  "covers": the response's areas (per-area covers_* truth).

Output is the labeled 17-column adjudication CSV the replay and scorer already
read. system_a/system_b columns are empty and winner is "label", so score it
with key "-":

  python scripts/replay-jev-rubric-variants.py bench.csv -o OUT --intent-variants ... --relevance-variants ...
  python scripts/score-rubric-variants.py bench.csv - OUT

Component pools (pool-v4+, "component_trees" set): coverage truth comes from
the per-response component labels - relevance derived by
component_trees.derive_relevance, plus "components" (every component of the
asked area -> filled yes/no), "skipped" and "mentioned". The coverage context
carries personas and, for follow-up questions ("targets"), the one open
component. Follow-up questions get coverage rows only (intent truth does not
depend on the question).

Pure function of the pool file: same pool in, byte-identical CSV out.
"""
from __future__ import annotations
import argparse, csv, dataclasses, hashlib, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from interview_engine.categorization import coverage_request, intent_request  # noqa: E402
from interview_engine.categorization_shadow import input_package  # noqa: E402
from interview_engine.intent_taxonomy import INTENT_ORDER  # noqa: E402

FIELDS = ["case_id", "input_ref", "slot", "session_id", "input_state", "input_context", "rubric", "rubric_version",
          "system_a_verdict", "system_a_confidence", "system_b_verdict", "system_b_confidence", "sample_reason",
          "human_verdict", "winner", "reviewer_id", "review_notes"]
CONTROL = set(INTENT_ORDER) - {"answer"}


def relevance(resp, area):
    if area not in resp.get("areas", []): return "dodged"
    if area in resp.get("partial_areas", []): return "partial"  # per-area override (pool-v3: ANS-06)
    return "addressed" if resp.get("full", True) else "partial"


def component_truth(pool, resp, q):
    from interview_engine.component_trees import component_ids, derive_relevance
    area, personas = q["area"], pool.get("personas")
    filled = set(resp.get("components", {}).get(area, []))
    deleg = set(resp.get("delegated", {}).get(area, []))
    skipped = [n for n in resp.get("skipped", []) if n == area or n.startswith(area + ".")]
    open_ = [q["targets"]] if q.get("targets") else None
    rel = derive_relevance(area, filled, skipped, deleg, open_components=open_, personas=personas)
    return ({"relevance": rel, "components": {c: c in filled for c in component_ids(area, personas)},
             "delegated": sorted(deleg), "skipped": skipped, "mentioned": list(resp.get("areas", []))},
            {"target": area, "personas": personas, **({"open": open_} if open_ else {})})


def build_rows(pool, intent_variant="intent-v1", relevance_variant="relevance-v1"):
    for r in pool["responses"]:
        bad = [l for l in r["intent"] if l not in INTENT_ORDER]
        if bad: raise ValueError(f"{r['id']}: unknown intent label(s) {bad}")
    excluded = {tuple(x.split(" x ")) for x in pool.get("pairing_rules", {}).get("exclusions", [])}
    session = "bench-" + hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()[:12]
    rows = []
    for resp in pool["responses"]:
        for q in pool["questions"]:
            if (resp["id"], q["id"]) in excluded: continue
            pair = f"{resp['id']} x {q['id']}"
            notes = f"{pair} | class {resp['cls']}" + (f" | {resp['note']}" if resp.get("note") else "")
            trees = bool(pool.get("component_trees"))
            slots = [] if q.get("targets") else [("intent", intent_request(q["text"], resp["text"], intent_variant), {"intent": resp["intent"]})]
            if "answer" in resp["intent"]:
                req = coverage_request(q["text"], resp["text"], q["area"], relevance_variant)
                if trees:
                    truth, ctx = component_truth(pool, resp, q)
                    slots.append(("coverage", dataclasses.replace(req, context=ctx), truth))
                else:
                    slots.append(("coverage", req, {"relevance": relevance(resp, q["area"]), "covers": list(resp.get("areas", []))}))
            for slot, req, truth in slots:
                pkg = input_package(req)
                ref = "sha256:" + hashlib.sha256(f"{slot}|{pkg['state']}|{json.dumps(pkg['context'], sort_keys=True)}".encode()).hexdigest()
                rows.append({"case_id": hashlib.sha256(f"{pair}|{slot}".encode()).hexdigest()[:16], "input_ref": ref, "slot": slot,
                             "session_id": session, "input_state": pkg["state"],
                             "input_context": json.dumps(pkg["context"], sort_keys=True),
                             "rubric": json.dumps(pkg["questions"], sort_keys=True), "rubric_version": pkg["rubric_version"],
                             "system_a_verdict": "", "system_a_confidence": "", "system_b_verdict": "", "system_b_confidence": "",
                             "sample_reason": "synthetic-benchmark", "human_verdict": json.dumps(truth, sort_keys=True),
                             "winner": "label", "reviewer_id": "by-construction", "review_notes": notes})
    return rows


def to_csv(rows) -> str:
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(ROOT / "benchmarks" / "intent-pool-v1.json"))
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    rows = build_rows(pool)
    Path(a.output).write_text(to_csv(rows), encoding="utf-8")
    n_int = sum(r["slot"] == "intent" for r in rows); cov = [r for r in rows if r["slot"] == "coverage"]
    from collections import Counter
    rel = Counter(json.loads(r["human_verdict"])["relevance"] for r in cov)
    print(f"pool: {len(pool['responses'])} responses x {len(pool['questions'])} questions")
    print(f"rows: {len(rows)}  intent: {n_int}  coverage: {len(cov)} {dict(rel)}  -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
