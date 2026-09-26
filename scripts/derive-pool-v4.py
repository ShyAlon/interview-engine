#!/usr/bin/env python3
"""Derive pool-v4 (component truth) from pool-v3.

  python scripts/derive-pool-v4.py [benchmarks/intent-pool-v3.json] -o benchmarks/intent-pool-v4.json

pool-v4 = pool-v3 responses, each labeled with the components it fills in its
own area(s) (component_trees, TREES_VERSION), plus four founder-skip rows
(NA-01..04) and ten follow-up questions. Labels: agent draft 2026-09-24, Shy's
markup applied (tree becomes truth; risks and constraints all-optional;
flows.exceptions, data.source, integrations.pricing, security.signin,
security.sensitive optional; plain-word "fast"/"small" replies are partial
with one clarification probe (2026-09-26); keep all NA rows).
Pure function of the input pool.
"""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from interview_engine.component_trees import DEFAULT_PERSONAS, TREES_VERSION, component_ids, followup  # noqa: E402

D = "front_desk"
# response -> area -> filled components (short ids; persona.key for per-persona)
FILLED = {
    "ANS-01": {"goal": ["problem", "who", "cost"]},
    "ANS-02": {"users": ["personas", "client.context", "groomer.context", f"{D}.context", "client.device", "groomer.device", "groomer.count"]},
    "ANS-03": {"flows": ["trigger", "actors", "steps", "outcome", "exceptions"]},
    "ANS-04": {"data": ["what", "source"]},
    "ANS-05": {"integrations": ["purpose"]},
    "ANS-06": {"performance": ["client.moment", "groomer.moment"], "reliability": ["allowed", "forbidden"]},
    "ANS-07": {"failures": ["worst", "never", "when_down"]},
    "ANS-08": {"constraints": ["budget", "deadline", "maintainer", "technology"]},
    "ANS-09": {"risks": ["assumption", "notice"]},
    "ANS-10": {"done": ["measure", "target", "by_when"]},
    "ANS-11": {"glossary": ["terms", "meaning"]},
    "ANS-P1": {"flows": ["trigger", "actors", "steps"]},
    "ANS-P2": {"users": ["personas"]},
    "ANS-P4": {"goal": ["problem"]},
    "ANS-P5": {"data": ["what"]},
    "ANS-P6": {"integrations": ["purpose"]},
    "ANS-P7": {"failures": ["worst", "never"]},
    "ANS-P8": {"risks": ["assumption"]},
    "ANS-P9": {"done": ["measure"]},
    "ANS-P10": {"glossary": ["terms"]},
    "ANS-C1": {"goal": ["problem"]},  # excluded from Q-goal anyway
    "ANS-R1": {"flows": ["actors", "steps"]},
    "MIX-01": {"flows": ["trigger", "actors", "steps"]},
    "MIX-02": {"integrations": ["purpose"]},
    "MIX-03": {"constraints": ["budget"]},
    "ANS-PERF": {"performance": ["client.moment", "client.number", "groomer.moment"]},
    "ANS-REL": {"reliability": ["allowed", "forbidden", "never_lost"]},
    "ANS-REL-P": {"reliability": ["forbidden"]},
    "ANS-SEC": {"security": ["client.see", "groomer.see", f"{D}.change", "sensitive"]},
    "ANS-SEC-P": {"security": ["sensitive"]},
    "ANS-SCALE": {"scalability": ["users_n", "volume", "peak"]},
    # ANS-P3, ANS-C2, ANS-PERF-P, ANS-SCALE-P: fill nothing (plain-word replies score partial
    # with one clarification probe, Shy 2026-09-26; ANS-P3/C2 are off-topic and stay dodged)
}
DELEGATED = {"MIX-02": {"integrations": ["systems"]}}
NA_ROWS = [
    {"id": "NA-01", "text": "Speed doesn't matter for Dana at the desk - she's never rushed with it.", "area": "performance", "skipped": [f"performance.{D}"], "filled": []},
    {"id": "NA-02", "text": "The groomers won't change anything in it - they only look.", "area": "security", "skipped": [], "filled": ["groomer.change"]},
    {"id": "NA-03", "text": "Scale isn't a thing for us - it's one salon and it will stay one salon.", "area": "scalability", "skipped": ["scalability"], "filled": []},
    {"id": "NA-04", "text": "We don't want it connected to anything - it runs on its own.", "area": "integrations", "skipped": ["integrations"], "filled": [],
     "note": "Contradicts ANS-05 (the salon uses payments and WhatsApp); kept to test an area-level skip (Shy: keep)."},
]
FOLLOWUP_TARGETS = ["goal.who", "users.client.context", "flows.outcome", "data.what", "integrations.systems",
                    "performance.client.number", "reliability.per_year", "security.groomer.see", "scalability.volume", "done.target"]


def _full(area, short):
    return f"{area}.{short}"


def derive(pool3):
    pool = copy.deepcopy(pool3)
    pool["status"] = "pool-v4 FINAL: component truth (agent draft + Shy's markup 2026-09-24; derivation rule and 12 recomputed cells approved 2026-09-26)."
    pool["component_trees"] = TREES_VERSION
    pool["personas"] = list(DEFAULT_PERSONAS)
    pool["history"] = pool.get("history", []) + ["2026-09-24: v4 - component truth per response (trees), NA-01..04 founder-skip rows, ten follow-up questions."]
    ids = {r["id"] for r in pool["responses"]}
    assert set(FILLED) <= ids and set(DELEGATED) <= ids
    for r in pool["responses"]:
        comps = {a: sorted(_full(a, s) for s in v) for a, v in FILLED.get(r["id"], {}).items()}
        for a, lst in comps.items():
            unknown = set(lst) - set(component_ids(a))
            assert not unknown, (r["id"], unknown)
        r["components"] = comps
        r["delegated"] = {a: sorted(_full(a, s) for s in v) for a, v in DELEGATED.get(r["id"], {}).items()}
        r["skipped"] = []
    for n in NA_ROWS:
        a = n["area"]
        pool["responses"].append({"cls": "answer", "id": n["id"], "text": n["text"], "intent": ["answer"], "areas": [a], "full": True,
                                  "components": {a: sorted(_full(a, s) for s in n["filled"])} if n["filled"] else {},
                                  "delegated": {}, "skipped": n["skipped"], "note": n.get("note", "Founder-skip test row (pool-v4).")})
    for cid in FOLLOWUP_TARGETS:
        assert cid in component_ids(cid.split(".")[0], required_only=True), cid
        pool["questions"].append({"id": "F-" + cid, "area": cid.split(".")[0], "targets": cid, "text": followup(cid)})
    pool["pairing_rules"]["followups"] = "follow-up questions pair with answer rows only; relevance truth = the targeted component filled (addressed) or not (dodged)"
    pool["pairing_rules"]["relevance"] = "derived from component truth by component_trees.derive_relevance"
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(ROOT / "benchmarks" / "intent-pool-v3.json"))
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    pool = derive(json.loads(Path(a.pool).read_text(encoding="utf-8")))
    Path(a.output).write_text(json.dumps(pool, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pool-v4: {len(pool['responses'])} responses, {len(pool['questions'])} questions -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
