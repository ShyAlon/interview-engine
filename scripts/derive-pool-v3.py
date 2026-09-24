"""Derive benchmarks/intent-pool-v3.json from pool-v2 plus the nfrs-split delta.

Taxonomy v2 (bench-3): Q-nfrs becomes four questions, 8 prototypes are added,
nfrs rows are re-mapped. Kept as a script so pool-v3 can be rebuilt after
pool-v2 is finalized (Shy's ENF-05..16 and ANS-P4..P10 edits). All texts and
rulings below are from Shy's pool-v3 markup, 2026-09-24.

    python scripts/derive-pool-v3.py [--v2 benchmarks/intent-pool-v2.json] [--out benchmarks/intent-pool-v3.json]
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    {"id": "Q-performance", "area": "performance", "text": "How fast and sleek does the app need to feel, and in which moments does waiting hurt most?"},
    {"id": "Q-reliability", "area": "reliability", "text": "When is it OK for the app to be unavailable, and for how long?"},
    {"id": "Q-security", "area": "security", "text": "Who should be able to see or change what, and is any of the data sensitive?"},
    {"id": "Q-scalability", "area": "scalability", "text": "How many clients and bookings do you expect the app to handle in a week?"},
]


def _ans(i, area, text, full, note):
    return {"cls": "answer", "id": i, "text": text, "intent": ["answer"], "areas": [area], "full": full, "note": note}


PROTOTYPES = [
    _ans("ANS-PERF", "performance", "Booking on a phone should take under a minute start to finish, and no screen should take more than a second or two. The groomers' tablet has to show the next dog instantly - they tap it with wet hands between dogs, so no spinners.", True, "Full: who, which moments, numbers."),
    _ans("ANS-PERF-P", "performance", "It needs to be fast. Nobody likes waiting.", False, "Partial: no moments, no numbers. Drill case."),
    _ans("ANS-REL", "reliability", "Saturday 8 to 2 is our busiest window - it can't be down then at all. A few minutes at night for updates is fine, even every week, and half an hour on a quiet Tuesday we can live with. A confirmed booking must never disappear.", True, "Full: windows, durations, what must never be lost."),
    _ans("ANS-REL-P", "reliability", "It should be up most of the time, especially on weekends.", False, "Partial: direction, no windows or durations. Drill case."),
    _ans("ANS-SEC", "security", "Clients see only their own dogs and bookings. Groomers see the day's schedule and the dog notes, bite warnings and vet info included, but not payment details. Only Dana and I can change prices or give refunds. We don't want card numbers stored with us at all - the payment company keeps them.", True, "Full: who sees/changes what, sensitive data. No standard needed."),
    _ans("ANS-SEC-P", "security", "Keep clients' details private, obviously.", False, "Partial: privacy matters, but not who sees what or which data."),
    _ans("ANS-SCALE", "scalability", "About 400 regular clients. 25 to 35 bookings on a normal day, up to 50 on a Saturday before the holidays. Two groomers and Dana at the desk.", True, "Full: expected volume for now, with numbers."),
    _ans("ANS-SCALE-P", "scalability", "Not huge - we're a small salon.", False, "Partial (Shy: 'requires a followup'). Drill case."),
]

# id -> fields to replace
REMAP = {
    "ANS-06": {"areas": ["performance", "reliability"], "full": True, "partial_areas": ["performance"],
               "note": "pool-v3: full for reliability (windows), partial for performance (no numbers). 'Big buttons' was usability - shelved, closes nothing."},
    "ANS-P3": {"areas": [], "full": False,
               "note": "pool-v3: Shy 'No, that's not an answer' - vacuous, no area covered; dodged against every question."},
}


def derive(v2: dict) -> dict:
    pool = copy.deepcopy(v2)
    qs = []
    for q in pool["questions"]:
        qs.extend(QUESTIONS if q["area"] == "nfrs" else [q])
    pool["questions"] = qs
    for r in pool["responses"]:
        if r["id"] in REMAP: r.update(REMAP[r["id"]])
        elif "nfrs" in r.get("areas", []):
            raise ValueError(f"{r['id']} still covers nfrs - add it to REMAP")
    ids = {r["id"] for r in pool["responses"]}
    pool["responses"].extend(p for p in PROTOTYPES if p["id"] not in ids)
    pool["coverage_taxonomy"] = "coverage-v2"
    pool["status"] = "pool-v3 DRAFT (derived from pool-v2 + nfrs-split delta). Queued behind bench-2."
    pool["history"] = list(pool.get("history", [])) + ["2026-09-24: v3 - nfrs split into performance, reliability, security, scalability (usability shelved). 8 prototypes, ANS-06/ANS-P3 re-mapped; Shy's pool-v3 markup."]
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2", default=str(ROOT / "benchmarks" / "intent-pool-v2.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "intent-pool-v3.json"))
    a = ap.parse_args(argv)
    pool = derive(json.loads(Path(a.v2).read_text()))
    Path(a.out).write_text(json.dumps(pool, indent=2, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(pool['responses'])} responses x {len(pool['questions'])} questions")


if __name__ == "__main__":
    main()
