#!/usr/bin/env python3
"""Explain where replayed variants miss on the synthetic benchmark. No model calls.

  python scripts/explain-benchmark-misses.py bench.csv runs/bench-1 [-o misses.md]

For every replay file: a confusion matrix per slot (truth rows x predicted
columns), then the misses grouped by response id (the pool row that caused
them) and by question id, with example texts. Intent labels are normalized
(dismiss -> skip) like the scorer; multi-label truth counts a prediction as
right when it is compatible with the set.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from interview_engine.intent_taxonomy import INTENT_ORDER, compatible, normalize  # noqa: E402

REL = ("addressed", "partial", "dodged")


def load(bench):
    rows = {}
    for r in csv.DictReader(open(bench, encoding="utf-8-sig", newline="")):
        pair = r["review_notes"].split(" |")[0]
        rid, _, qid = pair.partition(" x ")
        hv = json.loads(r["human_verdict"])
        truth = [normalize(x) for x in hv["intent"]] if r["slot"] == "intent" else [hv["relevance"]]
        answer = r["input_state"].rsplit("Answer: ", 1)[-1] if "Answer: " in r["input_state"] else ""
        rows[r["case_id"]] = {"slot": r["slot"], "rid": rid, "qid": qid, "truth": truth, "answer": answer}
    return rows


def right(slot, pred, truth):
    return compatible(pred, truth) if slot == "intent" else pred in truth


def report(bench, replay_dir):
    rows = load(bench); out = []; P = out.append
    for f in sorted(Path(replay_dir).glob("replay-*.jsonl")):
        recs = [json.loads(l) for l in f.open() if l.strip()]
        name = f.stem.replace("replay-", "")
        slot = recs[0]["slot"] if recs else "?"
        field = "intent" if slot == "intent" else "relevance"
        labels = INTENT_ORDER if slot == "intent" else REL
        conf = Counter(); misses = []
        for rec in recs:
            row = rows.get(rec["replay"]["case_id"])
            if not row or rec["jev"].get("status") != "ok": continue
            pred = normalize(str(rec["jev"]["verdict"].get(field)))
            t = "+".join(row["truth"]); conf[(t, pred)] += 1
            if not right(slot, pred, row["truth"]):
                misses.append((row, pred, rec["jev"].get("confidence", {}).get(field)))
        P(f"## {name} ({slot}) - {len(misses)} misses\n")
        truths = sorted({t for t, _ in conf}, key=lambda t: (t.split("+")[0] not in labels, t))
        P("| truth \\ pred | " + " | ".join(labels) + " |"); P("|" + "---|" * (len(labels) + 1))
        for t in truths: P(f"| {t} | " + " | ".join(str(conf[(t, l)] or "") for l in labels) + " |")
        by_r = defaultdict(list); by_q = Counter()
        for row, pred, c in misses: by_r[row["rid"]].append((row, pred, c)); by_q[row["qid"]] += 1
        P("\nMisses by response (response: count, predicted labels, questions):\n")
        for rid, ms in sorted(by_r.items(), key=lambda kv: -len(kv[1])):
            preds = Counter(p for _, p, _ in ms); qs = ",".join(sorted(r["qid"].replace("Q-", "") for r, _, _ in ms))
            P(f"- {rid} ({'+'.join(ms[0][0]['truth'])}): {len(ms)} -> {dict(preds)} on {qs}  | \"{ms[0][0]['answer'][:90]}\"")
        P("\nMisses by question: " + ", ".join(f"{q}={n}" for q, n in by_q.most_common()) + "\n")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench"); ap.add_argument("replay_dir"); ap.add_argument("-o", "--output"); a = ap.parse_args(argv)
    text = report(a.bench, a.replay_dir); print(text)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
