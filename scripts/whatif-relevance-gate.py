#!/usr/bin/env python3
"""What-if: re-score an existing relevance replay with a code-owned dodged gate.

No model calls. Reads a benchmark CSV, its pool, and a replay folder, and for
each relevance variant prints the pre-registered bench-2 gate measures twice:
as run, and with a gate applied - "if the gate signal is below THRESHOLD, the
label is forced to dodged". The gate signal is a field already present in the
replay verdict:

  --signal covers_target   (default) covers_<the asked area>, which every
                           v3-se / v3.1-se coverage call already returns
  --signal FIELD           any other verdict field, e.g. rel_answers_this for a
                           bench-3 run of the structural gate

  python scripts/whatif-relevance-gate.py bench2.csv runs/bench-2 [--pool benchmarks/intent-pool-v2.json]
      [--signal covers_target] [--thresholds 0.3,0.5,0.7] [-o whatif.md]
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_truth(bench, pool):
    resp = {r["id"]: r for r in pool["responses"]}
    qarea = {q["id"]: q["area"] for q in pool["questions"]}
    rows = {}
    for r in csv.DictReader(open(bench, encoding="utf-8-sig", newline="")):
        if r["slot"] != "coverage": continue
        rid, _, qid = r["review_notes"].split(" |")[0].partition(" x ")
        rows[r["case_id"]] = {"rid": rid, "area": qarea[qid], "truth": json.loads(r["human_verdict"])["relevance"],
                              "resp": resp[rid]}
    return rows


def kind(row):
    """Bench-2 gate buckets for a dodged-truth row."""
    r = row["resp"]
    if r["id"].startswith("ANS-C"): return "canned"
    return "concrete" if r.get("full", True) else "vague"


def measures(pred, rows):
    c = Counter(); n = Counter()
    for cid, row in rows.items():
        if cid not in pred: continue
        t, p = row["truth"], pred[cid]
        n[t] += 1; c[(t, p)] += 1
        if t == "dodged":
            k = kind(row); n[k] += 1; c[(k, p == "dodged")] += 1
    pp = sum(v for (t, p), v in c.items() if p == "partial" and t in ("addressed", "partial", "dodged"))
    return {
        "1 concrete wrong-q -> dodged": f"{c[('concrete', True)]}/{n['concrete']}",
        "2 vague wrong-q -> dodged": f"{c[('vague', True)]}/{n['vague']}",
        "3 dodged recall": f"{c[('dodged', 'dodged')] / max(1, n['dodged']):.2f}",
        "4 addressed kept": f"{c[('addressed', 'addressed')]}/{n['addressed']}",
        "5 partials kept": f"{c[('partial', 'partial')]}/{n['partial']}",
        "5b partials pushed to dodged": f"{c[('partial', 'dodged')]}/{n['partial']}",
        "6 canned -> dodged": f"{c[('canned', True)]}/{n['canned']}",
        "partial precision": f"{c[('partial', 'partial')] / max(1, pp):.2f}",
    }


def gated(verdicts, rows, signal, th):
    out = {}
    for cid, v in verdicts.items():
        field = f"covers_{rows[cid]['area']}" if signal == "covers_target" else signal
        s = v.get(field)
        try: low = s is not None and float(s) < th
        except (TypeError, ValueError): low = False
        out[cid] = "dodged" if low else str(v.get("relevance"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench"); ap.add_argument("replay_dir")
    ap.add_argument("--pool", default=str(ROOT / "benchmarks" / "intent-pool-v2.json"))
    ap.add_argument("--signal", default="covers_target")
    ap.add_argument("--thresholds", default="0.3,0.5,0.7")
    ap.add_argument("-o", "--output")
    a = ap.parse_args(argv)
    rows = load_truth(a.bench, json.loads(Path(a.pool).read_text()))
    ths = [float(x) for x in a.thresholds.split(",")]
    out = [f"# What-if: dodged gate on {a.signal} (no model calls)\n"]
    for f in sorted(Path(a.replay_dir).glob("replay-*.jsonl")):
        verdicts = {}
        for rec in (json.loads(l) for l in f.open() if l.strip()):
            cid = rec.get("replay", {}).get("case_id")
            if rec.get("slot") == "coverage" and cid in rows and rec["jev"].get("status") == "ok":
                verdicts[cid] = rec["jev"]["verdict"]
        if not verdicts: continue
        cols = [("as run", {c: str(v.get("relevance")) for c, v in verdicts.items()})]
        cols += [(f"gate<{t}", gated(verdicts, rows, a.signal, t)) for t in ths]
        table = [(name, measures(p, rows)) for name, p in cols]
        keys = list(table[0][1])
        out.append(f"## {f.stem.replace('replay-', '')} ({len(verdicts)} rows)\n")
        out.append("| measure | " + " | ".join(n for n, _ in table) + " |")
        out.append("|---|" + "---|" * len(table))
        for k in keys: out.append(f"| {k} | " + " | ".join(m[k] for _, m in table) + " |")
        out.append("")
    text = "\n".join(out); print(text)
    if a.output: Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
