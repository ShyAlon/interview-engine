#!/usr/bin/env python3
"""Score a component-tree replay against the pool-v4 bench (pre-registered gates).

  .venv/bin/python scripts/score-component-bench.py bench-v4.csv REPLAY_DIR [-o report.md]

Reads the bench CSV (build-intent-benchmark.py on a component pool) and every
replay-*.jsonl in REPLAY_DIR whose records carry component answers (comp__*).
Gates (Shy 2026-09-24) are scored on the 14 area questions; follow-up rows,
founder-skip rows and the "mentioned" signal are reported without a bar.
Call = value >= 0.5. Exit code 0 always; PASS/FAIL is printed per gate.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from interview_engine.component_trees import parse  # noqa: E402

GATES = [  # (id, text, bar)
    ("dodged_recall", "Derived dodged recall", .90),
    ("wrong_question", "Concrete answers to another question fill no asked component", .90),  # bench-2 bar 117/130
    ("false_yes", "False yes per component (overall, lower is better)", .05),
    ("component_recall", "Component recall (given and caught)", .90),
    ("partials", "Truth partials derived partial", .83),
    ("addressed", "Truth addressed derived addressed", .77),
]


def load(bench, replay_dir):
    rows = {r["case_id"]: r for r in csv.DictReader(open(bench, encoding="utf-8-sig", newline="")) if r["slot"] == "coverage"}
    runs = {}
    for f in sorted(Path(replay_dir).glob("replay-*.jsonl")):
        recs = [json.loads(l) for l in f.open() if l.strip()]
        recs = [r for r in recs if r.get("slot") == "coverage" and r["jev"].get("status") == "ok"
                and any(k.startswith("comp__") for k in r["jev"]["verdict"])]
        if recs: runs[f.stem.replace("replay-", "")] = recs
    return rows, runs


def score(rows, recs):
    s = Counter(); fy = Counter(); fn = Counter(); models = Counter(); extra = Counter()
    for rec in recs:
        cid = rec["replay"]["case_id"]
        if cid not in rows: continue
        row = rows[cid]; hv = json.loads(row["human_verdict"]); ctx = json.loads(row["input_context"])
        v = rec["jev"]["verdict"]; models[f"{rec['jev'].get('provider')}/{rec['jev'].get('model')}"] += 1
        filled, skipped, mentioned = parse(v)
        pred = str(v.get("relevance")); truth = hv["relevance"]
        if ctx.get("open"):  # follow-up question: no bar
            extra["followup_n"] += 1; extra["followup_ok"] += pred == truth; continue
        area = ctx["target"]; comps = hv["components"]
        for c, t in comps.items():
            p = c in filled
            if t: s["comp_pos"] += 1; s["comp_tp"] += p; fn[c] += (not p)
            else: s["comp_neg"] += 1; s["comp_fp"] += p; fy[c] += p
        if truth == "dodged": s["dodged_n"] += 1; s["dodged_ok"] += pred == "dodged"
        if truth == "partial": s["partial_n"] += 1; s["partial_ok"] += pred == "partial"
        if truth == "addressed": s["addr_n"] += 1; s["addr_ok"] += pred == "addressed"
        if hv.get("skipped"): extra["skip_n"] += 1; extra["skip_ok"] += set(hv["skipped"]) <= skipped
        for a in mentioned: extra["mentioned_yes"] += 1; extra["mentioned_tp"] += a in hv.get("mentioned", [])
    return s, fy, fn, models, extra


def wrong_question(rows, recs, concrete_ids):
    n = ok = 0
    for rec in recs:
        row = rows.get(rec["replay"]["case_id"])
        if not row: continue
        hv = json.loads(row["human_verdict"]); ctx = json.loads(row["input_context"])
        if ctx.get("open"): continue
        resp = row["review_notes"].split(" x ")[0]
        if resp in concrete_ids and ctx["target"] not in hv.get("mentioned", []):
            n += 1; filled, _, _ = parse(rec["jev"]["verdict"])
            ok += not any(c.startswith(ctx["target"] + ".") for c in filled)
    return ok, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench"); ap.add_argument("replay_dir"); ap.add_argument("-o", "--output")
    ap.add_argument("--pool", default=str(ROOT / "benchmarks" / "intent-pool-v4.json"))
    a = ap.parse_args(argv)
    rows, runs = load(a.bench, a.replay_dir)
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    # concrete answer = a response that fully answers (derived addressed) at least one of its own areas
    from interview_engine.component_trees import derive_relevance
    concrete = {r["id"] for r in pool["responses"] for ar in r.get("areas", [])
                if derive_relevance(ar, set(r.get("components", {}).get(ar, [])), r.get("skipped", []),
                                    set(r.get("delegated", {}).get(ar, [])), personas=pool.get("personas")) == "addressed"}
    out = []; P = out.append
    if not runs: P("no component replay files found"); print("\n".join(out)); return 0
    for name, recs in runs.items():
        s, fy, fn, models, extra = score(rows, recs)
        P(f"# {name}")
        P(f"model: {', '.join(f'{m} ({n})' for m, n in models.items())}")
        wq_ok, wq_n = wrong_question(rows, recs, concrete)
        vals = {"dodged_recall": (s["dodged_ok"], s["dodged_n"]), "wrong_question": (wq_ok, wq_n),
                "false_yes": (s["comp_fp"], s["comp_neg"]), "component_recall": (s["comp_tp"], s["comp_pos"]),
                "partials": (s["partial_ok"], s["partial_n"]), "addressed": (s["addr_ok"], s["addr_n"])}
        for i, (gid, text, bar) in enumerate(GATES, 1):
            k, n = vals[gid]; r = k / n if n else float("nan")
            passed = (r <= bar) if gid == "false_yes" else (r >= bar)
            need = f"<= {math.floor(bar * n)}/{n}" if gid == "false_yes" else f">= {math.ceil(bar * n)}/{n}"
            P(f"{i}. {text}: {k}/{n} = {r:.3f}  (bar {need})  {'PASS' if passed else 'FAIL'}")
        P(f"7. Follow-up rows (no bar): {extra['followup_ok']}/{extra['followup_n']} derived label = truth")
        P(f"8. Founder-skip rows (no bar): skip caught on {extra['skip_ok']}/{extra['skip_n']}")
        P(f"9. mentioned (no bar): precision {extra['mentioned_tp']}/{extra['mentioned_yes']}")
        worst = [f"{c} {n}" for c, n in fy.most_common(8) if n]
        if worst: P(f"10. Most false yes: {', '.join(worst)}")
        miss = [f"{c} {n}" for c, n in fn.most_common(8) if n]
        if miss: P(f"11. Most missed: {', '.join(miss)}")
        P("")
    text = "\n".join(out); print(text)
    if a.output: Path(a.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
