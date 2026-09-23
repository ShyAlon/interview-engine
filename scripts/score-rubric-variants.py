#!/usr/bin/env python3
"""Score replayed Jev rubric variants against Shy's labeled adjudication CSV.

  .venv/bin/python scripts/score-rubric-variants.py LABELED.csv KEY.json REPLAY_DIR [-o report.md]

Ground truth per row: human_verdict when filled (JSON object or bare label);
otherwise the winner side's verdict (A/B); on a tie, the value both sides
share. 'neither' rows with no human_verdict have no usable label and are
counted, not scored. The original baseline and Jev-v1 columns (unblinded with
the key) are scored on exactly the same rows for a like-for-like table.

Multi-label intent (locked 2026-09-23): human_verdict intent may be a single
string or a JSON array. A prediction is correct if it is in the label set, and
answer/delegate are compatible (delegate is an autonomy grant, not a
non-answer): predicting either one is correct when the set holds either one.
For the per-label P/R columns, a correct prediction counts as that label; a
miss counts against the first label in his set.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

INTENT_LABELS = ("answer", "delegate", "dismiss", "skip", "enough")


def _parse(v):
    v = (v or "").strip()
    if not v: return None
    try: return json.loads(v)
    except json.JSONDecodeError: return v


COMPATIBLE = {"intent": frozenset({"answer", "delegate"})}


def _labels(value):
    if isinstance(value, list): return [str(v) for v in value if v not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def truth_set(row, field):
    """His label set for this row, or None when no usable label exists."""
    hv = _parse(row.get("human_verdict"))
    if isinstance(hv, dict) and _labels(hv.get(field)): return _labels(hv[field])
    if isinstance(hv, list) and field == "intent" and _labels(hv): return _labels(hv)
    t = truth(row, field)
    return [t] if t else None


def is_correct(pred, labels, field):
    if pred in labels: return True
    group = COMPATIBLE.get(field, frozenset())
    return pred in group and any(l in group for l in labels)


def effective_truth(pred, labels, field):
    return pred if is_correct(pred, labels, field) else labels[0]


def truth(row, field):
    hv = _parse(row.get("human_verdict"))
    if isinstance(hv, dict) and isinstance(hv.get(field), list):
        ls = _labels(hv[field]); return ls[0] if ls else None
    if isinstance(hv, dict) and hv.get(field) not in (None, ""): return str(hv[field])
    if isinstance(hv, str) and field == "intent": return hv
    a, b = _parse(row["system_a_verdict"]) or {}, _parse(row["system_b_verdict"]) or {}
    w = (row.get("winner") or "").strip().lower()
    if w == "a": return str(a.get(field)) if a.get(field) is not None else None
    if w == "b": return str(b.get(field)) if b.get(field) is not None else None
    if w == "tie" and a.get(field) is not None and a.get(field) == b.get(field): return str(a[field])
    return None


def metrics(pairs, labels):
    n = len(pairs); acc = sum(p == t for p, t in pairs) / n if n else 0
    per = {}
    for l in labels:
        tp = sum(p == l and t == l for p, t in pairs); fp = sum(p == l and t != l for p, t in pairs)
        fn = sum(p != l and t == l for p, t in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else None
        f1 = 2 * prec * rec / (prec + rec) if rec and prec + rec else 0.0
        per[l] = (prec, rec, f1, tp + fn)
    present = [l for l in labels if per[l][3]]
    macro = sum(per[l][2] for l in present) / len(present) if present else 0
    return acc, macro, per


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("labeled"); ap.add_argument("key"); ap.add_argument("replay_dir")
    ap.add_argument("-o", "--output"); a = ap.parse_args(argv)
    rows = list(csv.DictReader(open(a.labeled, encoding="utf-8-sig", newline="")))
    key = {k["case_id"]: k for k in json.load(open(a.key))}
    out = []; P = out.append
    for slot, field, labels in (("intent", "intent", INTENT_LABELS), ("coverage", "relevance", ("addressed", "partial", "dodged"))):
        srows = [r for r in rows if r["slot"] == slot]
        gt = {r["case_id"]: truth_set(r, field) for r in srows}
        scored = {c: t for c, t in gt.items() if t}
        P(f"## {slot} / {field}\n")
        P(f"Labeled rows: {len(srows)}; with usable ground truth: {len(scored)}; excluded (neither/unresolved tie): {len(srows) - len(scored)}")
        P(f"Ground-truth distribution: {dict(Counter('+'.join(v) for v in scored.values()))}\n")
        systems = {}
        for r in srows:
            if r["case_id"] not in scored or r["case_id"] not in key: continue
            k = key[r["case_id"]]
            for side, col in (("system_a", "system_a"), ("system_b", "system_b")):
                v = _parse(r[col + "_verdict"]) or {}; c = _parse(r[col + "_confidence"]) or {}
                name = "baseline (original)" if k[side] == "baseline" else "jev-v1 (original)"
                systems.setdefault(name, {})[r["case_id"]] = (str(v.get(field)), c.get(field))
        for f in sorted(Path(a.replay_dir).glob("replay-*.jsonl")):
            recs = [json.loads(l) for l in f.open() if l.strip()]
            recs = [r for r in recs if r.get("slot") == slot]
            if not recs: continue
            name = f.stem.replace("replay-", "")
            for r in recs:
                cid = r["replay"]["case_id"]
                if cid in scored and r["jev"].get("status") == "ok":
                    systems.setdefault(name, {})[cid] = (str(r["jev"]["verdict"].get(field)), r["jev"]["confidence"].get(field))
        head = "| system | n | accuracy | macro-F1 | " + " | ".join(f"{l} P/R" for l in labels) + " | conf right / wrong |"
        P(head); P("|" + "---|" * (head.count("|") - 1))
        for name, preds in systems.items():
            pairs = [(p, effective_truth(p, scored[c], field)) for c, (p, _) in preds.items()]
            acc, macro, per = metrics(pairs, labels)
            right = [cf for c, (p, cf) in preds.items() if is_correct(p, scored[c], field) and isinstance(cf, (int, float))]
            wrong = [cf for c, (p, cf) in preds.items() if not is_correct(p, scored[c], field) and isinstance(cf, (int, float))]
            cr = f"{sum(right)/len(right):.2f}" if right else "-"; cw = f"{sum(wrong)/len(wrong):.2f}" if wrong else "-"
            cells = " | ".join(f"{per[l][0]:.2f}/{per[l][1]:.2f}" if per[l][1] is not None else f"{per[l][0]:.2f}/-" for l in labels)
            P(f"| {name} | {len(pairs)} | {acc:.2f} | {macro:.2f} | {cells} | {cr} / {cw} |")
        P("")
    text = "\n".join(out); print(text)
    if a.output: Path(a.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
