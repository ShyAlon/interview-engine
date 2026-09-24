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
defer is compatible with neither.

v1.3 (2026-09-24): dismiss is folded into skip. By default every intent
label - his labels, the original columns and replayed predictions - is
normalized (dismiss -> skip) before scoring. --legacy-dismiss scores under
the v1.2 six-label taxonomy instead, so v1.2-era numbers can be reproduced.

covers_* (per-area coverage, added 2026-09-24): scored per area as a yes/no
call at >= 0.5. Ground truth per row: human_verdict covers_<area> values, or a
"covers" list/object in human_verdict, when present; otherwise the winner
side's covers_* (on a tie, only areas where both sides agree). Only coverage
rows with a usable covers truth are scored. Keys may be "-" for no key file.
For the per-label P/R columns, a correct prediction counts as that label; a
miss counts against the first label in his set.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Intent labels and compatibility come from the single source of truth.
from interview_engine.intent_taxonomy import COMPATIBLE_GROUPS, INTENT_ORDER, LEGACY_ALIASES, compatible, normalize  # noqa: E402

INTENT_LABELS = INTENT_ORDER
LEGACY = False  # set by --legacy-dismiss
LEGACY_INTENT_LABELS = ("answer", "delegate", "defer", "dismiss", "skip", "enough")


def _norm(label, field):
    return label if (LEGACY or field != "intent") else normalize(label)


def _parse(v):
    v = (v or "").strip()
    if not v: return None
    try: return json.loads(v)
    except json.JSONDecodeError: return v


COMPATIBLE = {"intent": COMPATIBLE_GROUPS}  # answer+delegate; defer compatible with neither


def _labels(value):
    if isinstance(value, list): return [str(v) for v in value if v not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def truth_set(row, field):
    """His label set for this row, or None when no usable label exists."""
    hv = _parse(row.get("human_verdict"))
    def out(ls):  # normalized, de-duplicated, order kept
        seen = []
        for l in (_norm(x, field) for x in ls):
            if l not in seen: seen.append(l)
        return seen
    if isinstance(hv, dict) and _labels(hv.get(field)): return out(_labels(hv[field]))
    if isinstance(hv, list) and field == "intent" and _labels(hv): return out(_labels(hv))
    t = truth(row, field)
    return out([t]) if t else None


def is_correct(pred, labels, field):
    if field == "intent": return compatible(pred, labels, legacy=LEGACY)
    return pred in labels


def effective_truth(pred, labels, field):
    return _norm(pred, field) if is_correct(pred, labels, field) else labels[0]


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


from interview_engine.coverage_taxonomy import AREA_KEYS  # noqa: E402  single source of truth
AREAS = tuple(AREA_KEYS)


def _yes(v):
    if isinstance(v, bool): return v
    try: return float(v) >= .5
    except (TypeError, ValueError): return None


def covers_truth(row):
    """area -> bool for this coverage row, or None when there is no usable truth."""
    hv = _parse(row.get("human_verdict"))
    if isinstance(hv, dict):
        c = hv.get("covers")
        if isinstance(c, list): return {a: a in c for a in AREAS}
        if isinstance(c, dict): return {a: bool(_yes(c.get(a, c.get(f"covers_{a}", 0)))) for a in AREAS}
        keys = {a: _yes(hv[f"covers_{a}"]) for a in AREAS if f"covers_{a}" in hv}
        if keys: return {a: bool(keys.get(a, False)) for a in AREAS}
    a, b = _parse(row.get("system_a_verdict")) or {}, _parse(row.get("system_b_verdict")) or {}
    side = {"a": a, "b": b}.get((row.get("winner") or "").strip().lower())
    if side is not None:
        vals = {x: _yes(side.get(f"covers_{x}")) for x in AREAS}
        return {x: v for x, v in vals.items() if v is not None} or None
    if (row.get("winner") or "").strip().lower() == "tie":
        out = {x: _yes(a.get(f"covers_{x}")) for x in AREAS if _yes(a.get(f"covers_{x}")) is not None and _yes(a.get(f"covers_{x}")) == _yes(b.get(f"covers_{x}"))}
        return out or None
    return None


def covers_table(systems_covers, truths):
    """One row per system: per-area F1 plus overall micro accuracy/F1 over every scored area call."""
    head = "| system | rows | micro acc | micro F1 | " + " | ".join(AREAS) + " |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for name, preds in systems_covers.items():
        tp = fp = fn = tn = 0; per = []
        for area in AREAS:
            atp = afp = afn = 0
            for cid, pred in preds.items():
                t = truths[cid].get(area); p = _yes(pred.get(f"covers_{area}"))
                if t is None or p is None: continue
                atp += p and t; afp += p and not t; afn += (not p) and t; tn += (not p) and not t
            tp += atp; fp += afp; fn += afn
            per.append(f"{2 * atp / (2 * atp + afp + afn):.2f}" if atp + afp + afn else "-")
        n = tp + fp + fn + tn
        acc = f"{(tp + tn) / n:.2f}" if n else "-"; f1 = f"{2 * tp / (2 * tp + fp + fn):.2f}" if tp + fp + fn else "-"
        lines.append(f"| {name} | {len(preds)} | {acc} | {f1} | " + " | ".join(per) + " |")
    return lines


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
    ap.add_argument("-o", "--output")
    ap.add_argument("--legacy-dismiss", action="store_true", help="score intent under v1.2 (dismiss separate from skip)")
    a = ap.parse_args(argv)
    global LEGACY; LEGACY = a.legacy_dismiss
    rows = list(csv.DictReader(open(a.labeled, encoding="utf-8-sig", newline="")))
    key = {} if a.key == "-" else {k["case_id"]: k for k in json.load(open(a.key))}
    out = []; P = out.append
    for slot, field, labels in (("intent", "intent", LEGACY_INTENT_LABELS if LEGACY else INTENT_LABELS), ("coverage", "relevance", ("addressed", "partial", "dodged"))):
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
            pairs = [(_norm(p, field), effective_truth(p, scored[c], field)) for c, (p, _) in preds.items()]
            acc, macro, per = metrics(pairs, labels)
            right = [cf for c, (p, cf) in preds.items() if is_correct(p, scored[c], field) and isinstance(cf, (int, float))]
            wrong = [cf for c, (p, cf) in preds.items() if not is_correct(p, scored[c], field) and isinstance(cf, (int, float))]
            cr = f"{sum(right)/len(right):.2f}" if right else "-"; cw = f"{sum(wrong)/len(wrong):.2f}" if wrong else "-"
            cells = " | ".join(f"{per[l][0]:.2f}/{per[l][1]:.2f}" if per[l][1] is not None else f"{per[l][0]:.2f}/-" for l in labels)
            P(f"| {name} | {len(pairs)} | {acc:.2f} | {macro:.2f} | {cells} | {cr} / {cw} |")
        P("")
        if slot == "coverage":
            ct = {r["case_id"]: covers_truth(r) for r in srows}
            ct = {c: t for c, t in ct.items() if t}
            P("## coverage / covers_* per area (F1 per area; call = value >= 0.5)\n")
            P(f"Coverage rows with a usable covers truth: {len(ct)} of {len(srows)}\n")
            sc = {}
            for r in srows:
                if r["case_id"] not in ct or r["case_id"] not in key: continue
                for side in ("system_a", "system_b"):
                    name = "baseline (original)" if key[r["case_id"]][side] == "baseline" else "jev-v1 (original)"
                    sc.setdefault(name, {})[r["case_id"]] = _parse(r[side + "_verdict"]) or {}
            for f in sorted(Path(a.replay_dir).glob("replay-*.jsonl")):
                for rec in (json.loads(l) for l in f.open() if l.strip()):
                    cid = rec.get("replay", {}).get("case_id")
                    if rec.get("slot") == "coverage" and cid in ct and rec["jev"].get("status") == "ok":
                        sc.setdefault(f.stem.replace("replay-", ""), {})[cid] = rec["jev"]["verdict"]
            for line in covers_table(sc, ct): P(line)
            P("")
    text = "\n".join(out); print(text)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
