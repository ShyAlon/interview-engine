#!/usr/bin/env python3
"""Build a standalone A/B labeling tool (HTML) from an adjudication CSV.

  .venv/bin/python scripts/build-labeling-tool.py INPUT.csv -o TOOL.html
      [--key KEY.json] [--key-out KEY_SUBSET.json] [--only-unlabeled]
      [--slot intent|coverage|scope] [--title "..."] [--subtitle "..."]

The intent labels, their order, their criteria text and intent multi-select
all come from src/interview_engine/intent_taxonomy.py (embedded as JSON at
build time), so a rebuilt tool can never disagree with the engine. Intent
criteria in each row's rubric JSON are replaced by the taxonomy when the tool
renders; the exported CSV keeps the input's 17 columns and values unchanged
except winner / human_verdict / review_notes.

The localStorage key is derived from the embedded CSV content, so tools built
from different inputs never share progress, and a rebuild from the same input
resumes it. Existing winner / human_verdict / review_notes in the CSV preload
as labels (use --clear-labels for a blind round).

  --only-unlabeled   keep only rows with no winner and no human_verdict, or
                     winner=neither with an empty human_verdict (the rule a
                     case needs before it counts as labeled)
  --clear-labels     blank winner / human_verdict / review_notes for a blind round
  --key / --key-out  write the key entries for exactly the embedded cases; the
                     key is never embedded, so the tool stays blind
"""
from __future__ import annotations
import argparse, base64, csv, hashlib, io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from interview_engine.intent_taxonomy import TAXONOMY_VERSION, taxonomy_json  # noqa: E402

TEMPLATE = ROOT / "scripts" / "templates" / "labeling-tool.html"
COLUMNS = ["case_id", "input_ref", "slot", "session_id", "input_state", "input_context", "rubric", "rubric_version",
           "system_a_verdict", "system_a_confidence", "system_b_verdict", "system_b_confidence", "sample_reason",
           "human_verdict", "winner", "reviewer_id", "review_notes"]
LABEL_COLS = ("winner", "human_verdict", "review_notes")


def unlabeled(row) -> bool:
    w = (row.get("winner") or "").strip().lower(); hv = (row.get("human_verdict") or "").strip()
    return (not w and not hv) or (w == "neither" and not hv)


def build(rows, *, title, subtitle, export_name) -> tuple[str, str]:
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\r\n", extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    data = buf.getvalue().encode()
    digest = hashlib.sha256(data + TAXONOMY_VERSION.encode()).hexdigest()[:12]
    store = f"labeling-tool-v2:{digest}"
    tax = taxonomy_json().replace("</", "<\\/")
    html = TEMPLATE.read_text()
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for k, v in (("__CSV_B64__", base64.b64encode(data).decode()), ("__TAXONOMY_JSON__", tax), ("__STORE_KEY__", store),
                 ("__TITLE__", esc(title)), ("__SUBTITLE__", esc(subtitle)), ("__EXPORT_NAME__", export_name.replace("'", "")),
                 ("__TAXONOMY_VERSION__", esc(TAXONOMY_VERSION))):
        assert html.count(k) == 1, k
        html = html.replace(k, v)
    return html, store


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input"); p.add_argument("-o", "--output", required=True)
    p.add_argument("--key"); p.add_argument("--key-out"); p.add_argument("--slot")
    p.add_argument("--only-unlabeled", action="store_true"); p.add_argument("--clear-labels", action="store_true")
    p.add_argument("--title", default="A/B Labeling"); p.add_argument("--subtitle")
    a = p.parse_args(argv)
    rows = list(csv.DictReader(open(a.input, encoding="utf-8-sig", newline="")))
    missing = [c for c in COLUMNS if rows and c not in rows[0]]
    if missing: raise SystemExit(f"{a.input}: missing columns {missing}")
    if a.slot: rows = [r for r in rows if r["slot"] == a.slot]
    if a.only_unlabeled: rows = [r for r in rows if unlabeled(r)]
    if a.clear_labels: rows = [{**r, **{c: "" for c in LABEL_COLS}} for r in rows]
    if not rows: raise SystemExit("no rows selected")
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True); stem = out.stem
    html, store = build(rows, title=a.title, subtitle=a.subtitle or stem, export_name=f"{stem}-labeled.csv")
    out.write_text(html)
    if a.key:
        key = {k["case_id"]: k for k in json.load(open(a.key))}
        sub = [key[r["case_id"]] for r in rows if r["case_id"] in key]
        missing = [r["case_id"] for r in rows if r["case_id"] not in key]
        kout = Path(a.key_out) if a.key_out else out.with_name(stem + "-key.json")
        kout.write_text(json.dumps(sub, indent=2) + "\n")
        print(f"key: {kout} ({len(sub)} entries{', missing ' + str(missing) if missing else ''})")
    print(f"tool: {out} ({len(rows)} cases, storage key {store}, intent labels {TAXONOMY_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
