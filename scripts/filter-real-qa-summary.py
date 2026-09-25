#!/usr/bin/env python3
"""Remove reviewer-marked summary Q/A from a JEV-L2 labeled export.

Mark each summary question with EXCLUDE_SUMMARY in review_notes while labeling.
Mark a clarification request with CLARIFY in its note; these are preserved in
an audit CSV but excluded from the current five-label v1.3/v1.4 score. Neither
note marker is itself an intent label or ground truth.
The original 195-case corpus and browser storage remain untouched. The output
retains the same adjudication columns/case IDs so replay and scoring use it.
--dry-run counts only; it never writes files.
"""
from __future__ import annotations
import argparse,csv,re,sys
from pathlib import Path

TOKEN='EXCLUDE_SUMMARY'
CLARIFY='CLARIFY'
# Exact free-text notes from the 195-row JEV-L2 export. Do not use a broad
# "exclude"/"clarification" substring: that would silently drop other cases.
SUMMARY_NOTES={
    'exclude from the corpus', 'another blob to exclude',
    'a blob to exclude from the corpus', 'blob to exclude', 'exclude',
}
CLARIFY_NOTES={
    "it's actually asking for clarification. maybe we need an entirely new case for that.",
    'another clarification question',
}


def filter_rows(rows):
    kept=[];excluded=[];clarifications=[]
    for r in rows:
        note=(r.get('review_notes') or '').strip()
        if re.search(r"\bEXCLUDE_SUMMARY\b", note) or note.casefold() in SUMMARY_NOTES:excluded.append(r)
        elif re.search(r"\bCLARIFY\b", note) or note.casefold() in CLARIFY_NOTES:clarifications.append(r)
        else:kept.append(r)
    return kept,excluded,clarifications


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('export',type=Path,help='browser-exported labeled adjudication CSV')
    p.add_argument('-o','--output',type=Path,required=True,help='new, filtered CSV path; never overwrite export')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.export.resolve()==a.output.resolve():p.error('output must differ from original labeled export')
    with a.export.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not fields or not {'case_id','review_notes','human_verdict','slot'}<=set(fields):p.error('not an adjudication CSV')
    ids=[r['case_id'] for r in rows]
    if len(ids)!=len(set(ids)):p.error('duplicate case IDs: cannot score this export safely')
    kept,excluded,clarifications=filter_rows(rows)
    labeled=sum(bool((r.get('human_verdict') or '').strip()) for r in kept)
    print(f'1. Original {len(rows)} cases; excluded {len(excluded)} with {TOKEN} or known exclusion notes; set aside {len(clarifications)} with {CLARIFY} or known clarification notes.')
    print(f'2. Retained {len(kept)} cases; {labeled} with human verdict; {len(kept)-labeled} still unlabelled.')
    print(f'3. Paired v1.3/v1.4 replay: {2*len(kept)} Jev calls. Scorer should report {labeled} usable ground-truth rows if all verdicts parse.')
    if not kept:print('4. No cases remain; output not written.');return 2
    if a.dry_run:print('4. Dry run: no files written.');return 0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
    if clarifications:
        audit=a.output.with_name(a.output.stem+'-clarifications.csv')
        with audit.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(clarifications)
        print(f'4. Wrote filtered CSV: {a.output}; clarification audit: {audit}. Original export unchanged.')
    else:print(f'4. Wrote filtered CSV: {a.output}. Original export unchanged.')
    return 0

if __name__=='__main__':sys.exit(main())
