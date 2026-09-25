#!/usr/bin/env python3
"""Remove reviewer-marked summary Q/A from a JEV-L2 labeled export.

Mark each summary question with EXCLUDE_SUMMARY in review_notes while labeling.
The original 195-case corpus and browser storage remain untouched. The output
retains the same adjudication columns/case IDs so replay and scoring use it.
--dry-run counts only; it never writes files.
"""
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path

TOKEN='EXCLUDE_SUMMARY'


def filter_rows(rows):
    kept=[];excluded=[]
    for r in rows:
        if TOKEN in (r.get('review_notes') or '').upper():excluded.append(r)
        else:kept.append(r)
    return kept,excluded


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
    kept,excluded=filter_rows(rows)
    labeled=sum(bool((r.get('human_verdict') or '').strip()) for r in kept)
    print(f'1. Original {len(rows)} cases; excluded {len(excluded)} with exact note token {TOKEN}.')
    print(f'2. Retained {len(kept)} cases; {labeled} with human verdict; {len(kept)-labeled} still unlabelled.')
    print(f'3. Paired v1.3/v1.4 replay: {2*len(kept)} Jev calls. Scorer should report {labeled} usable ground-truth rows if all verdicts parse.')
    if not kept:print('4. No cases remain; output not written.');return 2
    if a.dry_run:print('4. Dry run: no files written.');return 0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
    print(f'4. Wrote filtered CSV: {a.output}. Original export unchanged.')
    return 0

if __name__=='__main__':sys.exit(main())
