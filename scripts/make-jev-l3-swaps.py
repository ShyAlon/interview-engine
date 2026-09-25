#!/usr/bin/env python3
"""Replay the six JEV-L3 wrong-area product-answer swaps from the audit CSV."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

def build(source,audit):
    with open(source,encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f));rows={r['case_id']:r for r in raw};fields=list(raw[0])
    if len(rows)!=len(raw):raise ValueError('duplicate source IDs')
    with open(audit,encoding='utf-8-sig',newline='') as f:items=[r for r in csv.DictReader(f) if r['group']=='synthetic-wrong-area']
    if len(items)!=6 or len({r['case_id'] for r in items})!=6:raise ValueError('expected six unique swaps')
    out=[]
    for s in items:
        source_id=s['source_case_id'];base=rows[source_id]
        q=base['input_state'].split('Founder answer:',1)[0].strip()
        if q!='Question: '+s['question']:raise ValueError('source question mismatch: '+source_id)
        state=q+'\nFounder answer: '+s['founder_answer']
        if len(s['case_id'])!=16:raise ValueError('bad swap case ID')
        row=dict(base);row.update(case_id=s['case_id'],input_ref='sha256:'+hashlib.sha256(state.encode()).hexdigest(),
                                  input_state=state,session_id='synthetic-jev-l3-swap',sample_reason='synthetic-jev-l3-swap',
                                  human_verdict='',winner='',reviewer_id='',review_notes='')
        out.append(row)
    return fields,out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('filtered_csv',type=Path);p.add_argument('audit_csv',type=Path)
    p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args()
    if a.output.resolve() in (a.filtered_csv.resolve(),a.audit_csv.resolve()):p.error('output must differ from inputs')
    fields,rows=build(a.filtered_csv,a.audit_csv)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} reviewed wrong-area swaps to {a.output}; six v1.6 replay calls.')
if __name__=='__main__':main()
