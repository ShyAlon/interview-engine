#!/usr/bin/env python3
"""Make a synthetic off-topic probe from real JEV-L2 questions; no Jev calls.

Pairs 12 distinct existing questions with deliberately unrelated cat facts.
This tests the proposed non-answer behavior, not correctness under the frozen
five-label intent taxonomy (where off-topic product text remains `answer`).
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

CASES={
 '708e80db09a22198':'My cat sleeps in a cardboard box every afternoon.',
 '625225c79963e2fa':'Our cat has orange fur and likes to sit by the window.',
 '08e8bf90ad46f670':'The cat prefers its blue blanket to the green one.',
 'a90ef9d67e072b60':'My cat eats twice a day and drinks from a ceramic bowl.',
 '008f4b767adf0a3a':'The neighbor has a long-haired cat named Miso.',
 'bd2b9866e0d21ce6':'Cats often nap in warm patches of sunlight.',
 '5e72bea87d05883d':'Our cat likes chasing toy mice across the floor.',
 '2126bc04d7912341':'My cat has a white spot on one ear.',
 'c4a6e49e9343fa6d':'The cat waits by the door before breakfast.',
 '6fc6ef8c785d964a':'Our cat purrs loudly when someone brushes its coat.',
 'b915b8b3b7867530':'My cat prefers cardboard scratching pads.',
 '9d4f93ff94f77f05':'A black cat visits the garden every evening.',
}
def build(source):
    with open(source,encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f);fields=rd.fieldnames;rows={r['case_id']:r for r in rd}
    missing=set(CASES)-set(rows)
    if missing:raise ValueError('missing source IDs: '+', '.join(sorted(missing)))
    out=[]
    for cid,cat in CASES.items():
        base=rows[cid];question=base['input_state'].split('Founder answer:',1)[0].strip()
        if not question.startswith('Question:'):raise ValueError('unrecognized question: '+cid)
        state=question+'\nFounder answer: '+cat
        row={k:base.get(k,'') for k in fields}
        row.update(case_id=hashlib.sha256((cid+'|cat-probe-v1|'+state).encode()).hexdigest()[:16],
                   input_ref='sha256:'+hashlib.sha256(state.encode()).hexdigest(),
                   input_state=state,session_id='synthetic-cat-probe',sample_reason='synthetic-cat-probe',
                   human_verdict='',winner='',reviewer_id='',review_notes='')
        out.append(row)
    return fields,out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('filtered_csv',type=Path);p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.filtered_csv.resolve()==a.output.resolve():p.error('output must differ from source')
    fields,rows=build(a.filtered_csv)
    print(f'{len(rows)} synthetic off-topic cat pairs from {len(CASES)} real questions; {2*len(rows)} paired Jev calls; no ground-truth intent assigned.')
    if not a.dry_run:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        print(f'Wrote {a.output}')
if __name__=='__main__':main()
