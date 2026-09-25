#!/usr/bin/env python3
"""Offline native v1.6 score: revised real truth, cats, and reviewed swaps.

Shy's JEV-L3 addressing verdict supersedes stale JEV-L2 intent `answer` or
`non-answer` only at matching real IDs. Control intent remains the separately
reviewed JEV-L2 mapping. L3 boundary/unsure cases are excluded, not guessed.
Synthetic wrong-area cases are separate, never added to the real denominator.
"""
from __future__ import annotations
import argparse,csv,importlib.util,json
from collections import Counter
from pathlib import Path

spec=importlib.util.spec_from_file_location('v15',Path(__file__).with_name('score-jev-l2-v15.py'))
v15=importlib.util.module_from_spec(spec);spec.loader.exec_module(v15)
LABELS=v15.LABELS

def read_replay(path,variant):
    out={}
    with open(path,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);cid=r['replay']['case_id']
            if cid in out or r['replay']['variant_id']!=variant or r['jev']['status']!='ok':raise ValueError('duplicate, wrong variant or failed replay: '+cid)
            verdict=r['jev']['verdict'];label=verdict['intent']
            if label not in LABELS:raise ValueError('unknown label: '+cid)
            if variant=='intent-v1.6':
                source=verdict.get('control'),verdict.get('addressing')
                if source[0] not in ('none','proceed','skip','enough') or source[1] not in ('addressed','not-addressed'):
                    raise ValueError('missing v1.6 source judgments: '+cid)
                expected=source[0] if source[0]!='none' else ('answer' if source[1]=='addressed' else 'non-answer')
                if label!=expected:raise ValueError('final label contradicts source judgments: '+cid)
            out[cid]=label
    return out

def review_rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=34 or len({r['case_id'] for r in rows})!=34:raise ValueError('expected 34 distinct JEV-L3 reviews')
    if any(r['verdict'] not in ('answer','non-answer','boundary') for r in rows):raise ValueError('missing/unknown JEV-L3 verdict')
    return rows

def validate_audit(review,audit):
    with open(audit,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    a={r['case_id']:r for r in rows}
    if len(a)!=34 or {r['case_id'] for r in review}!=set(a):raise ValueError('review/audit ID mismatch')
    expected=Counter({'real-disputed':20,'real-control':8,'synthetic-wrong-area':6})
    if Counter(r['group'] for r in rows)!=expected:raise ValueError('audit group mismatch')
    for r in review:
        base=a[r['case_id']]
        if r['question']!=base['question'] or r['founder_answer']!=base['founder_answer']:
            raise ValueError('review/audit Q/A mismatch: '+r['case_id'])
    return a

def truth_sets(real_csv,cat_csv,review_csv,audit_csv):
    review=review_rows(review_csv);audit=validate_audit(review,audit_csv)
    real=v15.real_truth(real_csv)
    skipped=[];changed=[];synthetic={}
    for r in review:
        cid=r['case_id'];label=r['verdict'];grp=audit[cid]['group']
        if grp=='synthetic-wrong-area':
            if label!='boundary':synthetic[cid]=[label]
            else:skipped.append(cid)
            continue
        if cid not in real:raise ValueError('reviewed real ID missing from source: '+cid)
        if label=='boundary':
            del real[cid];skipped.append(cid);continue
        # Addressing review does not override an explicit control move: this
        # is exactly the approved combination rule. Preserve it in the truth.
        old=real[cid]
        if old==['answer'] or old==['non-answer']:
            real[cid]=[label]
        if old!=real[cid]:changed.append((cid,old,real[cid]))
    with open(cat_csv,encoding='utf-8-sig',newline='') as f:cats=list(csv.DictReader(f))
    if len(cats)!=12 or len({r['case_id'] for r in cats})!=12 or any(r['sample_reason']!='synthetic-cat-probe' for r in cats):
        raise ValueError('not the 12-case cat probe')
    return real,{r['case_id']:['non-answer'] for r in cats},synthetic,skipped,changed

def analyze(real_csv,real_replay,cat_csv,cat_replay,review_csv,audit_csv,swap_csv,swap_replay):
    real,cats,swaps,skipped,changed=truth_sets(real_csv,cat_csv,review_csv,audit_csv)
    # The replay input for swaps is a replay-ready CSV built from the audit.
    with open(swap_csv,encoding='utf-8-sig',newline='') as f:swaprows=list(csv.DictReader(f))
    with open(audit_csv,encoding='utf-8-sig',newline='') as f:
        expected_swaps={r['case_id'] for r in csv.DictReader(f) if r['group']=='synthetic-wrong-area'}
    if {r['case_id'] for r in swaprows}!=expected_swaps:
        raise ValueError('swap CSV does not match audit')
    rp=read_replay(real_replay,'intent-v1.6')
    if set(rp)!=set(real)|set(skipped):raise ValueError('real replay IDs mismatch reviewed truth')
    swap_preds=read_replay(swap_replay,'intent-v1.6')
    if set(swap_preds)!=expected_swaps:raise ValueError('swap replay IDs mismatch audit')
    return {'interpretation':'Research-only opt-in v1.6; addressing reviews revise real truth; synthetic scores separate',
            'boundary_ids_unscored':skipped,'real_truth_changed':changed,
            'real':v15.score(real,{k:v for k,v in rp.items() if k in real}),
            'synthetic_cat':v15.score(cats,read_replay(cat_replay,'intent-v1.6')),
            'synthetic_wrong_area':v15.score(swaps,{k:v for k,v in swap_preds.items() if k in swaps})}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real_csv','real_replay','cat_csv','cat_replay','review_csv','audit_csv','swap_csv','swap_replay'):p.add_argument(name,type=Path)
    p.add_argument('-o','--output',type=Path);a=p.parse_args()
    result=analyze(*(getattr(a,name) for name in ('real_csv','real_replay','cat_csv','cat_replay','review_csv','audit_csv','swap_csv','swap_replay')))
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if a.output:a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
