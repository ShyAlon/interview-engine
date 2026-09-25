#!/usr/bin/env python3
"""Score native intent-v1.5 replay on adjudicated JEV-L2 plus cat probes.

Strict five-class comparison; maps reviewed legacy labels to proceed and
holds the single reviewed off-topic real row as non-answer. Cat probes are
synthetic, intentionally off-topic and scored separately from real cases.
No mutations to the source export or replay files.
"""
from __future__ import annotations
import argparse,csv,json
import importlib.util
from pathlib import Path
_strict_spec=importlib.util.spec_from_file_location("_jev_strict_scoring",Path(__file__).with_name("_jev_strict_scoring.py"))
_strict=importlib.util.module_from_spec(_strict_spec);_strict_spec.loader.exec_module(_strict)
label_stats=_strict.label_stats

# Import the offline reviewed mappings without coupling to a shell filename.
spec=importlib.util.spec_from_file_location('merged',Path(__file__).with_name('score-jev-l2-merged.py'))
merged=importlib.util.module_from_spec(spec);spec.loader.exec_module(merged)
LABELS=(*merged.LABELS,'non-answer')

def score(truth,records):
    return label_stats(truth,records,LABELS)

def read_replay(path):
    out={}
    for line in open(path):
        if not line.strip():continue
        row=json.loads(line);cid=row['replay']['case_id']
        if row['replay']['variant_id']!='intent-v1.5' or row['jev']['status']!='ok':raise ValueError('wrong variant or non-ok: '+cid)
        if cid in out:raise ValueError('duplicate replay ID: '+cid)
        label=row['jev']['verdict']['intent']
        if label not in LABELS:raise ValueError('unknown label: '+str(label))
        out[cid]=label
    return out

def real_truth(source):
    rows=list(csv.DictReader(open(source,encoding='utf-8-sig',newline='')));truth={}
    for r in rows:
        cid=r['case_id']
        if cid in truth:raise ValueError('duplicate truth ID: '+cid)
        if cid==merged.HOLDOUT:ls=['non-answer']
        else:ls=merged.OVERRIDES.get(cid,[merged.merged(x) for x in json.loads(r['human_verdict'])['intent']])
        if set(ls)-set(LABELS):raise ValueError('unknown truth: '+cid)
        truth[cid]=ls
    return truth

def analyze(real_csv,real_replay,cat_csv,cat_replay):
    real=real_truth(real_csv)
    cats=list(csv.DictReader(open(cat_csv,encoding='utf-8-sig',newline='')))
    if len(cats)!=12 or len({r['case_id'] for r in cats})!=12 or any(r['sample_reason']!='synthetic-cat-probe' for r in cats):
        raise ValueError('not the 12-case cat probe set')
    cat_truth={r['case_id']:['non-answer'] for r in cats}
    if set(real)&set(cat_truth):raise ValueError('real and cat IDs overlap')
    return {'interpretation':'Native opt-in intent-v1.5 only; real and synthetic scores separate; no production validation',
            'real':score(real,read_replay(real_replay)), 'synthetic_cat':score(cat_truth,read_replay(cat_replay))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real_csv','real_replay','cat_csv','cat_replay'):p.add_argument(name,type=Path)
    p.add_argument('-o','--output',type=Path);a=p.parse_args()
    result=analyze(a.real_csv,a.real_replay,a.cat_csv,a.cat_replay)
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if a.output:a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
