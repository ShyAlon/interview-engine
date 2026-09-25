#!/usr/bin/env python3
"""Offline JEV-L2 sensitivity analysis, not a four-choice classifier evaluation.

Maps historical defer/delegate predictions to `proceed`, applies Shy's reviewed
case rulings to a copy of truth, sets aside the non-answer case, and uses strict
four-class scoring. Neither source labels nor replay records are changed.
"""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
import importlib.util
_strict_spec=importlib.util.spec_from_file_location("_jev_strict_scoring",Path(__file__).with_name("_jev_strict_scoring.py"))
_strict=importlib.util.module_from_spec(_strict_spec);_strict_spec.loader.exec_module(_strict)
label_stats=_strict.label_stats

LABELS=('answer','proceed','skip','enough')
# September 25 reviewer rulings: B1/B2/C1 proceed, B3 non-answer held out,
# C2 unchanged, C3 enough, C4 skip. Group A's 15 defer/delegate cases merge.
OVERRIDES={
 '6fc6ef8c785d964a':['proceed'], '6cc798ab34b39db2':['proceed'],
 '588dedad98ae7425':['proceed'], 'bf15e609120d6bba':['enough'],
 'bd2b9866e0d21ce6':['skip'],
}
HOLDOUT='708e80db09a22198'
def merged(label): return 'proceed' if label in ('defer','delegate') else label

def analyze(source,paths):
    rows=list(csv.DictReader(open(source,encoding='utf-8-sig',newline='')))
    truth={}
    for r in rows:
        cid=r['case_id']
        if cid in truth: raise ValueError('duplicate truth ID: '+cid)
        if cid==HOLDOUT:continue
        raw=json.loads(r['human_verdict'])['intent']
        labels=OVERRIDES.get(cid,[merged(x) for x in raw])
        if not labels or set(labels)-set(LABELS):raise ValueError('invalid truth: '+cid)
        truth[cid]=labels
    results={}
    for path in paths:
        records=[json.loads(line) for line in open(path) if line.strip()]
        variants={x['replay']['variant_id'] for x in records}
        if len(variants)!=1:raise ValueError('mixed or empty replay: '+str(path))
        variant=variants.pop()
        by_id={}
        for x in records:
            cid=x['replay']['case_id']
            if cid in by_id:raise ValueError('duplicate replay ID: '+cid)
            if x['jev']['status']!='ok':raise ValueError('non-ok replay: '+cid)
            by_id[cid]=merged(x['jev']['verdict']['intent'])
        missing=set(truth)-set(by_id)
        if missing:raise ValueError(f'{variant} missing {len(missing)} truth IDs')
        if set(by_id)-set(truth)-{HOLDOUT}:raise ValueError('unexpected replay IDs')
        confusion=Counter()
        for cid,ls in truth.items():
            pred=by_id[cid]
            confusion[pred if pred in ls else ls[0],pred]+=1
        # Reuse the same strict set-valued metrics as native v1.5/v1.6,
        # while preserving the historical report's shape and confusion rule.
        stats=label_stats(truth,{cid:by_id[cid] for cid in truth},LABELS)
        results[variant]={'n':stats['n'],'accuracy':stats['accuracy'],'macro_f1':stats['macro_f1'],
                          'per_label':{label:{'precision':v['precision'],'recall':v['recall'],
                                              'f1':v['f1'],'support':v['support']}
                                       for label,v in stats['per_label'].items()},
                          'confusion':{f'{t}->{p}':n for (t,p),n in sorted(confusion.items())}}
    return {'source_rows':len(rows),'held_out_non_answer':HOLDOUT,'scored_rows':len(truth),
            'truth_distribution':dict(Counter('+'.join(x) for x in truth.values())),
            'sensitivity_only':'historical five-choice predictions remapped, no new model calls',
            'results':results}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('filtered_csv',type=Path);p.add_argument('replays',nargs='+',type=Path)
    p.add_argument('-o','--output',type=Path)
    a=p.parse_args();result=analyze(a.filtered_csv,a.replays)
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if a.output:a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
