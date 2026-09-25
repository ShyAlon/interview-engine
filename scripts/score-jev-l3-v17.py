#!/usr/bin/env python3
"""Score opt-in v1.7 on reviewed JEV-L3 truth: 185 real, 12 cats, 6 swaps.

The reviewed boundary/unsure cases remain unscored. Source export, replay,
review and audit files are read only. Use a v1.7 replay for each set.
"""
from __future__ import annotations
import argparse,csv,importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('v16',Path(__file__).with_name('score-jev-l3-v16.py'))
v16=importlib.util.module_from_spec(spec);spec.loader.exec_module(v16)

def analyze(real_csv,real_replay,cat_csv,cat_replay,review_csv,audit_csv,swap_csv,swap_replay):
    with open(real_csv,encoding='utf-8-sig',newline='') as f:real_rows=list(csv.DictReader(f))
    if len(real_rows)!=185 or len({r['case_id'] for r in real_rows})!=185:
        raise ValueError('expected 185 distinct filtered real cases')
    result=v16.analyze(real_csv,real_replay,cat_csv,cat_replay,review_csv,audit_csv,swap_csv,swap_replay,variant='intent-v1.7')
    if result['real']['n']!=183 or result['synthetic_cat']['n']!=12 or result['synthetic_wrong_area']['n']!=6:
        raise ValueError('expected 183 scored real (two boundary), 12 cats, six swaps')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real_csv','real_replay','cat_csv','cat_replay','review_csv','audit_csv','swap_csv','swap_replay'):p.add_argument(name,type=Path)
    p.add_argument('-o','--output',type=Path);a=p.parse_args()
    result=analyze(*(getattr(a,name) for name in ('real_csv','real_replay','cat_csv','cat_replay','review_csv','audit_csv','swap_csv','swap_replay')))
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if a.output:a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
