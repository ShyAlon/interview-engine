#!/usr/bin/env python3
"""Backfill v1 shadow rows from an exported transcript where request text is recoverable."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('jsonl');p.add_argument('--transcript',required=True,help='JSON list of {role,content} turns');p.add_argument('-o','--output',required=True);a=p.parse_args()
turns=json.loads(Path(a.transcript).read_text());candidates={}
for i,t in enumerate(turns):
 if t.get('role') not in {'user','human','founder'}:continue
 answer=t.get('content','');prior=next((x.get('content','') for x in reversed(turns[:i]) if x.get('role') in {'assistant','ai'}),'')
 for slot,state in [('coverage',f'Question: {prior}\nAnswer: {answer}'),('intent',f'Question: {prior}\nFounder answer: {answer}')]:
  candidates['sha256:'+hashlib.sha256(state.encode()).hexdigest()]={'state':state,'context':{},'questions':{},'rubric_version':'recovered-v1','rubric_ref':None}
# The first user turn is the recoverable scope input.
if turns:
 first=next((x.get('content','') for x in turns if x.get('role') in {'user','human','founder'}),'')
 candidates['sha256:'+hashlib.sha256(first.encode()).hexdigest()]={'state':first,'context':{},'questions':{},'rubric_version':'recovered-v1','rubric_ref':None}
found=0;out=[]
for line in Path(a.jsonl).read_text().splitlines():
 if not line.strip():continue
 r=json.loads(line);pkg=candidates.get(r.get('input_ref'))
 if pkg and not r.get('input'):r['input']=pkg;r['record_schema_version']=2;found+=1
 out.append(json.dumps(r,separators=(',',':')))
Path(a.output).write_text('\n'.join(out)+'\n');print(f'{a.output}: enriched {found}/{len(out)} records')
