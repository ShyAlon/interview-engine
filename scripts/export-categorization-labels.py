#!/usr/bin/env python3
"""Export blind-ready adjudication cases plus a separately stored system key."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from interview_engine.categorization_shadow import adjudication_sample,read_records,confidence_band
p=argparse.ArgumentParser();p.add_argument('jsonl',nargs='+');p.add_argument('-o','--output',required=True);p.add_argument('--key-output');p.add_argument('--agreements-per-stratum',type=int,default=2);a=p.parse_args()
records=adjudication_sample(read_records(a.jsonl),a.agreements_per_stratum);rows=[];keys=[]
for r in records:
    swap=int(hashlib.sha256(r['input_ref'].encode()).hexdigest(),16)%2==1
    sides=[('jev',r['jev']),('baseline',r['baseline'])] if swap else [('baseline',r['baseline']),('jev',r['jev'])]
    case_id=hashlib.sha256((r['input_ref']+'|'+r['slot']).encode()).hexdigest()[:16]
    inp=r.get('input') or {}
    rows.append({'case_id':case_id,'input_ref':r['input_ref'],'slot':r['slot'],'session_id':r.get('session_id') or '',
      'input_state':inp.get('state',''),'input_context':json.dumps(inp.get('context',{}),sort_keys=True),
      'rubric':json.dumps(inp.get('questions',{}),sort_keys=True),'rubric_version':inp.get('rubric_version',''),
      'system_a_verdict':json.dumps(sides[0][1].get('verdict',{}),sort_keys=True),'system_a_confidence':json.dumps(sides[0][1].get('confidence',{}),sort_keys=True),
      'system_b_verdict':json.dumps(sides[1][1].get('verdict',{}),sort_keys=True),'system_b_confidence':json.dumps(sides[1][1].get('confidence',{}),sort_keys=True),
      'sample_reason':'disagreement' if not r.get('agreement') else 'agreement:'+confidence_band(r['jev']),
      'human_verdict':'','winner':'','reviewer_id':'','review_notes':''})
    keys.append({'case_id':case_id,'system_a':sides[0][0],'system_b':sides[1][0],'baseline_model':r['baseline'].get('model'),'jev_model':r['jev'].get('model')})
fields=list(rows[0]) if rows else ['case_id','input_ref','slot','session_id','input_state','input_context','rubric','rubric_version','system_a_verdict','system_a_confidence','system_b_verdict','system_b_confidence','sample_reason','human_verdict','winner','reviewer_id','review_notes']
with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
key_path=Path(a.key_output or str(a.output)+'.key.json');key_path.write_text(json.dumps(keys,indent=2)+'\n')
print(a.output);print(key_path)
