#!/usr/bin/env python3
"""Export a stable human-label CSV from coverage shadow JSONL."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('jsonl',nargs='+');p.add_argument('-o','--output',required=True);a=p.parse_args()
rows=[]
for source in a.jsonl:
    for line in Path(source).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row.get('slot')!='coverage':continue
        rows.append({'input_ref':row['input_ref'],'session_id':row.get('session_id') or '',
          'schema_id':row.get('schema_id') or '', 'schema_version':row.get('schema_version') or '',
          'baseline_verdict':json.dumps(row['baseline'].get('verdict',{}),sort_keys=True),
          'jev_verdict':json.dumps(row['jev'].get('verdict',{}),sort_keys=True),
          'human_relevance':'','human_covered_facets':'','reviewer_id':'','review_notes':''})
fields=list(rows[0]) if rows else ['input_ref','session_id','schema_id','schema_version','baseline_verdict','jev_verdict','human_relevance','human_covered_facets','reviewer_id','review_notes']
with open(a.output,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(a.output)
