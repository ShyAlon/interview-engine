#!/usr/bin/env python3
"""Build a blind JEV-L3 question-addressing review set and standalone HTML.

Includes 20 disputed v1.5 non-answer calls, eight terse/partial controls,
and six synthetic wrong-area product replies. Original export/replay unchanged.
No old or model verdicts are shown to the reviewer. Only the reviewer export
is ground truth; synthetic source-answer transfers are audit-tagged in CSV.
"""
from __future__ import annotations
import argparse,base64,csv,hashlib,io,json,re
from pathlib import Path

CONTROLS=(
 '6991711c2cb46942','92bbd0b57b8ec950','0cdb1e72a090b542',
 '3b0a3c5658a30571','43b9223b11cbd6d1','17e80d1a953ce53c',
 'f2d3ca092d04723a','328940847a48f473',
)
# Target question, source of a substantive *different* product answer.
# Keep source provenance in the audit CSV, but hide it from the reviewer UI.
SWAPS=(
 ('708e80db09a22198','0cdb1e72a090b542'), # data-loss failure <- historical data
 ('6fc6ef8c785d964a','43b9223b11cbd6d1'), # encryption <- visualizations
 ('f86f07d67f204e89','3b0a3c5658a30571'), # registration-data location <- security
 ('a90ef9d67e072b60','17e80d1a953ce53c'), # email failure <- evaluation metric
 ('bd2b9866e0d21ce6','328940847a48f473'), # growth <- integrations
 ('c4a6e49e9343fa6d','92bbd0b57b8ec950'), # persistence <- evaluation method
)
COLS=('case_id','source_case_id','answer_source_case_id','group','question','founder_answer','verdict','review_notes')
QA=re.compile(r'^Question:\s*(.*?)\nFounder answer:\s*(.*)$',re.S)

def pair(row):
    match=QA.match(row['input_state'].strip())
    if not match:raise ValueError('unparseable Q/A: '+row['case_id'])
    return tuple(x.strip() for x in match.groups())

def build(source,replay):
    with open(source,encoding='utf-8-sig',newline='') as f:
        items=list(csv.DictReader(f));rows={r['case_id']:r for r in items}
    if len(items)!=len(rows):raise ValueError('duplicate source IDs')
    preds={}
    with open(replay,encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            rec=json.loads(line);cid=rec['replay']['case_id']
            if cid in preds:raise ValueError('duplicate replay ID: '+cid)
            if rec['replay']['variant_id']!='intent-v1.5' or rec['jev']['status']!='ok':raise ValueError('wrong or failed replay: '+cid)
            preds[cid]=rec['jev']['verdict']['intent']
    if set(rows)!=set(preds):raise ValueError('source/replay ID mismatch')
    flagged=[r['case_id'] for r in items if preds[r['case_id']]=='non-answer' and r['case_id']!='708e80db09a22198']
    if len(flagged)!=20:raise ValueError(f'expected exactly 20 disputed calls, got {len(flagged)}')
    if set(CONTROLS)&set(flagged) or any(preds.get(cid)!='answer' for cid in CONTROLS):raise ValueError('controls are not unflagged answers')
    def row(cid,grp,answer_id=None):
        q,a=pair(rows[cid]);source_id=answer_id or cid
        if answer_id:a=pair(rows[answer_id])[1]
        out_id=(hashlib.sha256(('JEV-L3|'+cid+'|'+source_id+'|'+q+'|'+a).encode()).hexdigest()[:16] if answer_id else cid)
        return dict(case_id=out_id,source_case_id=cid,answer_source_case_id=source_id,group=grp,
                    question=q,founder_answer=a,verdict='',review_notes='')
    result=[row(cid,'real-disputed') for cid in flagged]
    result += [row(cid,'real-control') for cid in CONTROLS]
    result += [row(cid,'synthetic-wrong-area',aid) for cid,aid in SWAPS]
    if len({r['case_id'] for r in result})!=len(result):raise ValueError('duplicate review IDs')
    return result

def html(rows):
    # Only Q/A, opaque case IDs and saved labels reach the UI. No system labels,
    # category hints or swap provenance appear even inside the HTML payload.
    payload=[{k:r[k] for k in ('case_id','question','founder_answer')} for r in rows]
    raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    key='jev-l3:'+hashlib.sha256(raw.encode()).hexdigest()[:16]
    template=Path(__file__).parent/'templates'/'jev-l3-review.html'
    result=template.read_text(encoding='utf-8')
    for k,v in {'__CASES_JSON__':raw,'__STORE_KEY__':key,'__EXPORT_NAME__':'JEV-L3-reviewed.csv'}.items():
        if result.count(k)!=1:raise ValueError('template token count: '+k)
        result=result.replace(k,v)
    return result,key

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('filtered_csv',type=Path);p.add_argument('v15_replay',type=Path)
    p.add_argument('-o','--output',type=Path,required=True,help='HTML output path')
    p.add_argument('--audit-csv',type=Path,help='full review manifest; default beside HTML')
    a=p.parse_args(argv)
    rows=build(a.filtered_csv,a.v15_replay);page,key=html(rows)
    audit=a.audit_csv or a.output.with_suffix('.csv')
    if len({a.output.resolve(),audit.resolve(),a.filtered_csv.resolve(),a.v15_replay.resolve()})!=4:p.error('input and output paths must differ')
    for path in (audit,a.output):path.parent.mkdir(parents=True,exist_ok=True)
    with audit.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader();w.writerows(rows)
    a.output.write_text(page,encoding='utf-8')
    print(f'JEV-L3: {len(rows)} cases (20 disputed, 8 real controls, 6 synthetic wrong-area); storage {key}.')
    print(f'Review HTML: {a.output}; audit CSV: {audit}. Source CSV and replay unchanged.')
if __name__=='__main__':main()
