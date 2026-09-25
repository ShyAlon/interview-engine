import csv,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('l3',ROOT/'scripts'/'build-jev-l3-review.py')
l3=importlib.util.module_from_spec(spec);spec.loader.exec_module(l3)

def test_l3_balanced_blind_and_stable(tmp_path):
    all_ids=['708e80db09a22198',*l3.CONTROLS,*[x for pair in l3.SWAPS for x in pair],*[f'flag{i}' for i in range(20)]]
    ids=list(dict.fromkeys(all_ids));source=tmp_path/'source.csv';replay=tmp_path/'replay.jsonl'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','input_state']);w.writeheader()
        for i,cid in enumerate(ids):w.writerow({'case_id':cid,'input_state':f'Question: Q{i}?\nFounder answer: A{i}.'})
    with replay.open('w') as f:
        for cid in ids:
            label='non-answer' if cid.startswith('flag') or cid=='708e80db09a22198' else 'answer'
            f.write(json.dumps({'replay':{'variant_id':'intent-v1.5','case_id':cid},'jev':{'status':'ok','verdict':{'intent':label}}})+'\n')
    rows=l3.build(source,replay)
    assert len(rows)==34 and [r['group'] for r in rows].count('real-disputed')==20
    assert [r['group'] for r in rows].count('real-control')==8
    assert [r['group'] for r in rows].count('synthetic-wrong-area')==6
    assert all(not r['verdict'] for r in rows)
    html,key=l3.html(rows)
    assert key==l3.html(rows)[1] and all(r['case_id'] in html for r in rows)
    assert 'flag0' in html and 'synthetic-wrong-area' not in html and 'real-control' not in html
    assert 'System A' not in html and 'intent-v1.5' not in html
    assert 'localStorage.setItem' in html and 'Export CSV' in html
    assert 'non-answer' in html and 'Boundary / unsure' in html

def test_l3_rejects_incomplete_replay(tmp_path):
    import pytest
    source=tmp_path/'s.csv';source.write_text('case_id,input_state\nx,"Question: q?\nFounder answer: a"\n')
    replay=tmp_path/'r.jsonl';replay.write_text('')
    with pytest.raises(ValueError,match='source/replay ID mismatch'):l3.build(source,replay)
