import csv,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location('followup',ROOT/'scripts'/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_merged_strict_and_holdout(tmp_path):
    mod=load('score-jev-l2-merged.py')
    rows=[('708e80db09a22198','defer'),('6fc6ef8c785d964a','defer'),('a90ef9d67e072b60',['answer','skip']),('x','delegate')]
    source=tmp_path/'truth.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','human_verdict']);w.writeheader()
        for cid,label in rows:w.writerow({'case_id':cid,'human_verdict':json.dumps({'intent':label if isinstance(label,list) else [label]})})
    replay=tmp_path/'replay.jsonl'
    with replay.open('w') as f:
        for cid,pred in [('708e80db09a22198','answer'),('6fc6ef8c785d964a','defer'),('a90ef9d67e072b60','answer'),('x','answer')]:
            f.write(json.dumps({'replay':{'variant_id':'intent-v1','case_id':cid},'jev':{'status':'ok','verdict':{'intent':pred}}})+'\n')
    result=mod.analyze(source,[replay])
    assert result['scored_rows']==3 and result['truth_distribution']=={'proceed':2,'answer+skip':1}
    assert result['results']['intent-v1']['accuracy']==2/3 # answer is NOT compatible with proceed

def test_cat_probe_deterministic_and_unlabeled(tmp_path):
    mod=load('make-jev-l2-cat-probe.py')
    source=tmp_path/'source.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','input_state','human_verdict','slot']);w.writeheader()
        for cid in mod.CASES:
            w.writerow({'case_id':cid,'input_state':'Question: What is the design?\nFounder answer: It is blue.',
                        'human_verdict':'{"intent":["answer"]}','slot':'intent'})
    fields,rows=mod.build(source)
    assert len(rows)==12 and len({x['case_id'] for x in rows})==12
    assert rows==mod.build(source)[1]
    assert all(not x['human_verdict'] and x['input_state'].startswith('Question: What is the design?\nFounder answer: ') for x in rows)
