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

def test_v1_5_is_opt_in_and_old_default_unchanged():
    from interview_engine.categorization_rubrics import get_variant, configured_variant
    from interview_engine.intent_taxonomy import INTENT_ORDER
    assert INTENT_ORDER==('answer','delegate','defer','skip','enough')
    assert configured_variant('intent').id=='intent-v1'
    v=get_variant('intent-v1.5')
    assert not v.production_ok and v.rubric_version=='categorization-request-v1.5'
    assert tuple(v.questions['intent'].criteria)==('answer','proceed','skip','enough','non-answer')
    assert 'office decor' in v.questions['intent'].criteria['non-answer']

def test_v1_5_real_truth_reviewed_mapping(tmp_path):
    mod=load('score-jev-l2-v15.py')
    p=tmp_path/'r.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','human_verdict']);w.writeheader()
        for cid,label in [('708e80db09a22198','defer'),('6fc6ef8c785d964a','defer'),('x','delegate'),('bf15e609120d6bba','answer')]:
            w.writerow({'case_id':cid,'human_verdict':json.dumps({'intent':[label]})})
    assert mod.real_truth(p)=={'708e80db09a22198':['non-answer'],'6fc6ef8c785d964a':['proceed'],
                               'x':['proceed'],'bf15e609120d6bba':['enough']}
    result=mod.score({'a':['non-answer'],'b':['proceed']},{'a':'answer','b':'proceed'})
    assert result['n']==2 and result['correct']==1 and result['per_label']['non-answer']['recall']==0
