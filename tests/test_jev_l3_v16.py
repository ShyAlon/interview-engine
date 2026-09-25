import csv,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location('v16',ROOT/'scripts'/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_reviewed_truth_boundary_control_and_synthetic(tmp_path):
    m=load('score-jev-l3-v16.py');swap=load('make-jev-l3-swaps.py')
    source=tmp_path/'real.csv';review=tmp_path/'review.csv';audit=tmp_path/'audit.csv';cats=tmp_path/'cats.csv'
    groups={**{f'd{i}':'real-disputed' for i in range(20)},
            **{f'c{i}':'real-control' for i in range(8)},
            **{f's{i}':'synthetic-wrong-area' for i in range(6)}}
    groups['d0']='real-disputed'
    real_ids=[*['708e80db09a22198'],*[f'd{i}' for i in range(20)],*[f'c{i}' for i in range(8)]]
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','input_state','human_verdict']);w.writeheader()
        for cid in real_ids:w.writerow({'case_id':cid,'input_state':f'Question: Question for {cid}?\nFounder answer: A {cid}.',
                                       'human_verdict':json.dumps({'intent':['answer']})})
    with review.open('w',newline='') as fr, audit.open('w',newline='') as fa:
        wr=csv.DictWriter(fr,fieldnames=['case_id','question','founder_answer','verdict']);wr.writeheader()
        wa=csv.DictWriter(fa,fieldnames=['case_id','question','founder_answer','group']);wa.writeheader()
        for cid,group in groups.items():
            q=f'Question for {cid}?';ans=f'A {cid}.'
            verdict='boundary' if cid=='d1' else 'non-answer' if cid in ('d0','s0') else 'answer'
            wr.writerow({'case_id':cid,'question':q,'founder_answer':ans,'verdict':verdict})
            wa.writerow({'case_id':cid,'question':q,'founder_answer':ans,'group':group})
    with cats.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','sample_reason']);w.writeheader()
        for i in range(12):w.writerow({'case_id':f'cat{i}','sample_reason':'synthetic-cat-probe'})
    real,cat,synthetic,skipped,changed=m.truth_sets(source,cats,review,audit)
    assert len(real)==28 and len(cat)==12 and len(synthetic)==6
    assert skipped==['d1'] and changed==[('d0',['answer'],['non-answer'])]
    assert real['708e80db09a22198']==['non-answer']
    assert synthetic['s0']==['non-answer']

def test_v16_replay_requires_raw_judgments(tmp_path):
    import pytest
    m=load('score-jev-l3-v16.py');p=tmp_path/'one.jsonl'
    def record(label,control,addressing):
        return {'replay':{'variant_id':'intent-v1.6','case_id':'a'},'jev':{'status':'ok','verdict':{'intent':label,'control':control,'addressing':addressing}}}
    p.write_text(json.dumps(record('non-answer','none','not-addressed'))+'\n')
    assert m.read_replay(p,'intent-v1.6')=={'a':'non-answer'}
    p.write_text(json.dumps(record('answer','none','not-addressed'))+'\n')
    with pytest.raises(ValueError,match='contradicts'):m.read_replay(p,'intent-v1.6')
