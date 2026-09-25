"""v1.7 request and scoring gates; frozen v1.6 behavior is unchanged."""
import importlib.util,json
from pathlib import Path
import pytest
from interview_engine.categorization import ClassificationAnswer as A
from interview_engine.categorization_rubrics import get_variant, configured_variant
ROOT=Path(__file__).resolve().parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location('v17',ROOT/'scripts'/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_v17_opt_in_and_precedence():
    assert configured_variant('intent').id=='intent-v1'
    v=get_variant('intent-v1.7')
    assert not v.production_ok and v.rubric_version=='categorization-request-v1.7'
    assert tuple(v.questions)==('control','addressing')
    for control,address,want in [('none','addressed','answer'),('none','not-addressed','non-answer'),
                                  ('proceed','not-addressed','proceed'),('proceed','addressed','answer'),
                                  ('skip','addressed','skip'),('enough','not-addressed','enough')]:
        got=v.apply({'control':A(control,.83),'addressing':A(address,.71)})
        assert got['intent'].value==want
        assert got['intent'].confidence==(.83 if want in ('proceed','skip','enough') else .71)
        assert got['control'].value==control and got['addressing'].value==address
    assert get_variant('intent-v1.6').apply({'control':A('proceed',.83),'addressing':A('addressed',.71)})['intent'].value=='proceed'

def test_v17_replay_checks_variant_and_raw_finalization(tmp_path):
    m=load('score-jev-l3-v16.py');p=tmp_path/'one.jsonl'
    def put(variant,label,control,address):
        p.write_text(json.dumps({'replay':{'variant_id':variant,'case_id':'d720283ad051482f'},
                                 'jev':{'status':'ok','verdict':{'intent':label,'control':control,'addressing':address}}})+'\n')
    put('intent-v1.7','answer','proceed','addressed')
    assert m.read_replay(p,'intent-v1.7')=={'d720283ad051482f':'answer'}
    with pytest.raises(ValueError,match='wrong variant'):m.read_replay(p,'intent-v1.6')
    put('intent-v1.7','proceed','proceed','addressed')
    with pytest.raises(ValueError,match='contradicts'):m.read_replay(p,'intent-v1.7')
    put('intent-v1.7','answer','none','not-addressed')
    with pytest.raises(ValueError,match='contradicts'):m.read_replay(p,'intent-v1.7')

def test_v17_scorer_rejects_bad_real_source(tmp_path):
    scorer=load('score-jev-l3-v17.py');source=tmp_path/'real.csv'
    source.write_text('case_id\na\n')
    with pytest.raises(ValueError,match='185 distinct'):
        scorer.analyze(source,None,None,None,None,None,None,None)
