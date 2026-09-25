"""Technical gate fails closed; no network or Jev calls."""
import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('regression',ROOT/'scripts/run-production-regression.py')
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)

def test_replay_validation_rejects_missing_duplicate_non_ok_and_changed_input(tmp_path):
    rows=[{'case_id':'c1','slot':'intent','input_ref':'sha256:abc'},
          {'case_id':'c2','slot':'intent','input_ref':'sha256:def'}]
    p=tmp_path/'replay-intent-v1.4.jsonl'
    def write(*records):p.write_text(''.join(json.dumps(x)+'\n' for x in records))
    def record(cid, status='ok', ref=None):
        return {'slot':'intent','input_ref':ref or dict(c1='sha256:abc',c2='sha256:def')[cid],
                'input':{'rubric_version':'categorization-request-v1.4'},
                'replay':{'case_id':cid,'variant_id':'intent-v1.4','repeat':1},'jev':{'status':status}}
    write(record('c1'),record('c2'))
    assert len(r.records(p,rows,'intent-v1.4','intent'))==2
    for bad,match in [((record('c1'),),'ID mismatch'),
                      ((record('c1'),record('c1')),'duplicate replay'),
                      ((record('c1'),record('c2','error')),'variant/slot/status mismatch'),
                      ((record('c1'),record('c2',ref='sha256:wrong')),'input ref mismatch')]:
        write(*bad)
        with pytest.raises(ValueError,match=match):r.records(p,rows,'intent-v1.4','intent')

def test_default_matrix_is_lean_and_full_flag_extends_it():
    source=(ROOT/'scripts/run-production-regression.py').read_text()
    assert 'if a.full_four_pools else (4,)' in source
    assert "'scope':'full-four-pools' if a.full_four_pools else 'lean'" in source
    import importlib.util
    s=importlib.util.spec_from_file_location('bench',ROOT/'scripts/build-intent-benchmark.py')
    b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
    n={}
    for i in range(1,5):
        pool=json.loads((ROOT/f'benchmarks/intent-pool-v{i}.json').read_text())
        n[i]=sum(row['slot']=='intent' for row in b.build_rows(pool))
    v4=json.loads((ROOT/'benchmarks/intent-pool-v4.json').read_text())
    trees=sum(row['slot']=='coverage' for row in b.build_rows(v4))
    assert (n[4],trees,n[4]+trees+185)==(1147,935,2267)
    assert sum(n.values())+trees+185==4687
