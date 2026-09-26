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

def test_pool_v4_mixed_slot_truth_filters_intent_only():
    s=importlib.util.spec_from_file_location('bench',ROOT/'scripts/build-intent-benchmark.py')
    b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
    rows=b.build_rows(json.loads((ROOT/'benchmarks/intent-pool-v4.json').read_text()))
    truth={row['case_id']:json.loads(row['human_verdict'])['intent'] for row in rows if row['slot']=='intent'}
    assert len(rows)==2082 and len(truth)==1147
    assert all(row['case_id'] not in truth for row in rows if row['slot']=='coverage')
    source=(ROOT/'scripts/run-production-regression.py').read_text()
    assert "for r in rows if r['slot']=='intent'} if not is_real" in source

def _write_leg(dest, rows, variant, slot):
    dest.mkdir(parents=True,exist_ok=True)
    def record(cid):
        return {'slot':slot,'input_ref':dict((r['case_id'],r['input_ref']) for r in rows)[cid],
                'input':{'rubric_version':r.INTENT_VARIANTS[variant].rubric_version},
                'replay':{'case_id':cid,'variant_id':variant,'repeat':1},'jev':{'status':'ok'}}
    ids=[row['case_id'] for row in rows if row['slot']==slot]
    (dest/f'replay-{variant}.jsonl').write_text(''.join(json.dumps(record(cid))+'\n' for cid in ids))
    (dest/'SUMMARY.txt').write_text(f'{variant:28s} r1  n={len(ids):3d}  errors=0  tokens=0  latency=0.0s  labels={{}}\n')

def test_resume_plan_reuses_validated_legs_and_lists_spend(tmp_path):
    rows=[{'case_id':'c1','slot':'intent','input_ref':'sha256:abc'},
          {'case_id':'c2','slot':'intent','input_ref':'sha256:def'}]
    jobs=[('leg-done','src',rows,'intent','intent-v1.4'),('leg-missing','src',rows,'intent','intent-v1.4')]
    _write_leg(tmp_path/'leg-done',rows,'intent-v1.4','intent')
    reusable,spend=r.resume_plan(jobs,tmp_path)
    assert reusable==[('leg-done',2)] and spend==[('leg-missing',2)]
    # a partial existing leg (replay without SUMMARY) blocks before any spend
    (tmp_path/'leg-missing').mkdir();(tmp_path/'leg-missing'/'replay-intent-v1.4.jsonl').write_text('')
    with pytest.raises(ValueError,match='partial replay leg'):r.resume_plan(jobs,tmp_path)
    # a stale SUMMARY (wrong n) also blocks
    _write_leg(tmp_path/'leg-missing',rows,'intent-v1.4','intent')
    (tmp_path/'leg-missing'/'SUMMARY.txt').write_text('intent-v1.4 r1  n=999  errors=0\n')
    with pytest.raises(ValueError,match='summary missing or mismatched'):r.resume_plan(jobs,tmp_path)

def test_resume_spend_requires_allow_spend_flag():
    source=(ROOT/'scripts/run-production-regression.py').read_text()
    assert "'--allow-spend'" in source
    assert 'resume_plan(jobs,out)' in source
    assert 'if spend and not a.allow_spend:' in source
