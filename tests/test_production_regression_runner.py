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
                'replay':{'case_id':cid,'variant_id':'intent-v1.4','repeat':1},'jev':{'status':status}}
    write(record('c1'),record('c2'))
    assert len(r.records(p,rows,'intent-v1.4','intent'))==2
    for bad,match in [((record('c1'),),'ID mismatch'),
                      ((record('c1'),record('c1')),'duplicate replay'),
                      ((record('c1'),record('c2','error')),'variant/slot/status mismatch'),
                      ((record('c1'),record('c2',ref='sha256:wrong')),'input ref mismatch')]:
        write(*bad)
        with pytest.raises(ValueError,match=match):r.records(p,rows,'intent-v1.4','intent')
