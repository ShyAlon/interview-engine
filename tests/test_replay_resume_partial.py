"""--resume-partial completes a partial replay without re-spending finished cases."""
import importlib.util, json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location('replay', ROOT/'scripts/replay-jev-rubric-variants.py')
rp = importlib.util.module_from_spec(s); s.loader.exec_module(rp)

CASES = [{'case_id': f'c{i}', 'slot': 'intent', 'input_ref': f'sha256:{i}'} for i in range(4)]

def rec(cid, status='ok', variant='intent-v1.4', repeat=1, ref=None, slot='intent'):
    return {'slot': slot, 'input_ref': ref or f'sha256:{cid[1:]}',
            'replay': {'case_id': cid, 'variant_id': variant, 'repeat': repeat},
            'jev': {'status': status}}

def write(path, *records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))

def test_load_partial_keeps_only_valid_ok_records(tmp_path):
    p = tmp_path/'replay-intent-v1.4.jsonl'
    write(p, rec('c0'), rec('c1'), rec('c2', status='error'), rec('c9'),
          rec('c3', variant='intent-v1'), rec('c3', ref='sha256:stale'))
    kept, todo, dropped = rp.load_partial(p, CASES, 'intent-v1.4', 'intent')
    assert [r['replay']['case_id'] for r in kept] == ['c0', 'c1']
    assert [c['case_id'] for c in todo] == ['c2', 'c3']
    assert dropped == 4

def test_load_partial_rejects_duplicate_kept_ids(tmp_path):
    p = tmp_path/'replay-intent-v1.4.jsonl'
    write(p, rec('c0'), rec('c0'))
    with pytest.raises(SystemExit, match='duplicate'):
        rp.load_partial(p, CASES, 'intent-v1.4', 'intent')

def test_load_partial_empty_file_runs_everything(tmp_path):
    p = tmp_path/'replay-intent-v1.4.jsonl'
    p.write_text('')
    kept, todo, dropped = rp.load_partial(p, CASES, 'intent-v1.4', 'intent')
    assert kept == [] and [c['case_id'] for c in todo] == [c['case_id'] for c in CASES] and dropped == 0
