"""Every-PR gate runs the fast core; the full suite is on demand.

Owner decision 2026-11-07, mirroring delivery-api #694/#695: the every-PR
gate is the deterministic non-Postgres subset (fast per
scripts/ci/classify_tests.py); the full suite runs on workflow_dispatch, a
weekly schedule, and ready_for_review.
"""
from pathlib import Path
import importlib.util
import yaml

ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/'.github'/'workflows'/'test.yml'

def doc():
    return yaml.safe_load(WF.read_text())

def on_block(d):
    return d.get('on',d.get(True,{}))

def test_gate_job_runs_fast_scope_only():
    gate=doc()['jobs']['gate']
    assert 'strategy' not in gate
    steps='\n'.join(str(s.get('run','')) for s in gate['steps'])
    assert 'ENGINE_TEST_SCOPE=gate' in steps,steps
    assert 'pytest -q\n' not in steps.replace('ENGINE_TEST_SCOPE=gate','')

def test_full_job_is_not_part_of_an_ordinary_pr():
    full=doc()['jobs']['full']
    cond=str(full.get('if',''))
    assert 'workflow_dispatch' in cond and 'schedule' in cond and 'ready_for_review' in cond,cond
    assert 'pull_request' not in cond.replace('ready_for_review',''),cond

def test_weekly_schedule_keeps_full_depth():
    sched=(on_block(doc()) or {}).get('schedule')
    assert sched,'weekly full-suite run missing'
    crons=[e['cron'] for e in sched if 'cron' in e]
    assert len(crons)==1,crons
    fields=crons[0].split()
    assert len(fields)==5 and fields[4]!='*',(
        'owner decision 2026-11-07: full suite weekly, not nightly; '
        f'cron must pin a day-of-week, got {crons[0]!r}')

def test_classifier_partitions_every_test_file():
    spec=importlib.util.spec_from_file_location('classify_tests',ROOT/'scripts'/'ci'/'classify_tests.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fast,heavy=m.classify()
    all_files=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').rglob('test_*.py'))
    assert sorted(fast+heavy)==all_files
    assert not set(fast)&set(heavy)
    for known_heavy in ('tests/test_pg_application_wiring.py',):
        assert known_heavy in heavy,known_heavy
    for known_fast in ('tests/test_ci_trigger_scope.py','tests/test_categorization.py'):
        assert known_fast in fast,known_fast
    assert 'tests/test_ci_fast_gate.py' in fast
    assert m.classify()==(fast,heavy),'classification must be deterministic'
