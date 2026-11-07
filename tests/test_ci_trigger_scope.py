"""PR CI runs on open/reopen/ready-for-review only, never on every push.

Owner decision 2026-10-10, restored 2026-11-07: automatic PR checks fire on
opened/reopened/ready_for_review. `synchronize` re-runs the suite on every push
to an open PR. Pushes are covered by the local gate (./scripts/test-local.sh).
"""
from pathlib import Path
import yaml

WORKFLOWS=Path(__file__).resolve().parents[1]/'.github'/'workflows'

def pr_triggers():
    found={}
    for wf in sorted(WORKFLOWS.glob('*.yml')):
        doc=yaml.safe_load(wf.read_text())
        on=doc.get('on',doc.get(True,{}))  # PyYAML parses bare `on:` as True
        pr=(on or {}).get('pull_request')
        if pr is not None:
            found[wf.name]=pr
    return found

def test_no_workflow_runs_pr_checks_on_synchronize():
    for name,pr in pr_triggers().items():
        types=pr.get('types') if isinstance(pr,dict) else None
        assert not types or 'synchronize' not in types,(
            f'{name} runs pull_request checks on synchronize (every push); '
            'owner rule is opened/reopened/ready_for_review only')

def test_pr_checks_still_fire_on_open_reopen_ready():
    ci=pr_triggers().get('test.yml')
    assert ci is not None,'test.yml lost its pull_request trigger entirely'
    types=set(ci.get('types',[]))
    assert {'opened','reopened','ready_for_review'} <= types,types
