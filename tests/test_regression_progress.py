"""Regression progress is visible without changing replay order or default behavior."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_replayer():
    spec = importlib.util.spec_from_file_location('regression_replayer', ROOT/'scripts/replay-jev-rubric-variants.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ordered_progress_and_wait_heartbeat(capsys):
    module = load_replayer()
    class DelayedResult:
        def __init__(self, value):
            self.value = value
            self.calls = 0
        def result(self, timeout):
            self.calls += 1
            if self.value == 'slow' and self.calls == 1:
                raise module.FutureTimeout()
            return self.value
    class Pool:
        def submit(self, run, case):
            return DelayedResult(case)
    rows = list(module.ordered_results(Pool(), lambda c:c, ['slow', 'fast'], 100, module.time.monotonic(), 'intent-v1.4', wait_seconds=0))
    assert rows == [(1, 'slow'), (2, 'fast')]
    assert '[intent-v1.4] waiting | 0/2 cases written' in capsys.readouterr().out


def test_progress_off_by_default_and_enabled_in_regression():
    source = (ROOT/'scripts/replay-jev-rubric-variants.py').read_text()
    assert 'default=0, help="print timestamped progress every N completed cases' in source
    assert 'done % a.progress_every == 0 or done == len(todo)' in source
    regression = (ROOT/'scripts/run-production-regression.py').read_text()
    assert "'--progress-every',100" in regression
    assert 'progress(idx,len(jobs),name' in regression
