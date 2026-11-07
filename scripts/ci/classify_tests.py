#!/usr/bin/env python3
"""Deterministic test-file classification for the fast PR gate.

Every tests/test_*.py file is either heavy or fast:
- heavy: boots a throwaway Postgres (managed_pg_server/managed_pool via
  pgserver) or drives a real browser. These run in the full suite on
  workflow_dispatch, the weekly schedule, and ready_for_review.
- fast: everything else; it gates every pull_request event.

Content-based and deterministic, mirroring delivery-api's classifier, so the
gate and the full suite partition tests without a hand-maintained list.
Misclassification degrades safely: a PG test in the fast set just boots its
throwaway server (slower, still correct); a fast file sent to the full suite
only loses per-PR coverage until the weekly run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PG_TOKENS = ('managed_pg_server', 'managed_pool', 'pgserver')
BROWSER_TOKENS = ('playwright', 'sync_playwright', 'async_playwright')

def heavy_reason(path):
    src = path.read_text(errors='replace')
    for token in PG_TOKENS:
        if token in src:
            return 'pg:' + token
    for token in BROWSER_TOKENS:
        if token in src:
            return 'browser:' + token
    return None

def classify():
    fast, heavy = [], []
    for path in sorted((ROOT / 'tests').rglob('test_*.py')):
        reason = heavy_reason(path)
        (heavy if reason else fast).append(path.relative_to(ROOT).as_posix())
    return fast, heavy

def main(argv):
    mode = argv[1] if len(argv) > 1 else 'fast'
    fast, heavy = classify()
    if mode == 'fast':
        print('\n'.join(fast))
    elif mode == 'heavy':
        print('\n'.join(heavy))
    elif mode == 'summary':
        print(f'fast={len(fast)} heavy={len(heavy)} total={len(fast)+len(heavy)}')
    else:
        raise SystemExit('usage: classify_tests.py [fast|heavy|summary]')

if __name__ == '__main__':
    main(sys.argv)
