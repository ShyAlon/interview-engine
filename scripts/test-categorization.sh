#!/usr/bin/env bash
# Focused categorization/Jev diagnostics. Safe with or without TYPESAFE_API_KEY.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CATEGORIZATION_TEST_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="${CATEGORIZATION_TEST_PYTHON:-python3}"
failures=0
progress(){ printf '%s | %-24s | %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1" "$2"; }
run_check(){
  name="$1"; shift
  progress "$name.start" "command=$*"
  tmp="$(mktemp "${TMPDIR:-/tmp}/categorization-check.XXXXXX")"
  if "$@" >"$tmp" 2>&1; then
    progress "$name.pass" "expected state reached"
    rm -f "$tmp"
  else
    code=$?; failures=$((failures+1))
    progress "$name.FAIL" "exit=$code command=$*" >&2
    echo "=== FULL DEVIATION OUTPUT: $name ===" >&2
    cat "$tmp" >&2
    echo "=== END DEVIATION OUTPUT ===" >&2
    rm -f "$tmp"
  fi
}
key_present="$($PYTHON -c 'from interview_engine.model_config import _env; print("yes" if _env("TYPESAFE_API_KEY") else "no")' 2>/dev/null || echo no)"
key_present="${key_present:-no}"
progress categorization.begin "python=$PYTHON key_present=$key_present"
if [ "$key_present" = yes ] && ! "$PYTHON" -c 'import typesafe_sdk' >/dev/null 2>&1; then
  progress sdk.bootstrap "key is configured but typesafe-sdk is absent; creating/updating $ROOT/.venv"
  python3 -m venv "$ROOT/.venv" || { progress sdk.FAIL "could not create $ROOT/.venv" >&2; exit 1; }
  "$ROOT/.venv/bin/python" -m pip install -q -e "$ROOT[dev]" || { progress sdk.FAIL "project dependency installation failed" >&2; exit 1; }
  PYTHON="$ROOT/.venv/bin/python"
  progress sdk.pass "focused runtime ready with pinned typesafe-sdk"
fi
run_check conformance "$PYTHON" -m pytest -vv --tb=long -s \
  "$ROOT/tests/test_categorization.py" "$ROOT/tests/test_scope_gate.py" \
  "$ROOT/tests/test_coverage.py"
run_check slot-contracts "$PYTHON" -m interview_engine.categorization_diagnostics --mode contracts
if [ "$key_present" = yes ]; then
  run_check live-jev "$PYTHON" -m interview_engine.categorization_diagnostics --mode live
else
  progress live-jev.skip "TYPESAFE_API_KEY is absent; fallback/conformance checks completed, live API check skipped"
fi
if [ "$failures" -ne 0 ]; then
  progress categorization.FAIL "$failures focused check(s) deviated" >&2
  exit 1
fi
progress categorization.pass "all available focused checks passed"
