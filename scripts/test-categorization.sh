#!/usr/bin/env bash
# Focused categorization/Jev diagnostics. Safe with or without TYPESAFE_API_KEY.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CATEGORIZATION_TEST_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="${CATEGORIZATION_TEST_PYTHON:-python3}"
progress(){ printf '%s | %-24s | %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$1" "$2"; }
compact_deviation(){
  "$PYTHON" - "$1" "$ROOT" <<'PY'
import json,re,sys
path,root=sys.argv[1:]
lines=open(path,errors="replace").read().splitlines()
keep=[]
for line in lines:
    stripped=line.strip()
    is_failure_json=False
    if stripped.startswith("{"):
        try: is_failure_json=json.loads(stripped).get("event") in {"failure","error"}
        except Exception: pass
    if (root in line or is_failure_json or re.match(r"^(FAILED|ERROR|E   |[A-Za-z_][\w.]*Error:|[A-Za-z_][\w.]*Exception:)",stripped)
            or " short test summary " in line or stripped.startswith("= FAILURES =")):
        if "site-packages" not in line: keep.append(line)
if not keep: keep=lines[-12:]
print("\n".join(keep[-80:]))
PY
}
run_check(){
  name="$1"; shift
  progress "$name.start" "command=$*"
  tmp="$(mktemp "${TMPDIR:-/tmp}/categorization-check.XXXXXX")"
  "$@" >"$tmp" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    progress "$name.pass" "expected state reached"; rm -f "$tmp"; return 0
  fi
  progress "$name.FAIL" "exit=$code command=$*" >&2
  echo "=== COMPACT DEVIATION: $name ===" >&2
  compact_deviation "$tmp" >&2
  echo "=== END DEVIATION ===" >&2
  rm -f "$tmp"
  return "$code"
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
TEST_CWD="$(mktemp -d "${TMPDIR:-/tmp}/categorization-hermetic.XXXXXX")"
trap 'rm -rf "$TEST_CWD"' EXIT
run_check conformance bash -c 'cd "$1"; shift; exec env -u TYPESAFE_API_KEY "$@"' _ "$TEST_CWD" \
  "$PYTHON" -m pytest -q --tb=short --rootdir "$ROOT" \
  "$ROOT/tests/test_categorization.py" "$ROOT/tests/test_scope_gate.py" \
  "$ROOT/tests/test_coverage.py" || exit 1
run_check slot-contracts "$PYTHON" -m interview_engine.categorization_diagnostics --mode contracts || exit 1
if [ "$key_present" = yes ]; then
  run_check live-jev "$PYTHON" -m interview_engine.categorization_diagnostics --mode live || exit 1
else
  progress live-jev.skip "TYPESAFE_API_KEY is absent; fallback/conformance checks completed, live API check skipped"
fi
progress categorization.pass "all available focused checks passed"
