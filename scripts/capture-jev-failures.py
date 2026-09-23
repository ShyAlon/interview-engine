#!/usr/bin/env python3
"""Re-issue every request where Jev-v1 disagreed with Shy's label and capture
the raw wire exchange for inspection.

  .venv/bin/python scripts/capture-jev-failures.py LABELED.csv KEY.json -o jev-failures.json

Covers intent (label) and coverage (relevance). For each failing case the JSON
holds: the stored input package, his label, Jev-v1's original verdict and
confidence, and for the re-issued call every HTTP attempt with the full
request (method, URL, headers, body) and full response (status, headers,
body). Bodies are stored twice: as text and as base64 of the exact bytes.
The Authorization header is replaced by "Bearer <redacted>"; nothing else is
changed. The request is rebuilt from the stored rubric, so it is the same
rubric Jev saw in the original run (the v1 rubric).

Needs TYPESAFE_API_KEY (env or the repo .env). Stops at the first error unless --keep-going.
"""
from __future__ import annotations
import argparse, base64, csv, importlib.util, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


score = _load("score-rubric-variants"); replay = _load("replay-jev-rubric-variants")
FIELDS = {"intent": "intent", "coverage": "relevance"}


def _body(b: bytes | None):
    b = b or b""
    try: text = b.decode("utf-8")
    except UnicodeDecodeError: text = None
    return {"text": text, "base64": base64.b64encode(b).decode("ascii"), "byte_length": len(b)}


def recording_transport(log, inner=None):
    import httpx2
    class Recording(httpx2.BaseTransport):
        def __init__(self): self._inner = inner or httpx2.HTTPTransport()
        def handle_request(self, request):
            body = request.read()
            headers = [[k, "Bearer <redacted>" if k.lower() == "authorization" else v]
                       for k, v in request.headers.multi_items()]
            entry = {"request": {"method": request.method, "url": str(request.url), "headers": headers, "body": _body(body)}}
            log.append(entry)
            response = self._inner.handle_request(request)
            try: wire = b"".join(response.stream)  # exact bytes off the wire (still content-encoded if compressed)
            finally:
                close = getattr(response.stream, "close", None)
                if close: close()
            replay = httpx2.Response(response.status_code, headers=response.headers, stream=httpx2.ByteStream(wire),
                                     extensions=response.extensions, request=request)
            decoded = replay.read()  # what the SDK parses
            entry["response"] = {"status": response.status_code, "headers": [[k, v] for k, v in response.headers.multi_items()],
                                 "body": _body(decoded), "wire_body_base64": base64.b64encode(wire).decode("ascii")}
            return replay
        def close(self): self._inner.close()
    return Recording()


def failing_rows(rows, key):
    out = []
    for r in rows:
        field = FIELDS.get(r["slot"]); k = key.get(r["case_id"])
        if not field or not k: continue
        truth = score.truth(r, field)
        if not truth: continue
        side = "system_a" if k["system_a"] == "jev" else "system_b"
        verdict = score._parse(r[side + "_verdict"]) or {}; conf = score._parse(r[side + "_confidence"]) or {}
        if str(verdict.get(field)) != truth:
            out.append((r, field, truth, verdict, conf))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labeled"); ap.add_argument("key"); ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--keep-going", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out_path = Path(a.output)
    if out_path.exists(): raise SystemExit(f"{out_path} exists; pick a new name")
    rows = list(csv.DictReader(open(a.labeled, encoding="utf-8-sig", newline="")))
    key = {k["case_id"]: k for k in json.load(open(a.key))}
    fails = failing_rows(rows, key)
    print(f"failing Jev-v1 cases: {len(fails)} (intent {sum(f[1]=='intent' for f in fails)}, relevance {sum(f[1]=='relevance' for f in fails)})")
    if a.dry_run: return 0

    from interview_engine import model_config
    from interview_engine.categorization import ClassificationRequest, TypeSafeProvider
    from interview_engine.categorization_shadow import input_package
    api_key = model_config._env("TYPESAFE_API_KEY")
    print(f"TYPESAFE_API_KEY present: {bool(api_key)}  typesafe-sdk installed: {TypeSafeProvider.available()}")
    if not api_key or not TypeSafeProvider.available(): raise SystemExit("Jev unavailable: need TYPESAFE_API_KEY and typesafe-sdk in this venv")
    from typesafe_sdk import TypeSafeClient
    import typesafe_sdk

    cases = []
    for i, (r, field, truth, verdict, conf) in enumerate(fails, 1):
        stored_q = {k: replay._question(v) for k, v in json.loads(r["rubric"] or "{}").items()}
        req = ClassificationRequest(r["slot"], r["input_state"], stored_q, json.loads(r.get("input_context") or "{}"))
        log: list = []
        client = TypeSafeClient(api_key=api_key, transport=recording_transport(log))
        case = {"case_id": r["case_id"], "input_ref": r["input_ref"], "slot": r["slot"], "field": field,
                "input_package": {**input_package(req), "rubric_version": r.get("rubric_version") or ""},
                "human_label": truth, "human_verdict_raw": r.get("human_verdict") or "", "winner": r.get("winner") or "",
                "review_notes": r.get("review_notes") or "",
                "jev_v1_original": {"verdict": verdict, "confidence": conf}, "reissued": {"exchanges": log}}
        try:
            res = TypeSafeProvider(client=client).classify(req)
            v = {k: x.value for k, x in res.answers.items()}
            case["reissued"].update({"status": "ok", "model": res.model, "verdict": v,
                                     "confidence": {k: x.confidence for k, x in res.answers.items()},
                                     "probabilities": {k: dict(x.probabilities) for k, x in res.answers.items()},
                                     "same_as_original": str(v.get(field)) == str(verdict.get(field))})
            print(f"[{i}/{len(fails)}] {r['case_id']} {field}: label={truth} jev_v1={verdict.get(field)} reissued={v.get(field)}")
        except Exception as e:  # noqa: BLE001
            case["reissued"].update({"status": "error", "error": {"type": type(e).__name__, "message": str(e)}})
            print(f"[{i}/{len(fails)}] {r['case_id']} ERROR {type(e).__name__}: {e}")
            if not a.keep_going:
                cases.append(case); break
        finally:
            client.close()
        cases.append(case)
    doc = {"generated": datetime.now(timezone.utc).isoformat(), "typesafe_sdk_version": getattr(typesafe_sdk, "__version__", None),
           "labeled_csv": str(a.labeled), "failing_cases": len(fails), "captured": len(cases), "cases": cases}
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(cases)} cases)")
    return 0 if all(c["reissued"].get("status") == "ok" for c in cases) else 1


if __name__ == "__main__":
    sys.exit(main())
