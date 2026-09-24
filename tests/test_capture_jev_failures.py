import csv, gzip, importlib.util, json
from pathlib import Path
import httpx2

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture", ROOT / "scripts/capture-jev-failures.py")
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)


def test_recording_transport_keeps_exact_bytes_and_redacts_auth():
    payload = json.dumps({"answers": {"intent": {"choice": "answer"}}}).encode()
    inner = httpx2.MockTransport(lambda req: httpx2.Response(200, headers={"content-encoding": "gzip", "x-request-id": "r1"},
                                                             content=gzip.compress(payload)))
    log = []
    with httpx2.Client(transport=cap.recording_transport(log, inner)) as c:
        r = c.post("https://api.example/v1/system-one", content=b'{"state":"x"}', headers={"Authorization": "Bearer sk-secret"})
    assert r.json() == json.loads(payload)                      # SDK still gets a readable response
    e = log[0]
    assert e["request"]["body"]["text"] == '{"state":"x"}'
    assert ["authorization", "Bearer <redacted>"] in e["request"]["headers"] and "sk-secret" not in json.dumps(log)
    assert e["response"]["body"]["text"] == payload.decode() and e["response"]["status"] == 200
    assert gzip.decompress(__import__("base64").b64decode(e["response"]["wire_body_base64"])) == payload


def test_failing_rows_uses_label_then_winner_and_unblinds_jev():
    rows = [
        {"case_id": "c1", "slot": "intent", "human_verdict": "", "winner": "A",
         "system_a_verdict": '{"intent": "skip"}', "system_a_confidence": "{}",
         "system_b_verdict": '{"intent": "answer"}', "system_b_confidence": '{"intent": 0.61}'},
        {"case_id": "c2", "slot": "coverage", "human_verdict": "", "winner": "tie",
         "system_a_verdict": '{"relevance": "partial"}', "system_a_confidence": "{}",
         "system_b_verdict": '{"relevance": "partial"}', "system_b_confidence": "{}"},
        {"case_id": "c3", "slot": "intent", "human_verdict": "", "winner": "neither",
         "system_a_verdict": '{"intent": "skip"}', "system_a_confidence": "{}",
         "system_b_verdict": '{"intent": "answer"}', "system_b_confidence": "{}"},
    ]
    key = {c: {"system_a": "baseline", "system_b": "jev"} for c in ("c1", "c2", "c3")}
    fails = cap.failing_rows(rows, key)
    assert [(r["case_id"], f, t, v) for r, f, t, v, _ in fails] == [("c1", "intent", "skip", {"intent": "answer"})]


def test_multilabel_sets_and_answer_delegate_compatibility():
    base = {"system_a_confidence": "{}", "system_b_confidence": "{}", "slot": "intent", "winner": "neither"}
    rows = [
        {**base, "case_id": "m1", "human_verdict": '{"intent": ["dismiss", "skip"]}',
         "system_a_verdict": '{"intent": "skip"}', "system_b_verdict": '{"intent": "answer"}'},     # jev answer not in set -> miss
        {**base, "case_id": "m2", "human_verdict": '{"intent": ["answer", "delegate"]}',
         "system_a_verdict": '{"intent": "skip"}', "system_b_verdict": '{"intent": "delegate"}'},   # in set -> ok
        {**base, "case_id": "m3", "human_verdict": '{"intent": "delegate"}',
         "system_a_verdict": '{"intent": "skip"}', "system_b_verdict": '{"intent": "answer"}'},     # compatible -> ok
        {**base, "case_id": "m4", "human_verdict": '{"intent": ["answer"]}',
         "system_a_verdict": '{"intent": "skip"}', "system_b_verdict": '{"intent": "dismiss"}'},    # miss
    ]
    key = {r["case_id"]: {"system_a": "baseline", "system_b": "jev"} for r in rows}
    fails = cap.failing_rows(rows, key)
    assert [(r["case_id"], t) for r, _, t, _, _ in fails] == [("m1", "skip"), ("m4", "answer")]  # v1.3: dismiss -> skip
    s = cap.score
    assert s.truth_set({"human_verdict": '{"intent": "skip"}'}, "intent") == ["skip"]
    assert s.is_correct("answer", ["delegate"], "intent") and not s.is_correct("answer", ["delegate"], "relevance")
    assert not s.is_correct("dismiss", ["answer", "delegate"], "intent")
    assert s.is_correct("dismiss", ["skip"], "intent") and s.truth_set({"human_verdict": '{"intent": ["dismiss", "skip"]}'}, "intent") == ["skip"]
