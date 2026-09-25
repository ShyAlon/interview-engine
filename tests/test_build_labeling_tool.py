"""Drift guard: a regenerated labeling tool's intent options are exactly the
taxonomy's (labels, order, text, multi-select). Hermetic: no browser."""
import base64, csv, importlib.util, io, json, re
from pathlib import Path
from interview_engine.intent_taxonomy import INTENT_ORDER, INTENT_SHORT, MULTI_SELECT, TAXONOMY_VERSION

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_tool", ROOT / "scripts" / "build-labeling-tool.py")
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)

OLD_RUBRIC = json.dumps({"intent": {"instructions": "Classify the founder's control intent", "multiple": False,
                                    "criteria": {"answer": "ordinary product answer", "skip": "old text"}}})


def _row(cid, slot="intent", rubric=OLD_RUBRIC, winner="", hv=""):
    r = {c: "" for c in bt.COLUMNS}
    r.update(case_id=cid, slot=slot, input_state="Question: q\nFounder answer: a", rubric=rubric,
             system_a_verdict='{"intent": "skip"}', system_b_verdict='{"intent": "answer"}', winner=winner, human_verdict=hv)
    return r


def _write(tmp, rows, name="in.csv"):
    p = tmp / name
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bt.COLUMNS); w.writeheader(); w.writerows(rows)
    return p


def _embedded(html, sid):
    return re.search(rf'<script id="{sid}" type="[^"]+">(.*?)</script>', html, re.S).group(1)


def test_regenerated_tool_labels_match_the_taxonomy(tmp_path):
    out = tmp_path / "tool.html"
    assert bt.main([str(_write(tmp_path, [_row("c1"), _row("c2")])), "-o", str(out)]) == 0
    html = out.read_text()
    tax = json.loads(_embedded(html, "taxonomy").replace("<\\/", "</"))
    assert tax["order"] == list(INTENT_ORDER)
    assert tax["short"] == INTENT_SHORT and list(tax["short"]) == list(INTENT_ORDER)
    assert tax["multi_select"] is MULTI_SELECT and tax["version"] == TAXONOMY_VERSION
    # the tool renders intent options from the embedded taxonomy, not from row rubrics or hardcoded lists
    assert "function rubricOf(c)" in html and "TAX.order.forEach" in html and "J(c.rubric)" in html
    assert html.count("J(c.rubric)") == 1  # only inside rubricOf
    assert "new Set(['intent'])" not in html and "__" + "TAXONOMY" not in html
    assert TAXONOMY_VERSION in html


def test_export_shape_is_the_17_columns_and_storage_key_is_per_input(tmp_path):
    a = tmp_path / "a.html"; b = tmp_path / "b.html"; a2 = tmp_path / "a2.html"
    in_a = _write(tmp_path, [_row("c1")], "a.csv"); in_b = _write(tmp_path, [_row("c9")], "b.csv")
    bt.main([str(in_a), "-o", str(a)]); bt.main([str(in_b), "-o", str(b)]); bt.main([str(in_a), "-o", str(a2)])
    key = lambda p: re.search(r"const STORE='([^']+)'", p.read_text()).group(1)
    assert key(a) != key(b) and key(a) == key(a2)
    data = base64.b64decode(_embedded(a.read_text(), "csvdata")).decode()
    assert next(csv.reader(io.StringIO(data))) == bt.COLUMNS == [
        "case_id", "input_ref", "slot", "session_id", "input_state", "input_context", "rubric", "rubric_version",
        "system_a_verdict", "system_a_confidence", "system_b_verdict", "system_b_confidence", "sample_reason",
        "human_verdict", "winner", "reviewer_id", "review_notes"]
    assert "exportBtn" in a.read_text() and "a-labeled.csv" in a.read_text()


def test_only_unlabeled_clear_labels_and_key_subset(tmp_path):
    rows = [_row("u1"), _row("n1", winner="neither"), _row("l1", winner="A", hv='{"intent": "skip"}'),
            _row("s1", slot="scope", winner="neither")]
    key = tmp_path / "key.json"
    key.write_text(json.dumps([{"case_id": c, "system_a": "baseline", "system_b": "jev"} for c in ("u1", "n1", "l1", "s1")]))
    out = tmp_path / "t.html"
    bt.main([str(_write(tmp_path, rows)), "-o", str(out), "--slot", "intent", "--only-unlabeled", "--clear-labels", "--key", str(key)])
    data = list(csv.DictReader(io.StringIO(base64.b64decode(_embedded(out.read_text(), "csvdata")).decode())))
    assert [r["case_id"] for r in data] == ["u1", "n1"] and all(not r["winner"] for r in data)
    assert [k["case_id"] for k in json.loads((tmp_path / "t-key.json").read_text())] == ["u1", "n1"]
    assert "baseline" not in out.read_text()  # key never embedded


def test_manual_intent_tool_hides_comparison_keeps_storage_key(tmp_path):
    inp = _write(tmp_path, [_row("c1"), _row("c2")])
    old = tmp_path / "old.html"; new = tmp_path / "new.html"
    assert bt.main([str(inp), "-o", str(old)]) == 0
    assert bt.main([str(inp), "-o", str(new), "--manual-intent", "--slot", "intent"]) == 0
    a, b = old.read_text(), new.read_text()
    key = lambda h: re.search(r"const STORE='([^']+)'", h).group(1)
    assert key(a) == key(b)  # labels already saved in this browser survive the UI swap
    assert "System A" not in b and "System B" not in b and 'id="choices"' not in b
    assert "advanceTimer=setTimeout" in b and "L.winner='neither'" in b
    assert base64.b64decode(_embedded(a, "csvdata")) == base64.b64decode(_embedded(b, "csvdata"))
    assert bt.main([str(inp), "-o", str(tmp_path / "bad.html"), "--manual-intent"]) == 0
    import pytest
    with pytest.raises(SystemExit):
        bt.main([str(_write(tmp_path, [_row("c1"), _row("s1", slot="scope")], "mix.csv")),
                 "-o", str(tmp_path / "bad2.html"), "--manual-intent"])
