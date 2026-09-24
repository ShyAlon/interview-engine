import importlib.util, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_whatif_gate_on_synthetic_replay(tmp_path):
    bench = tmp_path / "b.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts/build-intent-benchmark.py"), str(ROOT / "benchmarks/intent-pool-v2.json"),
                    "-o", str(bench)], check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("w", ROOT / "scripts/whatif-relevance-gate.py")
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    rows = w.load_truth(bench, json.loads((ROOT / "benchmarks/intent-pool-v2.json").read_text()))
    run = tmp_path / "runs"; run.mkdir()
    with (run / "replay-relevance-x.jsonl").open("w") as f:
        for cid, r in rows.items():  # judge: always partial; covers_target right on truth
            v = {"relevance": "partial", f"covers_{r['area']}": 0.0 if r["truth"] == "dodged" else 0.9}
            f.write(json.dumps({"slot": "coverage", "replay": {"case_id": cid}, "jev": {"status": "ok", "verdict": v}}) + "\n")
    out = tmp_path / "w.md"
    assert w.main([str(bench), str(run), "-o", str(out)]) == 0
    text = out.read_text()
    assert "| 3 dodged recall | 0.00 | 1.00 |" in text and "| 5 partials kept | 12/12 | 12/12 |" in text
