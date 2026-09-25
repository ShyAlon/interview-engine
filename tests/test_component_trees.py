"""Component trees (coverage-v2-trees): replay-only, v1 stays the default."""
import importlib.util, json
from pathlib import Path

import pytest

from interview_engine import component_trees as ct
from interview_engine.categorization import ClassificationAnswer
from interview_engine.categorization_rubrics import COMPONENT_VARIANTS, RELEVANCE_VARIANTS, configured_variant, get_variant
from interview_engine.coverage_taxonomy import AREA_KEYS_V2

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_tree_shape_and_rulings():
    assert list(ct.TREES) == AREA_KEYS_V2
    assert set(ct.ALL_OPTIONAL_AREAS) == {"risks", "constraints"}
    opt = {"flows.exceptions", "data.source", "integrations.pricing", "security.signin", "security.sensitive", "constraints.running_costs"}
    req = set().union(*(ct.component_ids(a, required_only=True) for a in AREA_KEYS_V2))
    assert not opt & req
    for a in AREA_KEYS_V2:  # every required component has its own follow-up
        for c in ct.TREES[a]:
            assert (not c.required) or c.followup, (a, c.key)


def test_derive_relevance():
    d = ct.derive_relevance
    assert d("goal", set()) == "dodged"
    assert d("goal", {"goal.problem"}) == "partial"
    assert d("goal", {"goal.problem", "goal.who"}) == "addressed"
    assert d("goal", {"users.personas"}) == "dodged"  # cross-area never fills
    assert d("risks", set()) == "dodged" and d("risks", {"risks.notice"}) == "addressed"  # all-optional: never partial
    assert d("scalability", set(), {"scalability"}) == "addressed"  # area skip closes the subtree
    perf = {"performance.client.moment", "performance.client.number", "performance.groomer.moment", "performance.groomer.number"}
    assert d("performance", perf) == "partial"
    assert d("performance", perf, {"performance.front_desk"}) == "addressed"  # persona branch skip
    assert d("reliability", {"reliability.allowed"}, open_components=["reliability.per_year"]) == "dodged"
    assert d("reliability", {"reliability.per_year"}, open_components=["reliability.per_year"]) == "addressed"


def test_statements_and_parse():
    st = ct.statements("security")
    assert "comp__security__groomer__see" in st and "na__security__front_desk" in st and "na__security" in st
    assert "mentioned__goal" in st and "mentioned__security" not in st
    filled, na, men = ct.parse({"comp__security__groomer__see": 1.0, "na__security": 0.9, "mentioned__goal": 1, "comp__security__signin": 0.1})
    assert filled == {"security.groomer.see"} and na == {"security"} and men == {"goal"}


def test_review_list_circles_back():
    lines = ct.review_list({"goal": {"filled": {"goal.problem"}}, "scalability": {"skipped": {"scalability": "one salon"}}})
    assert any(l.startswith("Circling back to goal.who") for l in lines)
    assert any("scalability" in l and "one salon" in l for l in lines)
    assert not any(l.startswith("Circling back to scalability.") for l in lines)


def test_variant_is_replay_only_and_v1_default(monkeypatch):
    v = COMPONENT_VARIANTS["relevance-trees-v1"]
    assert not v.production_ok and "relevance-trees-v1" not in RELEVANCE_VARIANTS
    with pytest.raises(ValueError): get_variant("relevance-trees-v1")  # production selection can't reach it
    monkeypatch.delenv("INTERVIEW_ENGINE_RUBRIC_RELEVANCE", raising=False)
    assert configured_variant("relevance").id == "relevance-v1"
    ctx = {"target": "goal", "personas": list(ct.DEFAULT_PERSONAS)}
    qs = v.build(ctx)
    assert "relevance" not in qs and "comp__goal__who" in qs
    ans = {k: ClassificationAnswer(0.0) for k in qs}; ans["comp__goal__problem"] = ClassificationAnswer(1.0)
    assert v.apply(ans, ctx)["relevance"].value == "partial"


@pytest.fixture(scope="module")
def bench():
    pool3 = json.loads((ROOT / "benchmarks" / "intent-pool-v3.json").read_text())
    pool = _load("derive-pool-v4").derive(pool3)
    assert pool == json.loads((ROOT / "benchmarks" / "intent-pool-v4.json").read_text()), "pool-v4 file is stale"
    return pool, _load("build-intent-benchmark").build_rows(pool)


def test_pool_v4_truth(bench):
    pool, rows = bench
    cov = [r for r in rows if r["slot"] == "coverage"]
    assert len(cov) == 935  # pinned Jev call count for relevance-trees-v1
    t = {r["review_notes"].split(" | ")[0]: json.loads(r["human_verdict"])["relevance"] for r in cov}
    assert t["ANS-01 x Q-goal"] == "addressed" and t["ANS-P4 x Q-goal"] == "partial"
    assert t["ANS-PERF-P x Q-performance"] == "dodged" and t["ANS-SCALE-P x Q-scalability"] == "dodged"
    assert t["ANS-08 x Q-constraints"] == "addressed" and t["ANS-P8 x Q-risks"] == "addressed"
    assert t["NA-03 x Q-scalability"] == "addressed" and t["NA-01 x Q-performance"] == "partial"
    assert t["ANS-02 x Q-performance"] == "dodged"
    assert t["ANS-REL x F-reliability.per_year"] == "dodged" and t["ANS-SCALE x F-scalability.volume"] == "addressed"


def test_perfect_replay_passes_every_gate(bench, tmp_path):
    pool, rows = bench
    csvp = tmp_path / "bench.csv"; csvp.write_text(_load("build-intent-benchmark").to_csv(rows))
    recs = []
    for r in rows:
        if r["slot"] != "coverage": continue
        hv = json.loads(r["human_verdict"]); ctx = json.loads(r["input_context"])
        v = {f"comp__{c.replace('.', '__')}": float(t) for c, t in hv["components"].items()}
        for n in ct.na_nodes(ctx["target"], ctx["personas"]): v[f"na__{n.replace('.', '__')}"] = float(n in hv["skipped"])
        v["relevance"] = hv["relevance"]
        recs.append({"slot": "coverage", "replay": {"case_id": r["case_id"]},
                     "jev": {"status": "ok", "provider": "fake", "model": "perfect", "verdict": v}})
    d = tmp_path / "out"; d.mkdir()
    (d / "replay-relevance-trees-v1.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    rep = tmp_path / "rep.md"
    _load("score-component-bench").main([str(csvp), str(d), "-o", str(rep)])
    text = rep.read_text()
    assert "model: fake/perfect" in text and text.count("PASS") == 6 and "FAIL" not in text


def test_trees_v2_gate_and_wording():
    from interview_engine.categorization_rubrics import COMPONENT_VARIANTS
    v = COMPONENT_VARIANTS["relevance-trees-v2"]
    assert not v.production_ok
    ctx = {"target": "users", "personas": list(ct.DEFAULT_PERSONAS)}
    qs = v.build(ctx)
    assert "about__users" in qs and set(k for k in qs if k.startswith("comp__")) == set(k for k in ct.statements("users") if k.startswith("comp__"))
    ans = {k: ClassificationAnswer(0.0) for k in qs}
    ans["comp__users__personas"] = ClassificationAnswer(1.0)
    assert v.apply(ans, ctx)["relevance"].value == "dodged"  # gate closed: the name-drop does not count
    ans["about__users"] = ClassificationAnswer(0.9)
    assert v.apply(ans, ctx)["relevance"].value == "partial"
    f, _, _ = ct.effective_fills("users", {"comp__users__personas": 1.0})  # no gate key (v1) = plain parse
    assert f == {"users.personas"}


def test_trees_v3_calibration_examples_and_isolation():
    v = COMPONENT_VARIANTS["relevance-trees-v3"]
    assert not v.production_ok and "relevance-trees-v3" not in RELEVANCE_VARIANTS
    with pytest.raises(ValueError): get_variant("relevance-trees-v3")
    for area in AREA_KEYS_V2:
        ctx = {"target": area, "personas": list(ct.DEFAULT_PERSONAS)}
        assert set(v.build(ctx)) == set(COMPONENT_VARIANTS["relevance-trees-v2"].build(ctx))
    flows = ct.statements_v3("flows")
    assert "even one step" in flows["about__flows"]
    assert "at least one thing" in flows["comp__flows__steps"]
    assert "at least two steps" in ct.statements_v2("flows")["comp__flows__steps"]
    assert "list of who uses the app" in ct.statements_v3("goal")["about__goal"]
    assert "owner and dog details" in ct.statements_v3("data")["comp__data__what"]
    assert "{example}" in ct.STATEMENTS_V3[("data", "what")]
    assert "name" in ct.statements_v3("integrations")["comp__integrations__systems"]
    with pytest.raises(ValueError): ct.statements_v3("data", domain="unknown")
    ctx = {"target": "flows", "personas": list(ct.DEFAULT_PERSONAS)}
    qs = v.build(ctx)
    ans = {k: ClassificationAnswer(0.0) for k in qs}
    ans["comp__flows__steps"] = ClassificationAnswer(1.0)
    assert v.apply(ans, ctx)["relevance"].value == "dodged"
    ans["about__flows"] = ClassificationAnswer(1.0)
    assert v.apply(ans, ctx)["relevance"].value == "partial"


def test_trees_v4_targeted_delta():
    v3 = COMPONENT_VARIANTS["relevance-trees-v3"]
    v4 = COMPONENT_VARIANTS["relevance-trees-v4"]
    assert not v4.production_ok
    with pytest.raises(ValueError): get_variant("relevance-trees-v4")
    for area in AREA_KEYS_V2:
        a = ct.statements_v3(area); b = ct.statements_v4(area)
        assert set(a) == set(b)
        changed = {k for k in a if a[k] != b[k]}
        expected = {"flows": {"about__flows", "comp__flows__steps"}, "done": {"comp__done__target"}}.get(area, set())
        assert changed == expected, (area, changed)
    assert "single actual booking action" in ct.statements_v4("flows")["about__flows"]
    assert "price or deposit" in ct.statements_v4("done")["comp__done__target"]
    ctx = {"target": "flows", "personas": list(ct.DEFAULT_PERSONAS)}
    assert set(v4.build(ctx)) == set(v3.build(ctx))


def test_trees_v5_targeted_delta():
    v4 = COMPONENT_VARIANTS["relevance-trees-v4"]
    v5 = COMPONENT_VARIANTS["relevance-trees-v5"]
    assert not v5.production_ok
    with pytest.raises(ValueError): get_variant("relevance-trees-v5")
    for area in AREA_KEYS_V2:
        a = ct.statements_v4(area); b = ct.statements_v5(area)
        assert set(a) == set(b)
        changed = {k for k in a if a[k] != b[k]}
        expected = {"done": {"about__done", "comp__done__measure", "comp__done__target"},
                    "users": {"about__users"}}.get(area, set())
        assert changed == expected, (area, changed)
        assert set(v4.build({"target": area})) == set(v5.build({"target": area}))
    assert "One named user role" in ct.statements_v5("users")["about__users"]
    assert "thin reply" in ct.statements_v5("done")["about__done"]
