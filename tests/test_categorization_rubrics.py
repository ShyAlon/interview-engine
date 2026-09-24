"""Rubric variants: intent-v1-133 stays byte-identical to the labeled rubric, v2 variants are versioned, and the
boundary cases from the rare-class design are encoded in the rubric text or,
for decomposed variants, in code-owned precedence. Hermetic: no Jev calls."""
import csv, json, subprocess, sys
from pathlib import Path
import pytest
from interview_engine.categorization import (ChoiceQuestion, ClassificationAnswer, INTENT_LABELS, NoulQuestion,
                                             coverage_request, intent_request)
from interview_engine.categorization_rubrics import (INTENT_VARIANTS, RELEVANCE_VARIANTS, V1_RUBRIC_VERSION, V1_DEFER_RUBRIC_VERSION,
                                                     configured_variant, finalize_decomposed_intent,
                                                     finalize_decomposed_relevance, get_variant)
from interview_engine.categorization_shadow import input_package

ROOT = Path(__file__).resolve().parents[1]
V1_REL = {"addressed": "directly and substantively answers it", "partial": "answers only part or remains vague",
          "dodged": "does not answer, changes subject, or refuses without delegating"}


@pytest.fixture(autouse=True)
def _no_env_variant(monkeypatch):
    monkeypatch.delenv("INTERVIEW_ENGINE_RUBRIC_INTENT", raising=False)
    monkeypatch.delenv("INTERVIEW_ENGINE_RUBRIC_RELEVANCE", raising=False)
    import interview_engine.model_config as mc
    monkeypatch.setattr(mc, "_dotenv", lambda: {})


V1_INTENT_133 = {"answer": "ordinary product answer", "delegate": "founder delegates the current detail",
                 "dismiss": "founder closes the current topic/area", "skip": "founder explicitly skips the current area",
                 "enough": "founder explicitly ends the whole interview"}


V1_1_INTENT = {**{k: V1_INTENT_133[k] for k in ("answer", "delegate")},
               "defer": "founder postpones the current question to answer later (not decided, not delegated)",
               **{k: V1_INTENT_133[k] for k in ("dismiss", "skip", "enough")}}
V1_2_INTENT = {"answer": "Founder gives product information, even if it doesn't directly address the question asked",
               "delegate": "Founder wants the interviewer to decide", "defer": "Founder wants to answer later",
               "dismiss": "Founder doesn't want to tackle this topic", "skip": "Founder wants to move on",
               "enough": "Founder wants to end the whole interview"}


V1_3_INTENT = {k: v for k, v in V1_2_INTENT.items() if k != "dismiss"}
V1_3_INTENT["skip"] = "Founder wants to move on, including not wanting to tackle this topic"


def test_default_intent_is_v1_3_from_the_taxonomy_and_relevance_v1_unchanged():
    from interview_engine.categorization_rubrics import V1_3_RUBRIC_VERSION
    i = intent_request("Q", "A"); c = coverage_request("Q", "A")
    assert i.questions["intent"] == ChoiceQuestion("Classify the founder's control intent", INTENT_LABELS)
    assert dict(INTENT_LABELS) == V1_3_INTENT and list(INTENT_LABELS) == list(V1_3_INTENT)
    assert "dismiss" not in INTENT_LABELS
    assert c.questions["relevance"] == ChoiceQuestion("Did the answer address the question?", V1_REL)
    assert input_package(i)["rubric_version"] == V1_3_RUBRIC_VERSION == "categorization-request-v1.3"
    assert input_package(c)["rubric_version"] == V1_RUBRIC_VERSION


def test_intent_v1_2_stays_replayable_frozen():
    from interview_engine.categorization_rubrics import V1_2_RUBRIC_VERSION
    i = intent_request("Q", "A", "intent-v1.2")
    assert i.questions["intent"] == ChoiceQuestion("Classify the founder's control intent", V1_2_INTENT)
    assert list(i.questions["intent"].criteria) == list(V1_2_INTENT)
    assert input_package(i)["rubric_version"] == V1_2_RUBRIC_VERSION == "categorization-request-v1.2"


def test_intent_v1_1_stays_replayable_frozen():
    i = intent_request("Q", "A", "intent-v1.1")
    assert i.questions["intent"] == ChoiceQuestion("Classify the founder's control intent", V1_1_INTENT)
    assert input_package(i)["rubric_version"] == V1_DEFER_RUBRIC_VERSION


def test_env_can_select_frozen_intent_versions(monkeypatch):
    for vid, ver in (("intent-v1.1", V1_DEFER_RUBRIC_VERSION), ("intent-v1-133", V1_RUBRIC_VERSION)):
        monkeypatch.setenv("INTERVIEW_ENGINE_RUBRIC_INTENT", vid)
        assert intent_request("Q", "A").rubric_version == ver


def test_intent_v1_133_is_the_labeled_rubric_byte_identical():
    i = intent_request("Q", "A", "intent-v1-133")
    assert i.questions["intent"] == ChoiceQuestion("Classify the founder's control intent", V1_INTENT_133)
    assert list(i.questions["intent"].criteria) == list(V1_INTENT_133)
    assert input_package(i)["rubric_version"] == V1_RUBRIC_VERSION


def test_every_v2_variant_has_its_own_rubric_version_and_ref():
    refs = set()
    for v in [*INTENT_VARIANTS.values(), *RELEVANCE_VARIANTS.values()]:
        req = intent_request("Q", "A", v.id) if v.slot_key == "intent" else coverage_request("Q", "A", None, v.id)
        pkg = input_package(req)
        assert pkg["rubric_version"] == v.rubric_version
        assert (pkg["rubric_version"] == V1_RUBRIC_VERSION) == (v.id in ("relevance-v1", "intent-v1-133"))
        refs.add(pkg["rubric_ref"])
    assert len(refs) == len(INTENT_VARIANTS) + len(RELEVANCE_VARIANTS)


def test_choice_variants_keep_the_canonical_label_sets():
    for v in INTENT_VARIANTS.values():
        if v.finalize is None and v.id not in ("intent-v1-133", "intent-v1.1", "intent-v1.2"): assert set(v.questions["intent"].criteria) == set(INTENT_LABELS)
    for v in RELEVANCE_VARIANTS.values():
        if v.finalize is None: assert set(v.questions["relevance"].criteria) == set(V1_REL)


def test_coverage_variants_only_swap_relevance():
    base = coverage_request("Q", "A"); v2 = coverage_request("Q", "A", None, "relevance-v2b-strict")
    assert {k: q for k, q in base.questions.items() if k.startswith("covers_")} == \
           {k: q for k, q in v2.questions.items() if k.startswith("covers_")}


def test_env_selects_production_variant_and_rejects_replay_only(monkeypatch):
    monkeypatch.setenv("INTERVIEW_ENGINE_RUBRIC_INTENT", "intent-v2c-offtopic")
    assert intent_request("Q", "A").rubric_version.endswith("intent-v2c-offtopic")
    monkeypatch.setenv("INTERVIEW_ENGINE_RUBRIC_INTENT", "intent-v2d-decomposed")
    with pytest.raises(ValueError, match="replay-only"): configured_variant("intent")
    monkeypatch.setenv("INTERVIEW_ENGINE_RUBRIC_INTENT", "relevance-v1")
    with pytest.raises(ValueError, match="relevance variant"): configured_variant("intent")
    with pytest.raises(ValueError, match="unknown rubric variant"): get_variant("intent-v9")


# --- boundary cases from the rare-class supplement design ---------------------
def _txt(v, label): return v.questions["intent"].criteria[label].lower()


@pytest.mark.parametrize("vid", ["intent-v2b-examples", "intent-v2c-offtopic"])
def test_example_rubrics_encode_the_delegate_and_product_rule_boundaries(vid):
    v = INTENT_VARIANTS[vid]
    # "You decide." / "Whatever you think is best" -> delegate
    assert "your call" in _txt(v, "delegate") and "whatever you think" in _txt(v, "delegate")
    # "do whatever calendar apps do" (delegating a sub-detail to convention) -> delegate
    assert "the way most" in _txt(v, "delegate")
    # "the app should decide" is a product rule, not a delegation
    assert "product decides" in _txt(v, "delegate") and "not a delegation" in _txt(v, "answer")


def test_offtopic_rule_is_only_in_the_offtopic_variant():
    # off-topic grocery-ordering reply -> intent answer; relevance judges it
    assert "off-topic" in INTENT_VARIANTS["intent-v2c-offtopic"].questions["intent"].instructions.lower()
    assert "off-topic" in _txt(INTENT_VARIANTS["intent-v2c-offtopic"], "answer")
    for vid in ("intent-v1", "intent-v2a-defs", "intent-v2b-examples"):
        assert "off-topic" not in INTENT_VARIANTS[vid].questions["intent"].instructions.lower()
    for vid in ("relevance-v2a-anchored", "relevance-v2b-strict"):
        assert "unrelated to the question" in RELEVANCE_VARIANTS[vid].questions["relevance"].criteria["dodged"]


def test_rubric_examples_do_not_copy_supplement_inputs_verbatim():
    held_out = ["you decide.", "whatever you think is best", "do whatever calendar apps do",
                "that's for the security people", "i'll ask my partner", "can we move on?", "rotate fairly"]
    for v in [*INTENT_VARIANTS.values(), *RELEVANCE_VARIANTS.values()]:
        blob = json.dumps(input_package(intent_request("", "", v.id) if v.slot_key == "intent"
                                        else coverage_request("", "", None, v.id))["questions"]).lower()
        assert not [h for h in held_out if h in blob], v.id


def _sig(**kw): return {k: ClassificationAnswer(v) for k, v in kw.items()}
BASE = dict(intent_ends_interview=0.0, intent_closes_topic=0.0, intent_declines=0.0, intent_hands_choice=0.0, intent_gives_content=0.0)


@pytest.mark.parametrize("signals,label", [
    (dict(intent_hands_choice=.95), "delegate"),                                # "You decide."
    (dict(intent_hands_choice=.8, intent_gives_content=.3), "delegate"),        # "whatever you think is best"
    (dict(intent_hands_choice=.7, intent_gives_content=.6), "delegate"),        # "do whatever calendar apps do"
    (dict(intent_gives_content=.9, intent_hands_choice=.2), "answer"),          # "the app should decide - rotate fairly"
    (dict(), "answer"),                                                         # off-topic grocery request, no control move
    (dict(intent_closes_topic=.9), "skip"),                                     # "Can we move on?" (v1.2: move on = skip)
    (dict(intent_declines=.8), "skip"),                                      # v1.2: won't tackle the topic = dismiss
    (dict(intent_declines=.6, intent_gives_content=.9), "answer"),              # content outweighs a soft decline
    (dict(intent_ends_interview=.9, intent_hands_choice=.9), "enough"),         # ending the interview wins
])
def test_decomposed_intent_precedence(signals, label):
    out = finalize_decomposed_intent(_sig(**{**BASE, **signals}))
    assert out["intent"].value == label and 0 <= out["intent"].confidence <= 1


@pytest.mark.parametrize("signals,label", [
    (dict(rel_on_question=.9, rel_concrete=.9, rel_all_parts=.8, rel_generic=.1), "addressed"),
    (dict(rel_on_question=.9, rel_concrete=.9, rel_all_parts=.3, rel_generic=.1), "partial"),
    (dict(rel_on_question=.8, rel_concrete=.2, rel_all_parts=.2, rel_generic=.1), "partial"),
    (dict(rel_on_question=.2, rel_concrete=.9, rel_all_parts=.1, rel_generic=.1), "dodged"),   # facts, wrong question
    (dict(rel_on_question=.7, rel_concrete=.7, rel_all_parts=.7, rel_generic=.8), "dodged"),   # canned text
])
def test_decomposed_relevance_mapping(signals, label):
    assert finalize_decomposed_relevance(_sig(**signals))["relevance"].value == label


# --- replay + scoring scripts ---------------------------------------------------
def _write_labeled(tmp_path):
    rows = [
        {"case_id": "c1", "input_ref": "sha256:1", "slot": "intent", "session_id": "s", "input_state": "Question: Q\nFounder answer: Your call",
         "input_context": "{}", "rubric": json.dumps({"intent": {"instructions": "x", "criteria": INTENT_LABELS, "multiple": False}}),
         "rubric_version": "recovered-v1", "system_a_verdict": '{"intent": "delegate"}', "system_a_confidence": '{"intent": 0.7}',
         "system_b_verdict": '{"intent": "answer"}', "system_b_confidence": '{"intent": 0.61}', "sample_reason": "disagreement",
         "human_verdict": "", "winner": "A", "reviewer_id": "", "review_notes": ""},
        {"case_id": "c2", "input_ref": "sha256:2", "slot": "coverage", "session_id": "s", "input_state": "Question: Q\nAnswer: A",
         "input_context": "{}", "rubric": json.dumps({"relevance": {"instructions": "x", "criteria": V1_REL, "multiple": False},
                                                     "covers_goal": {"instructions": "goal"}}),
         "rubric_version": "recovered-v1", "system_a_verdict": '{"relevance": "partial"}', "system_a_confidence": "{}",
         "system_b_verdict": '{"relevance": "dodged"}', "system_b_confidence": "{}", "sample_reason": "disagreement",
         "human_verdict": "", "winner": "B", "reviewer_id": "", "review_notes": ""},
    ]
    p = tmp_path / "labeled.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return p


def test_replay_dry_run_builds_every_variant_request(tmp_path):
    p = _write_labeled(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/replay-jev-rubric-variants.py"), str(p), "-o", str(tmp_path / "out"), "--dry-run"],
                       capture_output=True, text=True, env={"PYTHONPATH": ":".join(sys.path), "PATH": ""})
    assert r.returncode == 0, r.stderr
    assert "cases: 2" in r.stdout and f"Jev calls: {len(INTENT_VARIANTS) + len(RELEVANCE_VARIANTS)}" in r.stdout


def test_replay_writes_v2_records_per_variant_with_fake_jev(tmp_path, monkeypatch):
    import importlib.util, interview_engine.categorization as cat
    from interview_engine.categorization import ProviderResult
    class FakeJev:
        name = "typesafe"
        def __init__(self, **kw): pass
        @staticmethod
        def available(): return True
        def classify(self, req):
            ans = {k: ClassificationAnswer(0.8 if isinstance(q, NoulQuestion) else next(iter(q.criteria)), .7) for k, q in req.questions.items()}
            return ProviderResult(ans, "typesafe", "jev-test", 5, 1, .01)
    monkeypatch.setattr(cat, "TypeSafeProvider", FakeJev)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    spec = importlib.util.spec_from_file_location("replay", ROOT / "scripts/replay-jev-rubric-variants.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = tmp_path / "out"; lab = _write_labeled(tmp_path)
    assert mod.main([str(lab), "-o", str(out), "--workers", "1"]) == 0
    files = sorted(x.name for x in out.glob("replay-*.jsonl"))
    assert len(files) == len(INTENT_VARIANTS) + len(RELEVANCE_VARIANTS)
    rec = json.loads((out / "replay-intent-v2d-decomposed.jsonl").read_text().splitlines()[0])
    assert rec["record_schema_version"] == 2 and rec["replay"]["case_id"] == "c1"
    assert rec["input"]["rubric_version"].endswith("intent-v2d-decomposed") and rec["jev"]["verdict"]["intent"] in INTENT_LABELS
    cov = json.loads((out / "replay-relevance-v2b-strict.jsonl").read_text().splitlines()[0])
    assert "covers_goal" in cov["input"]["questions"] and cov["input"]["state"] == "Question: Q\nAnswer: A"
    with pytest.raises(SystemExit, match="not empty"): mod.main([str(lab), "-o", str(out)])
    key = tmp_path / "key.json"
    key.write_text(json.dumps([{"case_id": "c1", "system_a": "baseline", "system_b": "jev"},
                               {"case_id": "c2", "system_a": "baseline", "system_b": "jev"}]))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/score-rubric-variants.py"), str(lab), str(key), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "| baseline (original) | 1 | 1.00" in r.stdout and "| jev-v1 (original) | 1 | 1.00" in r.stdout  # answer vs delegate: compatible
    assert "intent-v2c-offtopic" in r.stdout and "relevance-v2a-anchored" in r.stdout


def test_scorer_accepts_multilabel_intent(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("score", ROOT / "scripts/score-rubric-variants.py")
    sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    lab = _write_labeled(tmp_path)
    rows = list(csv.DictReader(lab.open()))
    rows[0]["human_verdict"] = '{"intent": ["skip", "dismiss"]}'; rows[0]["winner"] = "neither"
    with lab.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    key = tmp_path / "key.json"
    key.write_text(json.dumps([{"case_id": "c1", "system_a": "baseline", "system_b": "jev"}, {"case_id": "c2", "system_a": "baseline", "system_b": "jev"}]))
    out = tmp_path / "empty"; out.mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/score-rubric-variants.py"), str(lab), str(key), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "{'skip': 1}" in r.stdout   # v1.3: dismiss folded into skip
    assert "| baseline (original) | 1 | 0.00" in r.stdout   # delegate not in {skip, dismiss}, not compatible


def test_decomposed_intent_maps_postponement_to_defer():
    out = finalize_decomposed_intent(_sig(**{**BASE, "intent_postpones": .9, "intent_gives_content": .2}))
    assert out["intent"].value == "defer"
    out = finalize_decomposed_intent(_sig(**{**BASE, "intent_postpones": .6, "intent_gives_content": .9}))
    assert out["intent"].value == "answer"
