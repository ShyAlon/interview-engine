"""2026-09-23 merge items: Jev owns scope (with the field-consistency rule),
the intent re-ask guard covers baseline skip as well as dismiss, and the
defer label is one taxonomy shared by engine, scorer and tool. Hermetic."""
import importlib.util, json
from pathlib import Path
from interview_engine.categorization import (ChoiceQuestion, ClassificationAnswer, ClassificationRequest, INTENT_LABELS,
                                             ProviderResult, classified_intent, guarded_intent)
from interview_engine.categorization_shadow import ShadowCategorizer
from interview_engine.coverage import DEFERRED, founder_steer, initial_coverage, next_area, progress_line, status_block
from interview_engine.intent_taxonomy import COMPATIBLE_GROUPS, INTENT_ORDER, compatible, taxonomy_json
from interview_engine.rendering import render_coverage_markdown
from interview_engine.scope_gate import reconcile_scope

ROOT = Path(__file__).resolve().parents[1]


class Cat:
    def __init__(self, result=None, error=None): self.result, self.error = result, error
    def active(self, slot): return True
    def classify(self, request):
        if self.error: raise self.error
        return self.result


def scope_req():
    return ClassificationRequest("scope", "idea", {
        "accepted": ChoiceQuestion("x", {"accepted": "a", "rejected": "r"}),
        "archetype": ChoiceQuestion("y", {"saas_crud": "s", "not_web_app": "n"})})


def scope_result(provider, accepted, archetype):
    return ProviderResult({"accepted": ClassificationAnswer(accepted, .8), "archetype": ClassificationAnswer(archetype, .8)},
                          provider, provider + "-model", 0, 0, .1, None)


# ---------------------------------------------------------------- scope
def test_jev_is_authoritative_for_scope_including_rejections(tmp_path):
    base = scope_result("llm", "accepted", "saas_crud"); jev = scope_result("typesafe", "rejected", "not_web_app")
    cat = ShadowCategorizer(Cat(base), Cat(jev), tmp_path / "s.jsonl", authoritative=("scope",))
    assert cat.classify(scope_req()) is jev
    row = json.loads((tmp_path / "s.jsonl").read_text())
    assert row["authority"] == "jev" and row["jev_fallback_count"] == 0


def test_baseline_is_fallback_on_jev_error_or_schema_failure(tmp_path):
    base = scope_result("llm", "accepted", "saas_crud")
    cat = ShadowCategorizer(Cat(base), Cat(error=TimeoutError("down")), tmp_path / "a.jsonl", authoritative=("scope",))
    assert cat.classify(scope_req()) is base
    bad = ProviderResult({"accepted": ClassificationAnswer("maybe", .9)}, "typesafe", "m", 0, 0, .1, None)
    cat = ShadowCategorizer(Cat(base), Cat(bad), tmp_path / "b.jsonl", authoritative=("scope",))
    assert cat.classify(scope_req()) is base
    row = json.loads((tmp_path / "b.jsonl").read_text())
    assert row["authority"] == "baseline" and row["jev_fallback_count"] == 1


def test_non_authoritative_slots_still_return_baseline(tmp_path):
    base = ProviderResult({"intent": ClassificationAnswer("skip", .95)}, "llm", "m", 0, 0, .1, None)
    jev = ProviderResult({"intent": ClassificationAnswer("answer", .95)}, "typesafe", "m", 0, 0, .1, None)
    cat = ShadowCategorizer(Cat(base), Cat(jev), tmp_path / "i.jsonl", authoritative=("scope",))
    req = ClassificationRequest("intent", "Q", {"intent": ChoiceQuestion("c", dict(INTENT_LABELS))})
    assert cat.classify(req) is base and cat.last_jev["intent"] is jev


def test_rejected_with_web_archetype_is_treated_as_accepted():
    assert reconcile_scope("rejected", "ecommerce") == ("accepted", True)
    assert reconcile_scope("rejected", "content_site") == ("accepted", True)
    assert reconcile_scope("rejected", "not_web_app") == ("rejected", False)
    assert reconcile_scope("accepted", "not_web_app") == ("accepted", False)


# ---------------------------------------------------------------- guard
def test_guard_fires_once_on_baseline_dismiss_or_skip_vs_jev_answer():
    assert guarded_intent("dismiss", "answer", False) == ("answer", "dismiss")
    assert guarded_intent("skip", "answer", False) == ("answer", "skip")
    assert guarded_intent("skip", "answer", True) == ("skip", None)       # second time: baseline wins
    assert guarded_intent("skip", "skip", False) == ("skip", None)
    assert guarded_intent("enough", "answer", False) == ("enough", None)  # enough is not guarded
    assert guarded_intent("delegate", "answer", False) == ("delegate", None)
    assert guarded_intent("skip", None, False) == ("skip", None)          # no Jev verdict: baseline


# ---------------------------------------------------------------- defer
def test_defer_is_in_the_single_taxonomy_and_compatible_with_nothing():
    assert "defer" in INTENT_ORDER and list(INTENT_LABELS) == list(INTENT_ORDER)
    assert compatible("answer", ["delegate"]) and compatible("delegate", ["answer"])
    assert not compatible("defer", ["answer"]) and not compatible("defer", ["delegate"])
    assert not compatible("answer", ["defer"]) and not compatible("delegate", ["defer"])
    assert json.loads(taxonomy_json())["order"] == list(INTENT_ORDER)


def test_scorer_reads_labels_from_the_taxonomy():
    spec = importlib.util.spec_from_file_location("score", ROOT / "scripts" / "score-rubric-variants.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert tuple(mod.INTENT_LABELS) == INTENT_ORDER
    assert mod.is_correct("defer", ["defer"], "intent") and not mod.is_correct("delegate", ["defer"], "intent")
    assert mod.is_correct("answer", ["delegate"], "intent")


def test_low_confidence_defer_falls_back_to_answer():
    assert classified_intent({"intent": ClassificationAnswer("defer", .5)}) == "answer"
    assert classified_intent({"intent": ClassificationAnswer("defer", .95)}) == "defer"


def test_deterministic_postponement_is_defer_not_delegate():
    assert founder_steer("I need to ask my partner, I'll get back to you.") == "defer"
    assert founder_steer("Let me check with our accountant.") == "defer"
    assert founder_steer("You decide.") == "delegate"
    assert founder_steer("Admins approve refunds over $50.") is None
    long = ("Parents and kids use it daily, admins approve every payout, reminders go out at 8am each morning "
            "and payouts happen weekly on Fridays. I'll check with legal on retention.")
    assert founder_steer(long) is None


def test_deferred_area_is_open_not_covered_and_the_interview_moves_on():
    cov = initial_coverage(); first = next_area(cov); cov[first] = DEFERRED
    assert next_area(cov) != first
    assert "deferred by founder" in progress_line(cov) and "coverage 0/" in progress_line(cov)
    assert "DEFERRED BY FOUNDER" in status_block(cov)
    assert "Deferred by you - still open" in render_coverage_markdown(cov)
