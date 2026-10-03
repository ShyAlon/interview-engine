from interview_engine.distill_qa import coerce_deferred

FOUNDER = {0: "We need a catalog.", 2: "Community features come in the future, not now."}


def n(**kw):
    return {"need_id": "SN-1", "statement": "x", "kind": "functional", "source_turns": [2], **kw}


def test_default_false_when_absent():
    assert coerce_deferred(n(), FOUNDER)["deferred"] is False


def test_true_kept_with_founder_postponing_wording():
    assert coerce_deferred(n(deferred=True), FOUNDER)["deferred"] is True
    assert coerce_deferred(n(deferred="true"), FOUNDER)["deferred"] is True


def test_true_dropped_without_founder_wording():
    assert coerce_deferred(n(deferred=True, source_turns=[0]), FOUNDER)["deferred"] is False
    assert coerce_deferred(n(deferred=True, source_turns=[]), FOUNDER)["deferred"] is False


def test_garbage_is_false_and_source_turns_unchanged():
    out = coerce_deferred(n(deferred="maybe"), FOUNDER)
    assert out["deferred"] is False and out["source_turns"] == [2]
