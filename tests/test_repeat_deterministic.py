"""Deterministic repeat check (no model): prior QUESTIONS only, config threshold, no filler."""
from interview_engine import question_config
from interview_engine.coverage import (_FALLBACK_QUESTIONS, _LAST_RESORT_QUESTION, best_prior_similarity,
                                       question_similarity, unique_question)

T1 = "When a 7-year-old opens the app, what does a session look like - do they pick a lesson and play through it, and what happens when they finish or get something wrong?"
T2 = "Who else besides the 7-year-old uses this - a parent or teacher setting it up or checking progress - and where do they use it?"
T3 = "Does the app need to exchange data with any existing system or service (e.g. GitHub, a music library, analytics), or is it fully self-contained?"
T4 = "Does the app need to exchange data with any existing system or service (e.g. GitHub Pages hosting, analytics, audio libraries), or is it fully self-contained with none?"
HIST = [{"area": "goal", "question": T1}, {"area": "users", "question": T2}]


def test_terminology_maps_synonyms_and_plurals():
    assert question_similarity("Which interactive features will children use?",
                               "What design elements will children use?") >= .45


def test_zip_turn3_integrations_question_passes_unchanged():
    # Shy's session 7315ff94: the closing question was replaced by the filler. No longer.
    assert unique_question(T3, HIST, "integrations") == T3


def test_zip_turn4_is_a_repeat_of_turn3_and_falls_to_area_fallback_not_filler():
    hist = HIST + [{"area": "integrations", "question": T3}]
    out = unique_question(T4, hist, "integrations")
    assert out != _LAST_RESORT_QUESTION
    assert out == _FALLBACK_QUESTIONS["integrations"]


def test_double_repeat_keeps_the_original_question_not_the_filler():
    hist = HIST + [{"area": "integrations", "question": T3},
                   {"area": "integrations", "question": _FALLBACK_QUESTIONS["integrations"]}]
    out = unique_question(T4, hist, "integrations")
    assert out == T4 and out != _LAST_RESORT_QUESTION


def test_final_text_is_never_a_verbatim_repeat():
    hist = HIST + [{"question": T3}, {"question": _FALLBACK_QUESTIONS["integrations"]}]
    out = unique_question(T3, hist, "integrations")
    assert out not in [h["question"] for h in hist]
    # even with the reframe taken, the original is kept rather than the filler
    reframe = out
    hist2 = hist + [{"question": reframe}]
    assert unique_question(T3, hist2, "integrations") != _LAST_RESORT_QUESTION


def test_only_prior_questions_are_compared_never_answers():
    # a candidate that echoes the founder's answer is not a repeat: answers are not an input at all
    import inspect
    assert list(inspect.signature(unique_question).parameters) == ["question", "prior_questions", "target", "threshold"]
    assert unique_question("Which kids use a tablet at home?", HIST, "users") == "Which kids use a tablet at home?"


def test_threshold_is_config_driven_and_validated():
    assert question_config.defaults()["question_repeat"]["similarity_threshold_pct"] == 45
    for bad in (5, 101, "45", True, 45.5):
        try:
            question_config.validate({"question_repeat": {"similarity_threshold_pct": bad}})
        except question_config.ConfigError:
            continue
        raise AssertionError(bad)
    assert question_config.validate({"question_repeat": {"similarity_threshold_pct": 80}})["question_repeat"]["similarity_threshold_pct"] == 80
    a, b = "Who uses the app at school?", "Who uses the app at home?"
    sim = question_similarity(a, b)
    assert unique_question(b, [{"question": a}], "users", sim - .01) != b      # stricter bar: repeat
    assert unique_question(b, [{"question": a}], "users", sim + .01) == b      # looser bar: new


def test_best_prior_similarity_names_the_match():
    sim, match = best_prior_similarity(T4, [T1, T3])
    assert match == T3 and sim > .45
