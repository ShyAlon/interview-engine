"""D12 3.3.1: fallback questions are self-contained (session 42b71e2e Q5 read
'What information does that use need...' with no antecedent)."""
import re

from interview_engine.coverage import (_FALLBACK_QUESTIONS, _LAST_RESORT_QUESTION, AREAS as _AREA_ROWS, reask_question,
                                       unique_question)

AREAS = [a[0] for a in _AREA_ROWS]
_DANGLING = re.compile(r"\b(?:that use|it|this|those|these|such)\b", re.IGNORECASE)


def test_every_area_has_a_fallback():
    assert set(AREAS) <= set(_FALLBACK_QUESTIONS)


def test_no_fallback_has_a_dangling_reference():
    for area, q in {**_FALLBACK_QUESTIONS, "last_resort": _LAST_RESORT_QUESTION}.items():
        assert not _DANGLING.search(q), (area, q)


def test_data_fallback_names_source_and_who_provides_it():
    q = _FALLBACK_QUESTIONS["data"]
    assert "come from" in q and "provides or maintains the information" in q and "responsible" not in q


def test_reask_is_self_contained_and_prefixed():
    for area in AREAS:
        q = reask_question(area)
        assert q.startswith("Sorry, that question was not clear.")
        assert not _DANGLING.search(q), q
    assert reask_question(None).endswith(_LAST_RESORT_QUESTION)


def test_final_reframe_has_no_dangling_this():
    q = unique_question("x", [{"question": "x"}, {"question": _FALLBACK_QUESTIONS["data"]}], "data")
    assert not _DANGLING.search(q)
