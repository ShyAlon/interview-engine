"""Data area stays THIN until a code-verified provenance field is present."""
from interview_engine.coverage import turn_output_schema
from interview_engine.data_gate import (data_source_verified, gate_data, mentions_content,
                                        open_question_blocks_data, quote_is_verified)

CUR = {"data": "covered"}
BEFORE = {"data": "uncovered"}
ANSWER = "The songs come from Duolingo Music, we just bundle them."
CTX = ["Learners play popular songs", ANSWER]


def test_schema_carries_optional_provenance_fields():
    props = turn_output_schema()["properties"]
    assert "none" in props["data_source"]["enum"] and "source_quote" in props
    assert "data_source" not in turn_output_schema()["required"]


def test_quote_must_be_substring_of_latest_founder_turn():
    assert quote_is_verified("come from  duolingo MUSIC", ANSWER)
    assert not quote_is_verified("licensed by Sony", ANSWER)
    assert not quote_is_verified("", ANSWER) and not quote_is_verified(None, ANSWER)


def test_content_nouns_without_verified_source_stay_thin():
    out = gate_data(BEFORE, CUR, {"covered": ["data"]}, ANSWER, CTX, [])
    assert out["data"] == "thin"


def test_fabricated_quote_stays_thin():
    p = {"data_source": "founder_supplied", "source_quote": "we author every lesson ourselves"}
    assert gate_data(BEFORE, CUR, p, ANSWER, CTX, [])["data"] == "thin"


def test_verified_quote_lets_data_close():
    p = {"data_source": "imported", "source_quote": "come from Duolingo Music"}
    assert data_source_verified(p, ANSWER)
    assert gate_data(BEFORE, CUR, p, ANSWER, CTX, [])["data"] == "covered"


def test_invalid_enum_value_does_not_verify():
    assert not data_source_verified({"data_source": "none", "source_quote": "Duolingo"}, ANSWER)


def test_no_content_nouns_means_gate_is_inactive():
    assert gate_data(BEFORE, CUR, {"covered": ["data"]}, "Users are managers.", ["Users are managers."], [])["data"] == "covered"


def test_open_provenance_question_keeps_data_thin_even_with_quote():
    q = ["Legal/IP provenance of lesson content: the source of that content is unspecified"]
    assert open_question_blocks_data(q) and not open_question_blocks_data(["Which colour theme?"])
    p = {"data_source": "imported", "source_quote": "come from Duolingo Music"}
    assert gate_data(BEFORE, CUR, p, ANSWER, CTX, q)["data"] == "thin"


def test_never_uncovers_previously_covered_or_touches_other_areas():
    assert gate_data({"data": "covered"}, {"data": "covered"}, {}, ANSWER, CTX, [])["data"] == "covered"
    out = gate_data(BEFORE, {"data": "skipped", "goal": "covered"}, {}, ANSWER, CTX, [])
    assert out == {"data": "skipped", "goal": "covered"}


def test_mentions_content_nouns():
    for w in ("song", "lesson", "catalog", "library", "content", "data"):
        assert mentions_content(f"our {w}s matter")
    assert not mentions_content("our users log in")
