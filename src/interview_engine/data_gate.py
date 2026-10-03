"""Code-owned gate for the "data" coverage area.

The interviewer self-reports coverage. For "data" that is not enough: a
session closed the area while the founder had named song/lesson content and
never said where it comes from (cef8ed07: SN-9 content from Duolingo Music,
SN-12 popular unlicensed songs, SN-6 no external sources, SN-10 self-contained;
"Legal/IP provenance of lesson content ... source unspecified" filed as an
open question while the area closed).

Rule: when the needs, the transcript, or an open/deferred question mention
content nouns, "data" stays THIN until the turn reports a data_source plus a
source_quote that is a substring of the founder's latest turn. Pure functions,
no model calls.
"""
from __future__ import annotations

import re

DATA_SOURCES = ("founder_supplied", "user_entered", "imported", "suggested_then_approved", "none_needed")

CONTENT_NOUNS = ("song", "lesson", "catalog", "library", "content", "data")
_NOUN_RE = re.compile(r"\b(?:" + "|".join(CONTENT_NOUNS) + r")s?\b", re.IGNORECASE)
_PROVENANCE_RE = re.compile(r"\b(?:source|sources|sourced|provenance|origin|licen[sc]\w*|where .{0,40}comes? from)\b",
                            re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().casefold()


def mentions_content(*texts: str) -> bool:
    return any(_NOUN_RE.search(t or "") for t in texts)


def open_question_blocks_data(questions: list[str]) -> bool:
    """An open or deferred question about content/source/provenance keeps data THIN."""
    return any(_NOUN_RE.search(q or "") and _PROVENANCE_RE.search(q or "") for q in questions or [])


def quote_is_verified(source_quote: str | None, founder_latest: str) -> bool:
    """The quote must be non-trivial and a substring of the founder's latest turn."""
    q = _norm(source_quote or "")
    return len(q) >= 3 and q in _norm(founder_latest)


def data_source_verified(parsed: dict, founder_latest: str) -> bool:
    return parsed.get("data_source") in DATA_SOURCES and quote_is_verified(parsed.get("source_quote"), founder_latest)


def gate_data(coverage_before: dict[str, str], coverage_after: dict[str, str], parsed: dict,
              founder_latest: str, context_texts: list[str], open_questions: list[str]) -> dict[str, str]:
    """Return coverage with "data" held THIN unless provenance is code-verified.

    Only downgrades what this turn newly marked COVERED; never touches skipped,
    delegated, or deferred, and never un-covers an area that was already covered
    before this turn."""
    out = dict(coverage_after)
    if out.get("data") != "covered" or coverage_before.get("data") == "covered":
        return out
    triggered = mentions_content(*context_texts) or open_question_blocks_data(open_questions)
    if not triggered:
        return out
    if open_question_blocks_data(open_questions) or not data_source_verified(parsed, founder_latest):
        out["data"] = "thin"
    return out
