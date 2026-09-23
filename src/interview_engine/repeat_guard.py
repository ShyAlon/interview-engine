"""Deterministic repeated-non-answer guard (2026-09-23 smoke finding).

The smoke run pasted one canned answer at several consecutive distinct
questions. Per-turn classifiers judge each answer on its own merits, so the
interview accepted the same non-responsive text as 'answer' turn after turn;
only the interviewer model itself eventually noticed and closed out - late,
and at prompt level rather than in code. Doctrine: a repeated non-answer to a
new question is a dodge. The first repeat freezes coverage and the area is
re-asked; the next repeat trips the existing two-attempts closeout, which
parks the area as a visible delegation. Detection is deterministic -
near-duplicate founder text across consecutive turns. No model call."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

# Short repeats ("yes", "no", "ok", "I don't know") are conversation, not
# dodges; a dodge repeats substantive text.
_MIN_CHARS = 20
_MIN_CONTENT_WORDS = 3

_RATIO_THRESHOLD = 0.85   # SequenceMatcher on normalized text
_JACCARD_THRESHOLD = 0.8  # content-word overlap, order-insensitive

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "if", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "so", "that", "the", "their", "them", "they", "this",
    "to", "us", "we", "with", "you", "your",
})


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").casefold()))


def _content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _normalize(text).split() if w not in _STOP)


def answers_match(current: str, previous: str) -> bool:
    """True when two founder answers are the same answer: identical after
    normalization, or near-duplicate by sequence ratio or content-word overlap
    (a reordered or lightly edited re-paste is still the same answer)."""
    a, b = _normalize(current), _normalize(previous)
    if not a or not b:
        return False
    if len(a) < _MIN_CHARS or len(b) < _MIN_CHARS:
        return False
    wa, wb = _content_words(current), _content_words(previous)
    if len(wa) < _MIN_CONTENT_WORDS or len(wb) < _MIN_CONTENT_WORDS:
        return False
    if a == b:
        return True
    if SequenceMatcher(None, a, b).ratio() >= _RATIO_THRESHOLD:
        return True
    return len(wa & wb) / len(wa | wb) >= _JACCARD_THRESHOLD


def founder_turns(messages) -> list[tuple[str, str]]:
    """(question, answer) pairs from the transcript, oldest first. The intake
    message is excluded: it answers no interviewer question."""
    turns: list[tuple[str, str]] = []
    last_question = ""
    for m in messages:
        mtype = getattr(m, "type", "")
        content = getattr(m, "content", "") or ""
        if mtype == "ai":
            last_question = content
        elif mtype == "human" and last_question:
            turns.append((last_question, content))
    return turns


def repeat_streak(messages) -> int:
    """How many times in a row the founder's latest answers repeat the one
    before. 0 = the latest answer is new; 1 = it repeats once (re-ask);
    2+ = it has repeated at least twice (escalate)."""
    turns = founder_turns(messages)
    streak = 0
    for i in range(len(turns) - 1, 0, -1):
        if answers_match(turns[i][1], turns[i - 1][1]):
            streak += 1
        else:
            break
    return streak
