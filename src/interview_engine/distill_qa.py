"""Deterministic post-distillation QA (ADR-0015). Prompt-level discipline failed
twice (run 2, run 3) - these checks are MECHANICAL, not instructions to the model:

1. ground_assumptions: an assumption that cannot be pointed at in THIS session's
   transcript is dropped (run-3 defect: the model echoed the prompt's own
   EXAMPLE assertions - "these sources can't conflict" - as founder statements).
2. dedup_constraints: a constraint whose fact is already a classified need is
   dropped (constraints limit the design space; they restate nothing).
3. is_solution_language: preview checks must be implementation-agnostic
   observable criteria, not API calls.
"""
from __future__ import annotations

import re

_STOP = {
    "the", "and", "for", "with", "must", "shall", "should", "will", "would", "have", "has",
    "this", "that", "these", "those", "from", "into", "your", "their", "them", "they",
    "when", "what", "which", "keep", "make", "made", "need", "needs", "able", "about",
    "over", "such", "than", "then", "them", "also", "only", "very", "more", "most",
    "some", "any", "all", "each", "other", "there", "here", "where", "while", "being",
    "been", "are", "was", "were", "not", "can", "could", "may", "might", "per", "via", "out", "off", "way", "easily", "easy",
}


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9\-]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _shared(a: set[str], b: set[str]) -> set[str]:
    """Overlap with light stemming: 'predict' catches 'predictive'."""
    out = set()
    for w in a:
        if w in b or any(len(w) >= 5 and (x.startswith(w) or w.startswith(x)) for x in b if len(x) >= 5):
            out.add(w)
    return out


def ground_assumptions(
    items: list[dict], transcript_turns: list[str], prompt_texts: list[str]
) -> tuple[list[dict], list[str]]:
    """Keep only assumptions provably grounded in this transcript.

    Drop (with a human-readable reason) when: source_turns point outside the
    transcript, the statement shares no vocabulary with the referenced turns,
    or the statement is an echo of a prompt example.
    """
    kept, dropped = [], []
    prompt_words = content_words(" ".join(prompt_texts))
    turns_words = content_words(" ".join(transcript_turns))
    for item in items:
        statement = str(item.get("statement", "")).strip()
        turns = item.get("source_turns") or []
        if not statement:
            dropped.append("(empty assumption statement)")
            continue
        if not turns or any(not isinstance(t, int) or t < 0 or t >= len(transcript_turns) for t in turns):
            dropped.append(f"'{statement[:70]}' - no valid source turns in this transcript")
            continue
        sw = content_words(statement)
        if sw:
            # echo guard: nearly all of the statement's vocabulary is in the prompt
            # but the referenced turns share almost none of it -> prompt echo.
            ref_words = content_words(" ".join(transcript_turns[t] for t in turns))
            in_prompt = len(_shared(sw, prompt_words)) / len(sw)
            in_turns = len(_shared(sw, ref_words)) / len(sw)
            if in_prompt >= 0.7 and in_turns < 0.3:
                dropped.append(f"'{statement[:70]}' - echoes interviewer prompt wording, not this transcript")
                continue
            if in_turns < 0.3 and not _shared(sw, ref_words):
                dropped.append(f"'{statement[:70]}' - no shared vocabulary with its source turns")
                continue
            # guard against impossible claims: statement absent from the whole transcript
            if not _shared(sw, turns_words):
                dropped.append(f"'{statement[:70]}' - nothing like it was said this session")
                continue
        kept.append(item)
    return kept, dropped


def dedup_constraints(constraints: list[str], need_statements: list[str]) -> tuple[list[str], list[str]]:
    """Drop constraints whose fact is already a need (the run-2/run-3
    double-classification). A constraint survives only if most of its content
    vocabulary appears in NO need statement."""
    kept, dropped = [], []
    need_word_sets = [content_words(s) for s in need_statements]
    for c in constraints:
        cw = content_words(c)
        if cw and any(
            len(_shared(cw, nw)) / len(cw) >= 0.5 or (len(_shared(cw, nw)) >= 2 and len(cw) <= 3)
            for nw in need_word_sets
        ):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


_SOLUTION_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\s+/|\bHTTP\b|\b[1-5]\d\d\s+(Unauthorized|OK|Found|Error|Created)"
    r"|/api/|endpoint|json response|status code|\bSQL\b|database table|button|text field|click",
    re.IGNORECASE,
)


def is_solution_language(text: str) -> bool:
    """True when a 'machine-checkable criterion' names an implementation
    (endpoint, status code, UI widget) instead of observable behavior."""
    return bool(_SOLUTION_RE.search(text))
