"""Session retrieval for question generation (JEV-D12, E4).

Deterministic, no embeddings. Each turn, code splits the founder's own turns
into spans, ranks them against the target area's cue set, and builds a short
quoted block that the interviewer sees next to the parked/drill blocks. The
block is dynamic, so the static prompt budget pin (3200 chars / 460 words) is
untouched. Caps and cue sets come from the question-quality config.
"""
from __future__ import annotations

import re

from . import question_config

_SENT = re.compile(r"(?<=[.!?])\s+|\n+|;\s+")
_HEADER = "SESSION FACTS (quote, do not paraphrase; ask about one, or name a gap): "


def build_spans(founder_turns: list[tuple[int, str]]) -> list[dict]:
    """founder_turns: (message index, text). One span per sentence or clause."""
    spans, seen = [], set()
    for idx, text in founder_turns:
        for part in _SENT.split(text or ""):
            part = part.strip().strip('"')
            key = part.casefold()
            if len(part) >= 8 and key not in seen:  # a repeated sentence is one fact
                seen.add(key)
                spans.append({"id": f"t{idx}", "turn": idx, "text": part})
    return spans


def _score(text: str, cues: list[str]) -> int:
    low = text.lower()
    words = re.findall(r"[a-z0-9']+", low)
    hit = 0
    for cue in cues:
        c = cue.lower()
        if " " in c:
            hit += c in low
        else:
            hit += any(w.startswith(c) for w in words)
    return hit


def rank_spans(spans: list[dict], area: str, cfg: dict | None = None) -> list[dict]:
    """Cue overlap first, then recency; spans with no cue hit are dropped."""
    cfg = cfg or question_config.current()
    cues = cfg["area_cues"].get(area, [])
    scored = [(_score(s["text"], cues), s["turn"], i, s) for i, s in enumerate(spans)]
    scored = [t for t in scored if t[0] > 0]
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [dict(s, score=sc) for sc, _, _, s in scored[: cfg["retrieval"]["top_k"]]]


def retrieval_block(founder_turns: list[tuple[int, str]], area: str | None, cfg: dict | None = None) -> str:
    """The dynamic block ('' when nothing relevant). Whole block, header included,
    stays within max_words and max_chars; lowest-ranked spans drop first."""
    if not area:
        return ""
    cfg = cfg or question_config.current()
    max_words, max_chars = cfg["retrieval"]["max_words"], cfg["retrieval"]["max_chars"]
    ranked = rank_spans(build_spans(founder_turns), area, cfg)
    chosen: list[str] = []
    for s in ranked:
        line = f'[{s["id"]}] "{s["text"]}"'
        candidate = "\n\n" + _HEADER + " ".join(chosen + [line])
        if len(candidate.split()) <= max_words and len(candidate) <= max_chars:
            chosen.append(line)
            continue
        if not chosen:  # even the best span is too long: cut it, never exceed the caps
            room_words = max_words - len((_HEADER + f'[{s["id"]}] ""').split())
            text = " ".join(s["text"].split()[: max(room_words, 0)])
            while text and len("\n\n" + _HEADER + f'[{s["id"]}] "{text}"') > max_chars:
                text = text[:-1].rstrip()
            if text:
                chosen.append(f'[{s["id"]}] "{text}"')
        break
    return ("\n\n" + _HEADER + " ".join(chosen)) if chosen else ""
