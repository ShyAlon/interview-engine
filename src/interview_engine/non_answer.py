"""Code-owned non-answer detector (JEV-D12 3.3.3).

Session 42b71e2e: the founder replied "I don't understand the question" and the
data area closed as covered, because the reply went through the same coverage
path as a real answer. A non-answer never closes or thins an area; the same
area is re-asked through a self-contained template. The list is closed and
verified by tests; no model call, and the categorizer's verdict is not trusted
to override it.
"""
from __future__ import annotations

import re

_NON_ANSWER_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^(?:\?+|what\s*\??|huh\s*\??|sorry\s*\??|pardon\s*\??|eh\s*\??)$",
    r"^(?:i|we)\s+(?:do\s*n[o']?t|don'?t|dont|did\s*n[o']?t|didn'?t|cannot|can'?t|cant)\s+"
    r"(?:really\s+)?(?:understand|get|follow)(?:\s+(?:the|that|this|it|your|what you(?:'re| are) asking|what you mean))?"
    r"(?:\s+(?:question|one|ask|part))?$",
    r"^(?:i'?m|i am|im|we are|we're)\s+(?:not\s+sure|unsure|confused)(?:\s+(?:what|how)\s+you\s+(?:mean|are asking|'re asking))?$",
    r"^(?:not\s+sure\s+(?:what\s+you\s+mean|what\s+you(?:'re| are)\s+asking|i\s+follow))$",
    r"^(?:what\s+do\s+you\s+mean|what\s+does\s+that\s+mean|what\s+are\s+you\s+asking|"
    r"can\s+you\s+(?:rephrase|clarify|explain)(?:\s+(?:that|the question|it))?|"
    r"(?:please\s+)?(?:rephrase|clarify)(?:\s+(?:that|the question))?|"
    r"that\s+(?:question\s+)?(?:is|was)\s+(?:unclear|confusing|vague)|"
    r"the\s+question\s+(?:is|was)\s+(?:unclear|confusing|vague)|"
    r"unclear|confusing|no\s+idea\s+what\s+you\s+mean)\s*\??$",
))


def is_non_answer(text: str) -> bool:
    """True when the whole reply says it did not understand the question."""
    t = re.sub(r"\s+", " ", (text or "").strip().strip('"\u201c\u201d')).rstrip(".!, ").strip()
    if not t:
        return True
    return any(p.search(t) for p in _NON_ANSWER_RES)
