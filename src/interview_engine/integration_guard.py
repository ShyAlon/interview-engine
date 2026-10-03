"""None-stated guard for the integrations area (JEV-D12, E3).

Deterministic: code decides whether the integrations question may be asked.
It is asked only if some founder turn names an external system; if a founder
turn says the product is self-contained, or names nothing, code closes the
area as NONE_STATED (visible in the coverage summary, never as founder-covered).
Cue lists come from the question-quality config, so they can be tuned.
"""
from __future__ import annotations

import re

from . import question_config

_HOST = re.compile(r"\b[a-z0-9-]+\.(?:com|io|org|net|ai|app|dev|co|gov|edu)\b", re.I)
_API = re.compile(r"\bAPIs?\b")
# Capitalized words that are not systems: sentence starters and common words.
_NOT_SYSTEMS = {"i", "we", "it", "the", "a", "an", "my", "our", "this", "that", "they", "kids", "users",
                "yes", "no", "ok", "okay", "so", "and", "but", "also", "if", "when", "for", "to", "in", "on"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _has_proper_noun(text: str) -> bool:
    for sentence in _SENT_SPLIT.split(text):
        words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*", sentence)
        for w in words[1:]:  # the first word of a sentence is capitalized anyway
            if w[0].isupper() and w.lower() not in _NOT_SYSTEMS and w != "I" and not w.isupper():
                return True
    return False


def names_external_system(turns: list[str], cfg: dict | None = None) -> bool:
    cfg = cfg or question_config.current()
    verbs = [v.lower() for v in cfg["integrations"]["external_system_verbs"]]
    for t in turns:
        low = t.lower()
        if _HOST.search(t) or _API.search(t) or _has_proper_noun(t):
            return True
        if any(re.search(rf"\b{re.escape(v)}\w*\b", low) for v in verbs):
            return True
    return False


def says_self_contained(turns: list[str], cfg: dict | None = None) -> bool:
    cfg = cfg or question_config.current()
    cues = [c.lower().replace("-", " ") for c in cfg["integrations"]["self_contained_cues"]]
    return any(c in t.lower().replace("-", " ") for t in turns for c in cues)


def sn_named_systems(intent_model: dict | None, cfg: dict | None = None) -> bool:
    """Stakeholder-need (SN) evidence of a named system: a non-empty
    external_integrations list, or a need statement that names one."""
    im = intent_model or {}
    if any(str(x).strip() for x in im.get("external_integrations") or []):
        return True
    statements = [str(n.get("statement", "")) for n in im.get("stakeholder_needs") or [] if isinstance(n, dict)]
    return names_external_system(statements, cfg)


def close_as_none_stated(founder_turns: list[str], cfg: dict | None = None, intent_model: dict | None = None) -> bool:
    """True when the integrations question must not be asked.

    A named system in the founder's turns or in the SN set keeps the question.
    A founder turn saying the product is self-contained closes it.
    """
    cfg = cfg or question_config.current()
    if not cfg["integrations"]["none_stated_enabled"]:
        return False
    if says_self_contained(founder_turns, cfg):
        return True
    if sn_named_systems(intent_model, cfg):
        return False
    return not names_external_system(founder_turns, cfg)
