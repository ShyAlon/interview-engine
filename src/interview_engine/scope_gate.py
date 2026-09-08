"""Web-apps-only scope gate. Hybrid by design (ADR-0005):
deterministic signals first, LLM only as tiebreaker.

In scope: web applications, including ones that consume web APIs/services.
Out of scope (polite decline): native mobile apps, desktop, hardware/firmware,
pure APIs with no web surface, physical systems (step 2 of the vision).
"""
from __future__ import annotations

import re

from vmodel_contracts import AppArchetype, IntakeSubmission, ScopeVerdict

# Deterministic signal sets. Auditable: every verdict lists what fired.
_OUT_SIGNALS: list[tuple[str, str]] = [
    (r"\b(native|ios|android|app store|play store|swiftui|kotlin)\b", "mentions native mobile"),
    (r"\b(desktop|electron app|windows app|mac app)\b", "desktop application"),
    (r"\b(hardware|firmware|embedded|pcb|sensor|drone|vehicle|device)\b", "hardware/firmware"),
    (r"\b(browser extension|chrome extension)\b", "browser extension"),
    (r"\b(cli tool|command.line|dev.?tool sdk)\b", "CLI/dev tooling, no web surface"),
    (r"\b(game|unity|unreal)\b", "game"),
]
_IN_SIGNALS: list[tuple[str, str]] = [
    (r"\b(web ?app|website|web platform|web application|saas|portal|dashboard)\b", "explicit web app"),
    (r"\b(marketplace|booking|storefront|checkout)\b", "web commerce pattern"),
    (r"\b(login|sign up|account|subscription)\b", "account-based web pattern"),
]

_ARCHETYPE_SIGNALS: list[tuple[str, AppArchetype]] = [
    (r"\bmarketplace|two.sided|buyers? and sellers?\b", AppArchetype.MARKETPLACE),
    (r"\be-?commerce|shop|store|cart|checkout\b", AppArchetype.ECOMMERCE),
    (r"\bdashboard|analytics|metrics|reporting\b", AppArchetype.DASHBOARD),
    (r"\bblog|cms|content|publish\b", AppArchetype.CONTENT_SITE),
    (r"\bsaas|crm|manage|workflow|booking|subscription\b", AppArchetype.SAAS_CRUD),
]

_CONFIDENCE_FLOOR = 0.65  # below this, the LLM tiebreaker runs (and pays tokens)


def classify(sub: IntakeSubmission) -> tuple[list[str], list[str], AppArchetype]:
    text = (sub.free_text + " " + " ".join(sub.known_answers.values())).lower()
    out_hits = [label for pat, label in _OUT_SIGNALS if re.search(pat, text)]
    in_hits = [label for pat, label in _IN_SIGNALS if re.search(pat, text)]
    archetype = AppArchetype.OTHER_WEB
    for pat, arch in _ARCHETYPE_SIGNALS:
        if re.search(pat, text):
            archetype = arch
            break
    return out_hits, in_hits, archetype


def verdict_deterministic(sub: IntakeSubmission) -> ScopeVerdict | None:
    """Return a confident verdict, or None to escalate to the LLM tiebreaker."""
    out_hits, in_hits, archetype = classify(sub)
    signals = out_hits + in_hits
    if out_hits and not in_hits:
        return ScopeVerdict(
            submission_id=sub.submission_id, accepted=False,
            archetype=AppArchetype.NOT_WEB_APP, confidence=min(0.7 + 0.1 * len(out_hits), 0.97),
            deterministic_signals=signals, llm_tiebreak_used=False,
            decline_message=(
                "This service builds systems-engineering specs for web applications "
                "(which may consume web APIs and services). What you described looks like "
                f"{out_hits[0]}, which is outside the current scope - so rather than hand "
                "you a spec that doesn't fit, we'll stop here."
            ),
        )
    if in_hits and not out_hits:
        return ScopeVerdict(
            submission_id=sub.submission_id, accepted=True, archetype=archetype,
            confidence=min(0.7 + 0.08 * len(in_hits), 0.95),
            deterministic_signals=signals, llm_tiebreak_used=False,
        )
    return None  # mixed or silent: pay the tiebreaker


TIE_BREAK_PROMPT = """Classify this product idea for a service that ONLY builds specs for web applications
(apps in the browser; they may consume web APIs and services). Answer STRICT JSON:
{"is_web_app": true|false, "archetype": one of saas_crud|marketplace|content_site|dashboard|ecommerce|other_web|not_web_app, "confidence": 0..1, "reason": one sentence}
Idea: {text}"""


def verdict_with_llm(sub: IntakeSubmission, chat_model, usage_log=None) -> ScopeVerdict:
    """The ONLY LLM call in the gate. Runs when deterministic classification
    is mixed or silent - ambiguous wording is a semantic problem, pay for it."""
    import json as _json

    from .errors import translate_json_error
    from .model_config import invoke_checked

    resp = invoke_checked(chat_model, TIE_BREAK_PROMPT.format(text=sub.free_text), "scope_tiebreak")
    if usage_log is not None:
        usage_log.record("scope_tiebreak", getattr(resp, "response_metadata", {}).get("model_provider", "unknown"),
                         getattr(resp, "response_metadata", {}).get("model_name", "unknown"),
                         getattr(resp, "usage_metadata", {}) or {})
    try:
        data = _json.loads(resp.content.strip().strip("`").removeprefix("json").strip())
        accepted = bool(data["is_web_app"])
    except (_json.JSONDecodeError, KeyError, TypeError) as e:
        raise translate_json_error(e, "scope_tiebreak") from e
    return ScopeVerdict(
        submission_id=sub.submission_id,
        accepted=accepted,
        archetype=AppArchetype(data["archetype"]),
        confidence=float(data["confidence"]),
        deterministic_signals=classify(sub)[0] + classify(sub)[1],
        llm_tiebreak_used=True,
        decline_message=None if accepted else (
            "This service builds specs for web applications only. "
            f"{data['reason']} - so we'll stop here rather than hand you a spec that doesn't fit."
        ),
    )
