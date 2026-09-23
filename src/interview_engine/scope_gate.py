"""Interactive-application scope gate. Hybrid by design (ADR-0005):
deterministic signals first, LLM only as tiebreaker. The tuneable scope
wording and signals are imported from vmodel-contracts.product_scope."""
from __future__ import annotations

import re

from vmodel_contracts import (AppArchetype, IntakeSubmission,
                              PRODUCT_SCOPE_DEFINITION, SCOPE_DECLINE_PREFIX,
                              SCOPE_IN_SIGNALS, SCOPE_OUT_SIGNALS, ScopeVerdict)

# Aliases kept module-local only for compatibility with existing tests/imports;
# the data itself has one owner in the contract layer.
_OUT_SIGNALS = SCOPE_OUT_SIGNALS
_IN_SIGNALS = SCOPE_IN_SIGNALS

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
                SCOPE_DECLINE_PREFIX + " What you described looks like "
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


TIE_BREAK_PROMPT = """Classify this product idea against this product scope:
{scope}
Accept interactive software applications delivered on web or mobile platforms, including games.
Reject non-software businesses/products (for example a bakery or logistics company), hardware/firmware,
and software with no interactive web/mobile application surface. Answer STRICT JSON:
{{{{"is_web_app": true|false, "archetype": one of saas_crud|marketplace|content_site|dashboard|ecommerce|other_web|not_web_app, "confidence": 0..1, "reason": one sentence}}}}
Idea: {{text}}""".format(scope=PRODUCT_SCOPE_DEFINITION)


def verdict_with_llm(sub: IntakeSubmission, chat_model, usage_log=None, raw_output_log=None) -> ScopeVerdict:
    """The ONLY LLM call in the gate. Runs when deterministic classification
    is mixed or silent - ambiguous wording is a semantic problem, pay for it."""
    import json as _json

    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    from . import model_config
    from .errors import translate_json_error
    from .model_config import invoke_checked
    from .repair import extract_json_value

    class _TieBreakResult(BaseModel):
        model_config = ConfigDict(extra="forbid")
        is_web_app: bool
        archetype: AppArchetype
        confidence: float = Field(ge=0.0, le=1.0)
        reason: str = Field(min_length=1)

    prompt = TIE_BREAK_PROMPT.format(text=sub.free_text)
    reminder = ("\n\nFINAL RESPONSE CONTRACT: Return only the JSON object requested above; "
                "all five fields are required; no prose or markdown fences.")
    last_error = None
    raw_path = None
    for attempt in (1, 2):
        feedback = ("\nYour previous response failed validation: " + str(last_error)[:400]
                    + "\nCorrect it and return only the complete JSON object.") if last_error else ""
        resp = invoke_checked(chat_model, prompt + feedback + reminder, "scope_tiebreak")
        if usage_log is not None:
            # Price from the configured route, not response metadata.
            cfg = model_config.for_stage("scope_tiebreak")
            usage_log.record("scope_tiebreak", cfg.provider, cfg.model,
                             getattr(resp, "usage_metadata", {}) or {})
        raw_path = raw_output_log.write("scope-tiebreak", "classification" + ("-repair" if attempt == 2 else ""), resp.content) if raw_output_log else None
        try:
            data = _TieBreakResult.model_validate(_json.loads(extract_json_value(resp.content)))
            accepted = data.is_web_app
            break
        except (_json.JSONDecodeError, ValueError, ValidationError) as e:
            last_error = e
            if attempt == 1:
                continue
            error = translate_json_error(e, "scope_tiebreak")
            if raw_path:
                error.next_action = f"inspect the raw response at {raw_path}, then " + (error.next_action or "retry.")
            raise error from e
    return ScopeVerdict(
        submission_id=sub.submission_id,
        accepted=accepted,
        archetype=data.archetype,
        confidence=data.confidence,
        deterministic_signals=classify(sub)[0] + classify(sub)[1],
        llm_tiebreak_used=True,
        decline_message=None if accepted else (
            SCOPE_DECLINE_PREFIX + " " + f"{data.reason} - so we'll stop here rather than hand you a spec that doesn't fit."
        ),
    )

# Provider-neutral adapters for the categorization registry. They preserve the
# ScopeVerdict contract while moving provider selection out of graph code.
def categorization_request(sub: IntakeSubmission):
    from .categorization import ChoiceQuestion, ClassificationRequest
    return ClassificationRequest(
        slot="scope",
        state=("Product idea:\n" + sub.free_text + "\nKnown answers:\n"
               + "\n".join(f"{k}: {v}" for k, v in sub.known_answers.items())),
        questions={
            "accepted": ChoiceQuestion(
                "Is this an interactive software application delivered on web or mobile?",
                {"accepted": "Interactive web/mobile software, including games",
                 "rejected": "Non-software, hardware/firmware, or software without an interactive web/mobile surface"}),
            "archetype": ChoiceQuestion(
                "Select the product archetype that best describes the idea",
                {value.value: value.value.replace("_", " ") for value in AppArchetype}),
        },
        context={"submission": sub.model_dump(mode="json")},
    )


def rules_provider(sub: IntakeSubmission):
    from .categorization import ClassificationAnswer
    verdict = verdict_deterministic(sub)
    if verdict is None:
        raise ValueError("deterministic scope signals are mixed or silent")
    return {
        "accepted": ClassificationAnswer("accepted" if verdict.accepted else "rejected", verdict.confidence),
        "archetype": ClassificationAnswer(verdict.archetype.value, verdict.confidence),
    }


def llm_provider(sub: IntakeSubmission, chat_model, usage_log=None, raw_output_log=None):
    from .categorization import ClassificationAnswer
    verdict = verdict_with_llm(sub, chat_model, usage_log, raw_output_log)
    return {
        "accepted": ClassificationAnswer("accepted" if verdict.accepted else "rejected", verdict.confidence),
        "archetype": ClassificationAnswer(verdict.archetype.value, verdict.confidence),
    }


WEB_ARCHETYPES = frozenset(a.value for a in AppArchetype if a is not AppArchetype.NOT_WEB_APP)


def reconcile_scope(accepted: str, archetype: str) -> tuple[str, bool]:
    """Field-consistency rule (2026-09-23 merge, agreed by Shy): 'rejected'
    with a web archetype is self-contradictory - the decider named what kind
    of web product it is. Treat it as accepted with that archetype. Returns
    (accepted_value, was_corrected)."""
    if accepted == "rejected" and archetype in WEB_ARCHETYPES:
        return "accepted", True
    return accepted, False


def verdict_with_categorizer(sub: IntakeSubmission, categorizer) -> ScopeVerdict:
    result = categorizer.classify(categorization_request(sub))
    accepted_answer = result.answers["accepted"]
    archetype_answer = result.answers["archetype"]
    accepted_value, _corrected = reconcile_scope(str(accepted_answer.value), str(archetype_answer.value))
    accepted = accepted_value == "accepted"
    archetype = AppArchetype(str(archetype_answer.value))
    confidence_values = [x for x in (accepted_answer.confidence, archetype_answer.confidence) if x is not None]
    confidence = min(confidence_values) if confidence_values else 0.0
    return ScopeVerdict(
        submission_id=sub.submission_id,
        accepted=accepted,
        archetype=archetype,
        confidence=confidence,
        deterministic_signals=classify(sub)[0] + classify(sub)[1],
        llm_tiebreak_used=result.provider == "llm",
        decline_message=None if accepted else (
            SCOPE_DECLINE_PREFIX + " The product was classified outside the current interactive web/mobile scope, "
            "so we'll stop here rather than hand you a spec that doesn't fit."),
    )
