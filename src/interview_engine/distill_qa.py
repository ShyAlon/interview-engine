"""Deterministic post-distillation QA (ADR-0015). Prompt-level discipline failed
twice (run 2, run 3) - these checks are MECHANICAL, not instructions to the model:

1. ground_assumptions: an assumption that cannot be pointed at in THIS session's
   transcript is dropped (run-3 defect: the model echoed the prompt's own
   EXAMPLE assertions - "these sources can't conflict" - as founder statements).
2. dedup_constraints: a constraint whose fact is already a classified need is
   dropped (constraints limit the design space; they restate nothing).
3. is_solution_language / names_technology / grounded_in: SHARED from
   vmodel_contracts.text_qa (v0.6.0) - one implementation, both engines.
"""
from __future__ import annotations

import re

from vmodel_contracts.text_qa import (
    content_words,
    grounded_in,
    is_solution_language,
    names_technology,
    shared_words as _shared,
)


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


def _repeats(cw: set[str], nw: set[str]) -> bool:
    shared = len(_shared(cw, nw))
    return bool(cw) and (shared / len(cw) >= 0.5 or (shared >= 2 and len(cw) <= 3))


def dedup_constraints(constraints: list[str], need_statements: list[str]) -> tuple[list[str], list[str]]:
    """Drop constraints whose fact is already a need (the run-2/run-3
    double-classification). A constraint survives only if most of its content
    vocabulary appears in NO need statement."""
    kept, dropped = [], []
    need_word_sets = [content_words(s) for s in need_statements]
    for c in constraints:
        cw = content_words(c)
        if any(_repeats(cw, nw) for nw in need_word_sets):
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


_ALSO_OVERLAPS = 0.3


def explain_duplicate_constraint(constraint: str, needs: list[tuple[str, str]]) -> str:
    """Founder-readable reason for a constraint dropped by dedup_constraints.

    `needs` are (need_id, statement) pairs; an empty id is named by its wording only.
    Names the best-matching need (the one the rule fired on), lists other needs that
    share at least a third of the constraint's wording, and says why a fact should
    not appear twice. Same matching as the rule; no new threshold."""
    cw = content_words(constraint)
    scored = sorted(((len(_shared(cw, content_words(st))) / len(cw), nid, st)
                     for nid, st in needs if cw), key=lambda x: -x[0])
    firing = [x for x in scored if _repeats(cw, content_words(x[2]))]
    if not firing:
        return "This design constraint repeats a classified stakeholder need."
    _, nid, st = firing[0]
    label = f"need {nid}" if nid else "a need"
    text = f'Same fact as {label}: "{st.strip()}"'
    others = [x for x in scored if x[2] != st and x[0] >= _ALSO_OVERLAPS]
    if others:
        text += " It also overlaps " + ", ".join(
            f"need {n}" if n else f'"{t.strip()[:60]}"' for _, n, t in others) + "."
    return (text + " A fact should live in one place, either as a need (what people need) or "
            "as a design constraint (a limit on how it is built). Listed twice, the two copies "
            "can drift apart. Keep one.")



def verify_tech_preferences(prefs: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """ADR-0019 'open but verified': every ecosystem preference must (1) carry
    source_turns (founder actually said it - ungrounded is DROPPED, same rule
    as assumptions) and (2) name a recognized ecosystem for its area -
    unrecognized names are KEPT but warned (niche-but-real exists), so the
    founder sees the warning at the approval gate and confirms or corrects."""
    from vmodel_contracts import text_qa

    kept, dropped, warnings = [], [], []
    for pref in prefs:
        if not isinstance(pref, dict):
            dropped.append(str(pref)[:60])
            continue
        if not pref.get("source_turns"):
            dropped.append(f"{pref.get('area', '?')}: {pref.get('preference', '?')[:50]} (no source turns)")
            continue
        kept.append(pref)
        area = str(pref.get("area", ""))
        if not text_qa.ecosystem_recognized(area, str(pref.get("preference", ""))):
            warnings.append(
                f"technology preference '{pref.get('preference')}' ({area}) is not a recognized "
                f"ecosystem name - confirm the founder meant it before approving")
    return kept, dropped, warnings


REFERENCE_KINDS = ("working_title", "competitor", "reference_product")


def reference_name_reason(item: dict, transcript: list) -> str:
    """Precise founder-readable evidence for a rejected reference candidate."""
    if not isinstance(item, dict):
        return "This reference is not a valid product-name item."
    name = str(item.get("name") or "").strip()
    turns = item.get("source_turns") or []
    if not name:
        return "This reference has no product name."
    if item.get("kind") not in REFERENCE_KINDS:
        return "This reference has an invalid reference kind."
    if not turns:
        return "This product name has no cited founder turns."
    evidence = []
    non_founder = False
    for turn in turns:
        if isinstance(turn, bool) or not isinstance(turn, int) or not 0 <= turn < len(transcript):
            evidence.append(f"Turn {turn}: no such interview turn.")
            continue
        message = transcript[turn]
        founder = getattr(message, "type", "") == "human"
        non_founder = non_founder or not founder
        role = "You" if founder else "Interviewer"
        quote = str(getattr(message, "content", ""))
        evidence.append(f'Turn {turn} ({role}): "{quote[:900]}"')
    reason = ("A cited turn is the interviewer, not you. Interviewer examples are not founder evidence."
              if non_founder else "This exact product name was not found in your cited words.")
    return (reason + "\n" + "\n".join(evidence))[:2000]


def reject_non_founder_citations(value, transcript: list, path: str = "distillation") -> None:
    """All producer provenance is founder-only, including framing citation maps."""
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{path}.{key}"
            if key == "source_turns":
                def check(citations, label):
                    if isinstance(citations, dict):
                        for field, turns in citations.items():
                            check(turns, f"{label}.{field}")
                        return
                    if not isinstance(citations, list):
                        raise ValueError(f"{label} must be a list of founder turn numbers")
                    for turn in citations:
                        invalid = (isinstance(turn, bool) or not isinstance(turn, int)
                                   or not 0 <= turn < len(transcript))
                        if invalid or getattr(transcript[turn], "type", "") != "human":
                            detail = reference_name_reason(
                                {"name": "cited evidence", "kind": "reference_product", "source_turns": [turn]},
                                transcript)
                            raise ValueError(f"{label} cites non-founder evidence: {detail} "
                                             "Cite only human/founder turns, or remove the unsupported assertion.")
                check(child, current)
            else:
                reject_non_founder_citations(child, transcript, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_non_founder_citations(child, transcript, f"{path}[{index}]")


def reject_non_founder_reference_citations(items: list[dict], transcript: list) -> None:
    reject_non_founder_citations(items, transcript, "reference_names")


def verify_reference_names(items: list[dict], founder_turns: dict[int, str]) -> tuple[list[dict], list[str]]:
    """Keep a reference name only when it is provably the founder's own: it cites founder turns, its kind is
    one of REFERENCE_KINDS, and the name appears (case-insensitive, whole word) in one of those turns.
    Case-insensitive duplicates collapse to the first. Everything else is DROPPED with a reason; the
    caller raises it for founder review, never silently keeps it."""
    kept, dropped, seen = [], [], set()
    for item in items:
        if not isinstance(item, dict):
            dropped.append(str(item)[:60])
            continue
        name = str(item.get("name") or "").strip()
        kind = str(item.get("kind") or "").strip()
        turns = item.get("source_turns") or []
        if not name:
            dropped.append("(empty reference name)")
        elif kind not in REFERENCE_KINDS:
            dropped.append(f"'{name[:50]}' - kind '{kind[:30]}' is not one of {', '.join(REFERENCE_KINDS)}")
        elif not turns or any(not isinstance(t, int) or t not in founder_turns for t in turns):
            dropped.append(f"'{name[:50]}' - no valid founder turns cited")
        else:
            pattern = r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(part) for part in name.split()) + r"(?![A-Za-z0-9])"
            if not any(re.search(pattern, founder_turns[t], re.IGNORECASE) for t in turns):
                dropped.append(f"'{name[:50]}' - not found in the founder's cited turns")
            elif name.casefold() in seen:
                continue
            else:
                seen.add(name.casefold())
                kept.append({"name": name, "kind": kind, "source_turns": [int(t) for t in turns]})
    return kept, dropped


# Founder wording that postpones a need. Same family as spec-generator's fallback
# markers; here it only CONFIRMS a model-proposed deferred=true against the
# founder's own turns, it never sets the flag by itself.
_DEFER_WORDING = re.compile(
    r"\b(?:in the future|future (?:phase|release|version)|phase (?:2|two|ii)|v2|post[- ]?mvp|down the road|"
    r"later(?: on)?|after launch|eventually|someday|not yet|not now|for now|next version)\b", re.IGNORECASE)


def coerce_deferred(need: dict, founder_turns: dict[int, str]) -> dict:
    """Normalize need['deferred'] to a strict bool. True survives only when one of the
    need's source_turns is a founder turn with postponing wording; otherwise False
    (the safe default - the need stays a current need). source_turns is untouched."""
    out = dict(need)
    raw = out.get("deferred")
    flag = raw if isinstance(raw, bool) else (str(raw).strip().lower() == "true" if raw is not None else False)
    if flag:
        flag = any(_DEFER_WORDING.search(founder_turns.get(t, "")) for t in (out.get("source_turns") or []))
    out["deferred"] = flag
    return out
