"""The founder-facing CONOPS segment.

The model extracts operational context/scenarios and proposes concept-level
options. Code owns presentation, exact founder-choice matching, provenance,
and final contract validation.
"""
from __future__ import annotations

import json
import re

from pydantic import TypeAdapter
from vmodel_contracts import (CONCEPT_DECISION, CONCEPT_OF_OPERATIONS,
                              ConceptDecision, ConceptOption, ConOpsDocument,
                              OperationalScenario, StakeholderContext)

CONOPS_SYSTEM = """Build the CONOPS comparison from this founder interview. This is product
concept, not component architecture. Extract stakeholder operating context and concrete end-to-end
operational scenarios. Then compare 2-4 genuinely distinct product concepts (for example native
app, PWA, web plus native bridge) with concise advantages and disadvantages grounded in the
founder's operation. Do NOT select a winner: the founder decides after seeing the comparison.
Return strict JSON only with stakeholder_context[], operational_scenarios[], concept_options[].
Every context and scenario must cite founder source_turns from the numbered transcript."""



def preseed_from_intent(intent: dict, founder_turns: set[int]) -> dict:
    """Code-owned CONOPS skeleton from already-validated intent facts.

    It seeds identities and provenance only. Goals, roles and operating context
    remain model work because inventing them mechanically would be fabrication.
    """
    needs = [n for n in (intent.get("stakeholder_needs") or []) if isinstance(n, dict)]
    all_turns = sorted({t for n in needs for t in (n.get("source_turns") or []) if t in founder_turns})
    contexts = []
    for stakeholder in intent.get("target_users") or []:
        name = str(stakeholder).strip()
        if not name:
            continue
        words = set(re.findall(r"[a-z0-9]+", name.casefold()))
        matched = sorted({t for n in needs
                          if words & set(re.findall(r"[a-z0-9]+", str(n.get("statement", "")).casefold()))
                          for t in (n.get("source_turns") or []) if t in founder_turns})
        contexts.append({"stakeholder": name, "source_turns": matched or all_turns})
    return {"stakeholder_context": contexts}


def merge_draft(base: dict, patch: dict, seed: dict | None = None) -> dict:
    """Merge a targeted repair without allowing it to erase valid draft work."""
    out = json.loads(json.dumps(base if isinstance(base, dict) else {}))

    def merge_items(existing: list, incoming: list, identity: str) -> list:
        result = [dict(x) for x in existing if isinstance(x, dict)]
        by_id = {str(x.get(identity, "")).strip().casefold(): x for x in result if x.get(identity)}
        for item in incoming:
            if not isinstance(item, dict):
                continue
            ident = str(item.get(identity, "")).strip().casefold()
            if ident and ident in by_id:
                by_id[ident].update({k: v for k, v in item.items() if v not in (None, "", [])})
            else:
                result.append(dict(item))
                if ident:
                    by_id[ident] = result[-1]
        return result

    identities = {"stakeholder_context": "stakeholder", "operational_scenarios": "scenario_id",
                  "concept_options": "name"}
    for key, identity in identities.items():
        original = out.get(key) or []
        seeded = (seed or {}).get(key) or []
        # A draft's explicit identities win. Seeds guarantee identity when the
        # model omits the collection/item, but never invent a second alias for
        # a stakeholder the model named differently.
        repaired = patch.get(key) or []
        authoritative = original or repaired
        if authoritative:
            existing_ids = {str(x.get(identity, "")).strip().casefold()
                            for x in authoritative if isinstance(x, dict)}
            seeded = [x for x in seeded if str(x.get(identity, "")).strip().casefold() in existing_ids]
        out[key] = merge_items(merge_items(seeded, original, identity), repaired, identity)
    for key, value in patch.items():
        if key not in identities and value not in (None, "", []):
            out[key] = value
    return out


def repair_prompt(draft: dict, error: Exception, intent: dict) -> str:
    """Request only schema-invalid fields; valid draft content is immutable input."""
    return ("TARGETED CONOPS REPAIR. Do not reroll or rewrite valid content. Return one JSON object "
            "containing only missing or invalid fields, nested under stakeholder_context, "
            "operational_scenarios, or concept_options with each item's identity field. "
            "Use the validation errors and validated intent evidence below. Every repaired context "
            "or scenario source_turns must cite founder turns already present in the draft/intent.\n\n"
            f"VALIDATION ERRORS:\n{error}\n\nEXISTING DRAFT (preserve):\n{json.dumps(draft)}"
            f"\n\nVALIDATED INTENT EVIDENCE:\n{json.dumps(intent)}")

def validate_draft(data: dict, founder_turns: set[int]) -> dict:
    """Validate the model-authored draft and reject fabricated provenance."""
    contexts = TypeAdapter(list[StakeholderContext]).validate_python(
        data.get("stakeholder_context"))
    scenarios = TypeAdapter(list[OperationalScenario]).validate_python(
        data.get("operational_scenarios"))
    options = TypeAdapter(list[ConceptOption]).validate_python(data.get("concept_options"))
    if not contexts or not scenarios or len(options) < 2:
        raise ValueError("CONOPS needs context, scenarios, and at least two concepts")
    cited = [t for x in [*contexts, *scenarios] for t in x.source_turns]
    invalid = sorted(set(cited) - founder_turns)
    if invalid:
        raise ValueError(f"CONOPS cites non-founder transcript turns: {invalid}")
    names = [o.name.casefold() for o in options]
    if len(set(names)) != len(names):
        raise ValueError("concept option names must be unique")
    return {"stakeholder_context": [x.model_dump(mode="json") for x in contexts],
            "operational_scenarios": [x.model_dump(mode="json") for x in scenarios],
            "concept_options": [x.model_dump(mode="json") for x in options]}


def comparison_question(draft: dict) -> str:
    lines = [f"{CONCEPT_DECISION.title()} - choose the operating concept the architecture must execute:"]
    for i, raw in enumerate(draft["concept_options"], 1):
        o = ConceptOption.model_validate(raw)
        lines += [f"{i}. {o.name}: {o.summary}",
                  f"   + {'; '.join(o.advantages)}",
                  f"   - {'; '.join(o.disadvantages)}"]
    lines.append("Reply with the option number or exact concept name, then ` - ` and why you chose it. This is an explicit founder decision.")
    return "\n".join(lines)


def decision_reply(reply: str, options: list[dict]) -> tuple[ConceptOption, str] | None:
    """Exact option + founder rationale. No fuzzy/model inference."""
    choice, sep, rationale = reply.strip().partition(" - ")
    if not sep or not rationale.strip():
        return None
    m = re.fullmatch(r"(?:option\s*)?(\d+)[.!]?", choice.strip(), re.I)
    if m:
        i = int(m.group(1)) - 1
        selected = ConceptOption.model_validate(options[i]) if 0 <= i < len(options) else None
    else:
        matches = [ConceptOption.model_validate(o) for o in options
                   if choice.strip().casefold() == str(o.get("name", "")).strip().casefold()]
        selected = matches[0] if len(matches) == 1 else None
    return (selected, rationale.strip()) if selected else None


def finalize(draft: dict, session_id: str, founder_reply: str, source_turn: int) -> ConOpsDocument:
    parsed = decision_reply(founder_reply, draft["concept_options"])
    if parsed is None:
        raise ValueError("founder reply did not exactly select one concept and give a rationale")
    selected, rationale = parsed
    return ConOpsDocument.model_validate({
        "session_id": session_id,
        **draft,
        "decision": ConceptDecision(
            selected_concept=selected.name,
            rationale=rationale,
            founder_confirmed=True,
            source_turns=[source_turn],
        ).model_dump(mode="json"),
    })


def render_conops(doc: ConOpsDocument) -> str:
    lines = [f"# {CONCEPT_OF_OPERATIONS}", "", "## Stakeholder context"]
    for c in doc.stakeholder_context:
        lines += [f"- **{c.stakeholder}** ({c.role}): {c.operating_environment}",
                  f"  Goals: {'; '.join(c.goals)}"]
    lines += ["", "## Operational scenarios"]
    for s in doc.operational_scenarios:
        lines += [f"### {s.scenario_id} {s.name}", f"Trigger: {s.trigger}",
                  f"Actors: {', '.join(s.actors)}"]
        lines += [f"{i}. {step}" for i, step in enumerate(s.operational_flow, 1)]
        lines.append(f"Success: {s.success_outcome}")
        if s.exceptions:
            lines.append(f"Exceptions: {'; '.join(s.exceptions)}")
    lines += ["", "## Concept comparison"]
    for o in doc.concept_options:
        marker = " (selected)" if o.name.casefold() == doc.decision.selected_concept.casefold() else ""
        lines += [f"### {o.name}{marker}", o.summary,
                  f"Advantages: {'; '.join(o.advantages)}",
                  f"Disadvantages: {'; '.join(o.disadvantages)}"]
    lines += ["", "## Founder decision", f"**{doc.decision.selected_concept}**",
              doc.decision.rationale]
    return "\n".join(lines) + "\n"
