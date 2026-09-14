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
    lines.append("Reply with the option number or exact concept name. This is an explicit founder decision.")
    return "\n".join(lines)


def selected_option(reply: str, options: list[dict]) -> ConceptOption | None:
    """Exact name or standalone option number only. No fuzzy/model inference."""
    text = reply.strip()
    m = re.fullmatch(r"(?:option\s*)?(\d+)[.!]?", text, re.I)
    if m:
        i = int(m.group(1)) - 1
        return ConceptOption.model_validate(options[i]) if 0 <= i < len(options) else None
    matches = [ConceptOption.model_validate(o) for o in options
               if text.casefold() == str(o.get("name", "")).strip().casefold()]
    return matches[0] if len(matches) == 1 else None


def finalize(draft: dict, session_id: str, founder_reply: str, source_turn: int) -> ConOpsDocument:
    selected = selected_option(founder_reply, draft["concept_options"])
    if selected is None:
        raise ValueError("founder reply did not exactly select one compared concept")
    return ConOpsDocument.model_validate({
        "session_id": session_id,
        **draft,
        "decision": ConceptDecision(
            selected_concept=selected.name,
            rationale=f"Founder selected {selected.name} after reviewing the recorded trade-offs.",
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
