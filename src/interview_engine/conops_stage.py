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



def validate_draft(data: dict, founder_turns: set[int]) -> dict:
    """Validate the model-authored draft and reject fabricated provenance."""
    contexts = TypeAdapter(list[StakeholderContext]).validate_python(
        data.get("stakeholder_context"))
    scenarios = TypeAdapter(list[OperationalScenario]).validate_python(
        data.get("operational_scenarios"))
    raw_options = []
    from vmodel_contracts import canonical_concept_id
    for raw in data.get("concept_options") or []:
        item = dict(raw)
        item.setdefault("concept_id", canonical_concept_id(str(item.get("name", ""))))
        raw_options.append(item)
    options = TypeAdapter(list[ConceptOption]).validate_python(raw_options)
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


def founder_decision(draft: dict) -> dict:
    """Code-owned UI contract for every CONOPS option choice.

    The model owns option substance. Code owns stable values, ordering and the
    exact answer format consumed by ``decision_reply``.
    """
    options=[]
    for index,raw in enumerate(draft["concept_options"],1):
        option=ConceptOption.model_validate(raw)
        options.append({"value":str(index),"label":option.name,"summary":option.summary,
                        "advantages":list(option.advantages),"disadvantages":list(option.disadvantages)})
    from vmodel_contracts.decisions import founder_decision as decision_contract
    return decision_contract(CONCEPT_DECISION,"Choose the operating concept","Which concept should the architecture execute?",options,
            rationale_required=True,rationale_label="Why is this the right concept?",answer_format="<value> - <rationale>")


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
            selected_concept_id=selected.concept_id,
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


def scenario_seeds_from_intent(intent: dict, founder_turns: set[int]) -> list[dict]:
    """Derive scenario identities from intent concepts/scope, never model prose."""
    concepts = [str(x).strip() for x in (intent.get("operational_concepts") or []) if str(x).strip()]
    if not concepts:
        concepts = [str(x).strip() for x in (intent.get("in_scope") or []) if str(x).strip()][:3]
    if not concepts:
        concepts = [str(intent.get("core_value_proposition") or "Primary operation").strip()]
    source_map = intent.get("source_turns") or {}
    source = [t for key in ("operational_concepts", "in_scope", "core_value_proposition")
              for t in (source_map.get(key) or []) if t in founder_turns]
    if not source:
        source = sorted({t for n in (intent.get("stakeholder_needs") or [])
                         for t in (n.get("source_turns") or []) if t in founder_turns})
    return [{"scenario_id": f"OS-{i}", "name": name, "source_turns": sorted(set(source))}
            for i, name in enumerate(concepts, 1)]


def item_repair_prompt(kind: str, identity: dict, candidate: dict, error: Exception, intent: dict) -> str:
    return (f"TARGETED {kind.upper()} REPAIR. Return one JSON object containing only missing or invalid fields. "
            "Do not rename the code-owned identity and do not rewrite valid fields. "
            f"Identity: {json.dumps(identity)}\nValidation errors: {error}\n"
            f"Existing item: {json.dumps(candidate)}\nValidated intent evidence: {json.dumps(intent)}")


def derive_stakeholder_contexts(intent: dict, founder_turns: set[int]) -> list[dict]:
    """Derive only facts already present in the validated intent artifact."""
    seeds = preseed_from_intent(intent, founder_turns)["stakeholder_context"]
    needs = [n for n in (intent.get("stakeholder_needs") or []) if isinstance(n, dict)]
    constraints = [str(x).strip() for x in (intent.get("constraints") or []) if str(x).strip()]
    concepts = [str(x).strip() for x in (intent.get("operational_concepts") or []) if str(x).strip()]
    for seed in seeds:
        words = set(re.findall(r"[a-z0-9]+", seed["stakeholder"].casefold()))
        matched = [n for n in needs if words & set(re.findall(r"[a-z0-9]+", str(n.get("statement","")).casefold()))]
        if not matched and len(seeds) == 1:
            matched = needs
        seed["role"] = seed["stakeholder"]
        seed["goals"] = [str(n["statement"]).strip() for n in matched if str(n.get("statement","")).strip()]
        environment = constraints + concepts
        if environment:
            seed["operating_environment"] = "; ".join(environment)
        seed["source_turns"] = sorted({t for n in matched for t in (n.get("source_turns") or [])
                                       if t in founder_turns}) or seed["source_turns"]
        seed["_derived_fields"] = [k for k in ("role","goals","operating_environment","source_turns")
                                    if seed.get(k) not in (None,"",[])]
    return seeds


def derive_scenario(seed: dict, intent: dict) -> dict:
    """Pre-fill scenario fields only where IntentModel already owns substance."""
    out = dict(seed)
    users = [str(x).strip() for x in (intent.get("target_users") or []) if str(x).strip()]
    flows = [str(x).strip() for x in (intent.get("operational_concepts") or []) if str(x).strip()]
    if users:
        out["actors"] = users
    if flows and out.get("name") in flows:
        out["operational_flow"] = [out["name"]]
    value = str(intent.get("core_value_proposition") or "").strip()
    if value:
        out["success_outcome"] = value
    out["exceptions"] = []
    out["_derived_fields"] = [k for k in ("scenario_id","name","actors","operational_flow",
                                           "success_outcome","exceptions","source_turns") if k in out]
    return out
