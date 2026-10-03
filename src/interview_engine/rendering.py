"""Human-readable artifact rendering, independent of transport."""
from __future__ import annotations
import json
from pathlib import Path
from vmodel_contracts import IntentModel

def _prefer_migrated_intent(outdir: Path, session: str, checkpoint_intent: dict) -> tuple[dict, Path | None]:
    """Use a current, schema-valid disk artifact over a stale checkpoint copy.

    The migration command named by the version error targets
    intent_model.raw.json, so that file is checked first. A stale or invalid
    candidate is ignored; normal compatibility handling remains downstream.
    """
    from vmodel_contracts import check_compatible
    for name in ("intent_model.raw.json", "intent_model.json"):
        path = outdir / name
        if not path.is_file():
            continue
        try:
            candidate = json.loads(path.read_text())
            check_compatible(candidate)
            validated = IntentModel.model_validate(candidate)
        except Exception:
            continue
        if validated.session_id == session:
            return validated.model_dump(mode="json"), path
    return checkpoint_intent, None

def render_intent_markdown(intent: IntentModel) -> str:
    """Human-readable rendering for the founder review gate (scoping doc, step 2),
    in systems-engineering terms: needs (functional / non-functional) distinct
    from scope, constraints, and open questions. Requirements (verifiable
    shall-statements) are derived downstream - this gate approves intent."""
    from vmodel_contracts import RequirementKind

    from .distill_qa import content_words, _shared

    functional = [n for n in intent.stakeholder_needs if n.kind == RequirementKind.FUNCTIONAL]
    non_functional = [n for n in intent.stakeholder_needs if n.kind == RequirementKind.NON_FUNCTIONAL]
    lines = [
        f"# What we understood\n",
        f"**Problem.** {intent.problem_statement}\n",
        f"**For.** {', '.join(intent.target_users)}\n",
        f"**Core value.** {intent.core_value_proposition}\n",
        "\n## Stakeholder needs - functional",
        *([f"- **{n.need_id}** {n.statement}" for n in functional] or ["- (none captured)"]),
        "\n## Stakeholder needs - non-functional",
        *([f"- **{n.need_id}** {n.statement}" for n in non_functional] or ["- (none captured)"]),
        "\n## Scope",
        "**In scope**", *[f"- {s}" for s in intent.in_scope],
    ]
    core_words = content_words(" ".join(
        [intent.problem_statement, intent.core_value_proposition, *intent.in_scope]))

    _GENERIC = {"application", "app", "system", "product", "platform", "software", "solution",
                "web", "service", "tool", "website", "site", "mobile", "mission-critical",
                "mission", "critical", "use", "using", "used", "users", "user", "new", "via"}

    def _flag(item: str) -> str:
        # deterministic contradiction flag (run-3): an exclusion fighting the headline value.
        # Run-5: "mission-critical application" flagged on 'application' - generic product
        # vocabulary is not overlap. Require a NON-GENERIC shared word, or 2+ shared.
        shared = _shared(content_words(item), core_words)
        signal = {w for w in shared if w not in _GENERIC}
        if not signal and len(shared) < 2:
            return ""
        shown = sorted(signal or shared)
        return f"  ⚠ mentions {shown} - confirm this is not the core value"

    if intent.scope_exclusions:
        explicit = [x for x in intent.scope_exclusions if x.explicit]
        candidates = [x for x in intent.scope_exclusions if not x.explicit]
        lines += ["\n**Out of scope**",
                  *([f"- {x.item}{_flag(x.item)}" for x in explicit] or ["- (none stated explicitly)"])]
        if candidates:
            lines += ["\n**Candidate exclusions - inferred, confirm before relying**",
                      *[f"- {x.item}{_flag(x.item)}" for x in candidates]]
    else:
        lines += ["\n**Out of scope**",
                  *([f"- {s}{_flag(s)}" for s in intent.out_of_scope]
                    or ["- (none identified - boundary unverified, press the interviewer next run)"])]
    if intent.external_integrations:
        lines += ["\n## External integrations", *[f"- {s}" for s in intent.external_integrations]]
    if intent.constraints:
        lines += ["\n## Constraints", *[f"- {s}" for s in intent.constraints]]
    if intent.assumption_items:
        lines += ["\n## Assumptions (asserted by you - validated later, not built)",
                  *[f"- {a.statement} (turn {', '.join(str(t) for t in a.source_turns)})"
                    for a in intent.assumption_items]]
    elif intent.assumptions:
        lines += ["\n## Assumptions (asserted by you - validated later, not built)",
                  *[f"- {a}" for a in intent.assumptions]]
    if intent.glossary:
        lines += ["\n## Glossary (your terms, as you used them)",
                  *[f"- **{t}** - {m}" for t, m in intent.glossary.items()]]
    if intent.done_criteria:
        lines += ["\n## Done criteria (your acceptance conditions - checked at delivery, not built as needs)",
                  *[f"- {d}" for d in intent.done_criteria]]
    if intent.operational_concepts:
        lines += ["\n## Operational concepts (people/processes around the system - not requirements)",
                  *[f"- {o}" for o in intent.operational_concepts]]
    if intent.open_questions:
        lines += ["\n## Open questions (never silently resolved)", *[f"- {q}" for q in intent.open_questions]]
    lines += ["\n---",
              "_Approving intent (needs + scope). Verifiable requirements - shall-statements, "
              "functional and non-functional - are derived from this in the spec-generator stage._"]
    return "\n".join(lines)

def render_coverage_markdown(coverage: dict[str, str]) -> str:
    """The interview's coverage map for the approval gate (ADR-0016) - honest
    about what was covered, skipped by the founder, or left thin."""
    if not coverage:
        return ""
    from .coverage import AREA_KEYS, _LABELS
    groups = {"covered": [], "delegated": [], "thin": [], "skipped": [], "deferred": [], "none_stated": [], "uncovered": []}
    for k in AREA_KEYS:
        groups.get(coverage.get(k, "uncovered"), groups["uncovered"]).append(_LABELS[k])
    lines = ["\n## Interview coverage"]
    if groups["covered"]:
        lines.append("**Covered:** " + "; ".join(groups["covered"]))
    if groups["delegated"]:
        lines.append("**Covered by your delegation (recorded as assumptions):** " + "; ".join(groups["delegated"]))
    if groups["thin"]:
        lines.append("**Thin (mentioned, not concrete):** " + "; ".join(groups["thin"]))
    if groups["skipped"]:
        lines.append("**Skipped by you:** " + "; ".join(groups["skipped"]))
    if groups["deferred"]:
        lines.append("**Deferred by you - still open, awaiting your answer:** " + "; ".join(groups["deferred"]))
    if groups["none_stated"]:
        lines.append("**None stated (no external system was mentioned; not asked):** " + "; ".join(groups["none_stated"]))
    if groups["uncovered"]:
        lines.append("**Not covered:** " + "; ".join(groups["uncovered"]))
    return "\n".join(lines)

def render_preview_markdown(preview: list[dict]) -> str:
    """Sample SN -> R derivations for the approval gate (option B, 2026-09-08).
    Presentation-only: the full requirement set derives in spec-generator."""
    if not preview:
        return ""
    lines = ["\n## Preview: how needs become requirements (samples)"]
    for i, p in enumerate(preview, 1):
        kind = p.get("kind", "?")
        lines.append(f"- **{p.get('need_id', 'SN-?')} -> R-p{i}** ({kind}): {p.get('statement')}")
        if p.get("example_criterion"):
            lines.append(f"  - check: {p['example_criterion']}")
    lines.append("_Illustrative only - the full, traceable requirement set is derived in spec-generator._")
    return "\n".join(lines)
