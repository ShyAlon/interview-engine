"""Terminal demo of the interview engine. The web front end (delivery-api)
drives the same graph through a checkpointer; this CLI exists so the loop is
runnable end-to-end before any UI exists.

Usage:  python -m interview_engine.cli "a marketplace for surf lessons in Tel Aviv"
Resume: python -m interview_engine.cli --session <session-id>
        (checkpoints persist to out/<session>/checkpoints.sqlite - a crash
        mid-interview, or even at final validation, loses nothing)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from vmodel_contracts import AppArchetype, IntakeSubmission, IntentModel

from .errors import UserFacingError, run_cli, translate_json_error
from .graph import build_graph
from .progress import ProgressBus, stderr_printer
from .telemetry import UsageLog, jsonl_sink


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
    groups = {"covered": [], "thin": [], "skipped": [], "uncovered": []}
    for k in AREA_KEYS:
        groups.get(coverage.get(k, "uncovered"), groups["uncovered"]).append(_LABELS[k])
    lines = ["\n## Interview coverage"]
    if groups["covered"]:
        lines.append("**Covered:** " + "; ".join(groups["covered"]))
    if groups["thin"]:
        lines.append("**Thin (mentioned, not concrete):** " + "; ".join(groups["thin"]))
    if groups["skipped"]:
        lines.append("**Skipped by you:** " + "; ".join(groups["skipped"]))
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


def _print_usage_table(log) -> None:
    s = log.summary()
    if not s["stages"]:
        return
    print("\nLLM usage (per stage):", file=sys.stderr)
    for stage, st in s["stages"].items():
        print(f"  {stage:<14} {st['calls']:>3} calls  {st['prompt_tokens']:>7} in / "
              f"{st['completion_tokens']:>6} out tok  {st['seconds']:>6.1f}s  ${st['cost_usd']:.4f}  "
              f"({', '.join(st['models'])})", file=sys.stderr)
    print(f"  {'TOTAL':<14} {sum(st['calls'] for st in s['stages'].values()):>3} calls  "
          f"{'':>7} {'':>6} {'':>3} {s['total_seconds']:>6.1f}s  ${s['total_cost_usd']:.4f}",
          file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="*", help="the product idea (omit with --session to resume)")
    ap.add_argument("-s", "--session", help="resume this session id from its checkpoint")
    args = ap.parse_args()

    resuming = args.session is not None
    # human-readable session id: local timestamp + short collision suffix
    session = args.session or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4])
    outdir = Path("out") / session
    db_path = outdir / "checkpoints.sqlite"
    if resuming and not db_path.exists():
        # resuming from a different cwd than the original run is the common
        # miss - say exactly where we looked (absolute) and hunt for the
        # session under sibling project dirs before giving up.
        searched = db_path.resolve()
        siblings = sorted(
            p for p in Path.cwd().parent.glob(f"*/out/{session}/checkpoints.sqlite") if p.exists()
        )
        hint = "typo in the session id, or the run happened before checkpoints were persisted to disk."
        if siblings:
            found = "\n".join(f"  {p}" for p in siblings)
            hint = (f"the session DOES exist, one directory over - the original run happened from "
                    f"a different working directory:\n{found}")
        raise UserFacingError(
            f"no checkpoint found for session {session}. Searched: {searched}",
            hint,
            "cd into the directory you started the interview from and retry the same command; "
            "or run without --session to start a fresh interview.",
        )
    outdir.mkdir(parents=True, exist_ok=True)

    bus = ProgressBus()
    bus.subscribe(stderr_printer())
    log = UsageLog(session, sink=jsonl_sink(outdir / "usage.jsonl"))

    # Disk-backed checkpoints + JSON-native state: a crash mid-interview (or at
    # final validation) resumes with --session; nothing vmodel-specific enters
    # the msgpack serde unregistered (AppArchetype allow-listed as belt).
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[AppArchetype]))
    graph = build_graph(log, checkpointer=checkpointer, progress=bus)
    cfg = {"configurable": {"thread_id": session}}

    if resuming:
        print(f"Resuming session {session} from {db_path.resolve()}", file=sys.stderr)
        current = graph.get_state(cfg)
        if not current.values:
            raise UserFacingError(
                f"session {session} has no saved state.",
                "the checkpoint file exists but is empty.",
                "run without --session to start a fresh interview.",
            )
        finished = "intent_model" in current.values  # crashed AT/AFTER distillation
        if finished:
            print("The interview had already completed; re-rendering the intent model from the checkpoint.",
                  file=sys.stderr)
        state = None  # resume from the checkpoint
    else:
        finished = False
        current = None
        idea = " ".join(args.idea) or input("Describe your product idea: ")
        print(f"session: {session}", file=sys.stderr)
        print(f"(if anything crashes, resume with: python -m interview_engine.cli --session {session})",
              file=sys.stderr)
        print("(interview controls: 'skip' = skip this area, 'enough' = end the interview)\n",
              file=sys.stderr)
        submission = IntakeSubmission(
            free_text=idea, founder_id="cli-founder", submission_id=session
        )
        # mode="json": enums and models enter checkpoints as plain JSON, which
        # keeps the msgpack serde warning-free across langgraph upgrades.
        state = {"messages": [HumanMessage(content=idea)], "submission": submission.model_dump(mode="json")}

    result = current.values if finished else None
    while not finished:
        result = graph.invoke(state, cfg)
        state = None  # further invokes resume from the checkpoint with new input only
        verdict = result.get("verdict")
        if verdict and not verdict["accepted"]:
            print("\n" + verdict["decline_message"])
            return
        if "intent_model" in result:
            break
        last = result["messages"][-1].content
        if "[INTERVIEW COMPLETE]" in last:
            state = {}  # marker already in checkpoint; route onward to distill
            continue
        print(f"\nInterviewer: {last}")
        reply = input("You: ")
        state = {"messages": [HumanMessage(content=reply)]}

    # The graph validated the intent model in-node; write the raw artifact
    # BEFORE re-validating so a future contract bump can never strand a run.
    (outdir / "intent_model.raw.json").write_text(json.dumps(result["intent_model"], indent=2))
    from vmodel_contracts import ContractVersionMismatch, check_compatible

    try:
        check_compatible(result["intent_model"])
    except ContractVersionMismatch as e:
        raise UserFacingError(
            f"this session's intent model is from an older contracts version ({e.have}).",
            "the interview ran before a contract upgrade; the transcript is preserved in the checkpoint.",
            "if the schema is otherwise compatible, re-stamp the raw artifact: "
            "python -m vmodel_contracts.migrate <path-to-intent_model.raw.json> (validates first, refuses on real drift); "
            "otherwise start a fresh session (the interview is the cheap part to redo).",
        ) from e
    try:
        intent = IntentModel.model_validate(result["intent_model"])
    except Exception as e:
        raise UserFacingError(
            "the distillation produced an intent model that fails the contract schema.",
            f"validation detail: {type(e).__name__}",
            f"the raw model is preserved at {outdir}/intent_model.raw.json and the interview is checkpointed - "
            f"rerun with --session {session}.",
        ) from e
    (outdir / "intent_model.json").write_text(intent.model_dump_json(indent=2))
    md = render_intent_markdown(intent)
    md += "\n" + render_coverage_markdown(result.get("coverage") or {})
    preview = result.get("requirement_preview") or []
    if preview:
        (outdir / "requirement_preview.json").write_text(json.dumps(preview, indent=2))
        md += "\n" + render_preview_markdown(preview)
    (outdir / "intent_model.md").write_text(md)

    # The review gate: cheapest place to catch a misunderstanding (scoping doc, step 2).
    print("\n" + md)
    ok = input("\nApprove this intent model? [y/N] ").strip().lower()
    if ok != "y":
        print("Rejected - nothing generated. The interview can resume with corrections.")
        return
    log.write_summary(outdir / "usage_summary.json")
    _print_usage_table(log)
    print(f"\nApproved. Artifacts in {outdir.resolve()}/")
    print("Next: hand intent_model.json to spec-generator.")


if __name__ == "__main__":
    run_cli(main)
