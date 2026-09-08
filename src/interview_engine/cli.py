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
    """Human-readable rendering for the founder review gate (scoping doc, step 2)."""
    lines = [
        f"# What we understood\n",
        f"**Problem.** {intent.problem_statement}\n",
        f"**For.** {', '.join(intent.target_users)}\n",
        f"**Core value.** {intent.core_value_proposition}\n",
        "\n**In scope**", *[f"- {s}" for s in intent.in_scope],
        "\n**Out of scope**", *[f"- {s}" for s in intent.out_of_scope],
    ]
    if intent.external_integrations:
        lines += ["\n**Integrations**", *[f"- {s}" for s in intent.external_integrations]]
    if intent.constraints:
        lines += ["\n**Constraints**", *[f"- {s}" for s in intent.constraints]]
    if intent.open_questions:
        lines += ["\n**Open questions (never silently resolved)**", *[f"- {q}" for q in intent.open_questions]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", nargs="*", help="the product idea (omit with --session to resume)")
    ap.add_argument("-s", "--session", help="resume this session id from its checkpoint")
    args = ap.parse_args()

    resuming = args.session is not None
    session = args.session or str(uuid.uuid4())
    outdir = Path("out") / session
    outdir.mkdir(parents=True, exist_ok=True)
    db_path = outdir / "checkpoints.sqlite"
    if resuming and not db_path.exists():
        raise UserFacingError(
            f"no checkpoint found for session {session} ({db_path} does not exist).",
            "typo in the session id, or the run happened before checkpoints were persisted to disk.",
            "run without --session to start a fresh interview.",
        )

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
        print(f"Resuming session {session} from checkpoint.", file=sys.stderr)
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
        print(f"(if anything crashes, resume with: python -m interview_engine.cli --session {session})\n",
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
    (outdir / "intent_model.md").write_text(md)

    # The review gate: cheapest place to catch a misunderstanding (scoping doc, step 2).
    print("\n" + md)
    ok = input("\nApprove this intent model? [y/N] ").strip().lower()
    if ok != "y":
        print("Rejected - nothing generated. The interview can resume with corrections.")
        return
    print(f"\nApproved. Artifacts in {outdir}/  |  LLM cost so far: ${log.total_usd():.4f}")
    print("Next: hand intent_model.json to spec-generator.")


if __name__ == "__main__":
    run_cli(main)
