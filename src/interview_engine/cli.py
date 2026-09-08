"""Terminal demo of the interview engine. The web front end (delivery-api)
drives the same graph through a checkpointer; this CLI exists so the loop is
runnable end-to-end before any UI exists.

Usage:  python -m interview_engine.cli "a marketplace for surf lessons in Tel Aviv"
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from vmodel_contracts import IntakeSubmission, IntentModel

from .errors import UserFacingError, run_cli, translate_json_error
from .graph import build_graph
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
    idea = " ".join(sys.argv[1:]) or input("Describe your product idea: ")
    session = str(uuid.uuid4())
    outdir = Path("out") / session
    outdir.mkdir(parents=True, exist_ok=True)

    log = UsageLog(session, sink=jsonl_sink(outdir / "usage.jsonl"))
    graph = build_graph(log)
    cfg = {"configurable": {"thread_id": session}}

    submission = IntakeSubmission(
        free_text=idea, founder_id="cli-founder", submission_id=session
    )
    state = {"messages": [HumanMessage(content=idea)], "submission": submission.model_dump()}

    while True:
        result = graph.invoke(state, cfg)
        verdict = result.get("verdict")
        if verdict and not verdict["accepted"]:
            print("\n" + verdict["decline_message"])
            return
        if "intent_model" in result:
            break
        last = result["messages"][-1].content
        if "[INTERVIEW COMPLETE]" in last:
            state = result
            continue  # let routing reach distill on next invoke
        print(f"\nInterviewer: {last}")
        reply = input("You: ")
        state = {**result, "messages": [HumanMessage(content=reply)]}

    try:
        intent = IntentModel.model_validate(result["intent_model"])
    except Exception as e:
        raise UserFacingError(
            "the distillation produced an intent model that fails the contract schema.",
            f"validation detail: {type(e).__name__}",
            "retry - or route distillation to a stronger model (the intent model is the system of record).",
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
