"""The interview as a LangGraph state machine.

Why LangGraph here (ADR-0003): the interview is stateful, resumable, branches
(scope gate, review gate), and has a genuine human-in-the-loop pause. That is
exactly what a graph runtime is for. Deterministic classification stays plain
code called from a node - no orchestration ceremony around a function call.

Flow:
  intake -> scope_gate --declined--> END (polite exit)
                       --accepted--> interview_turn <--> (loop until complete)
                                   -> distill_intent -> founder_review (interrupt)
                                                     -> END (IntentModel emitted)
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from vmodel_contracts import IntakeSubmission, ScopeVerdict

from . import model_config, scope_gate
from .progress import ProgressBus, track
from .prompts import DISTILL_SYSTEM, INTERVIEWER_SYSTEM
from .telemetry import UsageLog


class InterviewState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    submission: dict          # IntakeSubmission as dict
    verdict: dict             # ScopeVerdict as dict
    intent_model: dict        # IntentModel as dict, after distillation
    turn_count: int


MAX_TURNS = 60  # hard stop: 30-60 min target; a runaway interview is a cost leak


def build_graph(usage_log: UsageLog, checkpointer=None, progress: ProgressBus | None = None):
    probing_model = model_config.build_chat_model("probing")
    distill_model = model_config.build_chat_model("distillation")
    tiebreak_model = model_config.build_chat_model("scope_tiebreak")

    def node_scope_gate(state: InterviewState) -> dict:
        if "verdict" in state:  # resuming a thread: never re-pay (or re-roll) the gate
            return {}
        with track(progress, "scope_gate", "Classifying scope"):
            sub = IntakeSubmission.model_validate(state["submission"])
            verdict = scope_gate.verdict_deterministic(sub)
            if verdict is None:
                verdict = scope_gate.verdict_with_llm(sub, tiebreak_model, usage_log)
        return {"verdict": verdict.model_dump(mode="json")}

    def route_scope(state: InterviewState) -> str:
        return "interview_turn" if state["verdict"]["accepted"] else END

    def node_interview_turn(state: InterviewState) -> dict:
        from langchain_core.messages import AIMessage, SystemMessage

        last = state["messages"][-1] if state.get("messages") else None
        if last is not None and "[INTERVIEW COMPLETE]" in getattr(last, "content", ""):
            return {}  # marker already emitted: let routing reach distill without burning a call

        turn = state.get("turn_count", 0) + 1
        with track(progress, "interview_turn", "Interviewing", f"turn {turn}"):
            history = state["messages"]
            resp = model_config.invoke_checked(probing_model, [SystemMessage(content=INTERVIEWER_SYSTEM), *history], "probing")
            usage_log.record("probing", model_config.for_stage("probing").provider, model_config.for_stage("probing").model,
                             (getattr(resp, "usage_metadata", None) or {}))
        return {"messages": [resp], "turn_count": turn}

    def route_interview(state: InterviewState) -> str:
        last = state["messages"][-1]
        done = "[INTERVIEW COMPLETE]" in getattr(last, "content", "")
        if done or state.get("turn_count", 0) >= MAX_TURNS:
            return "distill_intent"
        return "await_founder"  # interrupt point: founder's reply arrives, graph resumes

    def node_distill(state: InterviewState) -> dict:
        """Validate INSIDE the node: a contract-invalid intent model never
        leaves the graph. Shape drift is repaired mechanically (repair.py);
        anything else gets ONE retry with the validation error fed back to
        the model, then a human-readable failure. Root cause of the first
        field crash: a small model emitted `source_turns: {}`-style drift and
        validation only happened later in the CLI, outside any retry loop."""
        import json as _json

        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import ValidationError
        from vmodel_contracts import IntentModel

        from .errors import UserFacingError, translate_json_error
        from .repair import coerce_for_model

        transcript = "\n".join(
            f"[{i}] {getattr(m, 'type', '?')}: {getattr(m, 'content', '')}"
            for i, m in enumerate(state["messages"])
        )
        messages = [SystemMessage(content=DISTILL_SYSTEM + "\n\n" + transcript)]
        last_err: Exception | None = None
        with track(progress, "distill_intent", "Distilling intent model"):
            for attempt in (1, 2):  # first try, then ONE retry with the validation error as feedback
                resp = model_config.invoke_checked(distill_model, messages, "distillation")
                usage_log.record("distillation", model_config.for_stage("distillation").provider,
                                 model_config.for_stage("distillation").model,
                                 (getattr(resp, "usage_metadata", None) or {}))
                try:
                    data = _json.loads(resp.content.strip().strip("`").removeprefix("json").strip())
                except _json.JSONDecodeError as e:
                    raise translate_json_error(e, "distillation") from e
                data.setdefault("session_id", state["submission"]["submission_id"])
                data = coerce_for_model(data, IntentModel)
                try:
                    intent = IntentModel.model_validate(data)
                    return {"intent_model": intent.model_dump(mode="json")}
                except ValidationError as e:
                    last_err = e
                    if attempt == 2:
                        raise UserFacingError(
                            "the distillation produced an intent model that fails the contract schema, and the retry did not fix it.",
                            f"validation detail: {last_err}",
                            "rerun (the interview resumes from its checkpoint) - or route distillation to a stronger model: "
                            "INTERVIEW_ENGINE_MODEL_DISTILLATION=... The intent model is the system of record; it must validate.",
                        ) from e
                    if attempt == 1:
                        if progress is not None:
                            from .progress import ProgressEvent
                            import time as _t
                            progress.emit(ProgressEvent("distill_intent", "Distilling intent model",
                                                        "started", _t.time(), None,
                                                        "retrying with schema feedback"))
                        messages = [*messages, resp, HumanMessage(content=(
                            "Your previous output failed contract validation:\n"
                            + "\n".join(f"- {err['loc']}: {err['msg']}" for err in e.errors()[:8])
                            + "\nReturn the corrected JSON only, matching the schema exactly."
                        ))]

    g = StateGraph(InterviewState)
    g.add_node("scope_gate", node_scope_gate)
    g.add_node("interview_turn", node_interview_turn)
    g.add_node("distill_intent", node_distill)
    g.set_entry_point("scope_gate")
    g.add_conditional_edges("scope_gate", route_scope, {"interview_turn": "interview_turn", END: END})
    g.add_conditional_edges("interview_turn", route_interview,
                            {"interview_turn": "interview_turn", "await_founder": END, "distill_intent": "distill_intent"})
    g.add_edge("distill_intent", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
