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
from .prompts import DISTILL_SYSTEM, INTERVIEWER_SYSTEM
from .telemetry import UsageLog


class InterviewState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    submission: dict          # IntakeSubmission as dict
    verdict: dict             # ScopeVerdict as dict
    intent_model: dict        # IntentModel as dict, after distillation
    turn_count: int


MAX_TURNS = 60  # hard stop: 30-60 min target; a runaway interview is a cost leak


def build_graph(usage_log: UsageLog, checkpointer=None):
    probing_model = model_config.build_chat_model("probing")
    distill_model = model_config.build_chat_model("distillation")
    tiebreak_model = model_config.build_chat_model("scope_tiebreak")

    def node_scope_gate(state: InterviewState) -> dict:
        if "verdict" in state:  # resuming a thread: never re-pay (or re-roll) the gate
            return {}
        sub = IntakeSubmission.model_validate(state["submission"])
        verdict = scope_gate.verdict_deterministic(sub)
        if verdict is None:
            verdict = scope_gate.verdict_with_llm(sub, tiebreak_model, usage_log)
        return {"verdict": verdict.model_dump()}

    def route_scope(state: InterviewState) -> str:
        return "interview_turn" if state["verdict"]["accepted"] else END

    def node_interview_turn(state: InterviewState) -> dict:
        from langchain_core.messages import AIMessage, SystemMessage

        history = state["messages"]
        resp = probing_model.invoke([SystemMessage(content=INTERVIEWER_SYSTEM), *history])
        usage_log.record("probing", "config", model_config.for_stage("probing").model,
                         (getattr(resp, "usage_metadata", None) or {}))
        return {"messages": [resp], "turn_count": state.get("turn_count", 0) + 1}

    def route_interview(state: InterviewState) -> str:
        last = state["messages"][-1]
        done = "[INTERVIEW COMPLETE]" in getattr(last, "content", "")
        if done or state.get("turn_count", 0) >= MAX_TURNS:
            return "distill_intent"
        return "await_founder"  # interrupt point: founder's reply arrives, graph resumes

    def node_distill(state: InterviewState) -> dict:
        from langchain_core.messages import SystemMessage

        transcript = "\n".join(
            f"[{i}] {getattr(m, 'type', '?')}: {getattr(m, 'content', '')}"
            for i, m in enumerate(state["messages"])
        )
        resp = distill_model.invoke([SystemMessage(content=DISTILL_SYSTEM + "\n\n" + transcript)])
        usage_log.record("distillation", "config", model_config.for_stage("distillation").model,
                         (getattr(resp, "usage_metadata", None) or {}))
        import json as _json
        data = _json.loads(resp.content.strip().strip("`").removeprefix("json").strip())
        data.setdefault("session_id", state["submission"]["submission_id"])
        return {"intent_model": data}

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
