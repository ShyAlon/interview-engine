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
from .coverage import (apply_reply, founder_control, initial_coverage, next_area,
                         progress_line, status_block)
from .progress import ProgressBus, ProgressEvent, track
from .prompts import INTERVIEWER_SYSTEM
from .telemetry import UsageLog


class InterviewState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    submission: dict          # IntakeSubmission as dict
    verdict: dict             # ScopeVerdict as dict
    intent_model: dict        # IntentModel as dict, after distillation
    requirement_preview: list  # sample derived requirements for the approval gate (presentation-only)
    coverage: dict[str, str]  # area -> uncovered|thin|covered|skipped (ADR-0016)
    coverage_target: str      # the area the last question aimed at ("skip" applies to it)
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

        coverage = dict(state.get("coverage") or initial_coverage())
        turn = state.get("turn_count", 0) + 1

        # founder control (ADR-0016): deterministic, never costs a model call
        control = founder_control(getattr(last, "content", "")) if last is not None else None
        if control == "enough":
            return {"messages": [AIMessage(content="Understood - wrapping up.\n\n[INTERVIEW COMPLETE]")],
                    "coverage": coverage, "turn_count": turn}
        if control == "skip":
            target = state.get("coverage_target") or next_area(coverage)
            if target:
                coverage[target] = "skipped"

        target = next_area(coverage)
        with track(progress, "interview_turn", "Interviewing", f"turn {turn}"):
            history = state["messages"]
            system = INTERVIEWER_SYSTEM + "\n\n" + status_block(coverage)
            resp = model_config.invoke_checked(probing_model, [SystemMessage(content=system), *history], "probing")
            usage_log.record("probing", model_config.for_stage("probing").provider, model_config.for_stage("probing").model,
                             (getattr(resp, "usage_metadata", None) or {}))

        # structured reply (option a): {question, covered, thin, skipped, complete}.
        # Parse failure degrades gracefully: the text is the question, coverage stands.
        import json as _json
        content = resp.content.strip().strip("`").removeprefix("json").strip()
        question, complete = content, False
        try:
            parsed = _json.loads(content)
            if isinstance(parsed, dict) and isinstance(parsed.get("question"), str):
                coverage = apply_reply(coverage, parsed)
                question = parsed["question"].strip()
                complete = bool(parsed.get("complete"))
            elif progress is not None:
                import time as _t
                progress.emit(ProgressEvent("interview_turn", "Coverage", "completed", _t.time(), 0.0,
                                            "unstructured reply - coverage not updated this turn"))
        except _json.JSONDecodeError:
            if progress is not None:
                import time as _t
                progress.emit(ProgressEvent("interview_turn", "Coverage", "completed", _t.time(), 0.0,
                                            "unstructured reply - coverage not updated this turn"))
        if complete:
            question += "\n\n[INTERVIEW COMPLETE]"
        if progress is not None:
            import time as _t
            progress.emit(ProgressEvent("interview_turn", "Coverage", "completed", _t.time(), 0.0,
                                        progress_line(coverage)))
        return {"messages": [AIMessage(content=question)], "coverage": coverage,
                "coverage_target": target or "", "turn_count": turn}

    def route_interview(state: InterviewState) -> str:
        last = state["messages"][-1]
        done = "[INTERVIEW COMPLETE]" in getattr(last, "content", "")
        if done or state.get("turn_count", 0) >= MAX_TURNS:
            return "distill_intent"
        return "await_founder"  # interrupt point: founder's reply arrives, graph resumes

    def node_distill(state: InterviewState) -> dict:
        """PHASED distillation (ADR-0011): needs -> scope -> preview -> assemble/validate.
        Four visible steps instead of one black box; each LLM phase is small enough
        for a local 14B, validated on its own, and retried once with schema feedback.
        The preview is presentation-only (graceful degrade); needs/scope/assembly are
        the system of record and fail loudly."""
        import json as _json

        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import ValidationError
        from vmodel_contracts import IntentModel, IntentNeed

        from .distill_qa import dedup_constraints, ground_assumptions, is_solution_language
        from .errors import UserFacingError, translate_json_error
        from .progress import ProgressEvent
        from .prompts import DISTILL_NEEDS_SYSTEM, DISTILL_PREVIEW_SYSTEM, DISTILL_SCOPE_SYSTEM
        from .repair import coerce_for_model, normalize_enums

        transcript = "\n".join(
            f"[{i}] {getattr(m, 'type', '?')}: {getattr(m, 'content', '')}"
            for i, m in enumerate(state["messages"])
        )

        def _call(system: str, extra: str, feedback: str | None = None):
            messages = [SystemMessage(content=system + "\n\n" + extra)]
            if feedback:
                messages.append(HumanMessage(content=feedback))
            resp = model_config.invoke_checked(distill_model, messages, "distillation")
            usage_log.record("distillation", model_config.for_stage("distillation").provider,
                             model_config.for_stage("distillation").model,
                             (getattr(resp, "usage_metadata", None) or {}))
            return resp, _json.loads(resp.content.strip().strip("`").removeprefix("json").strip())

        def _phase(stage, label, detail, system, extra, validate):
            """One LLM phase: call -> repair -> validate -> ONE retry with the error as feedback."""
            last_err: Exception | None = None
            with track(progress, stage, label, detail):
                fb = None
                for attempt in (1, 2):
                    try:
                        resp, data = _call(system, extra, fb)
                        if not isinstance(data, dict):
                            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
                        return validate(coerce_for_model(data, IntentModel))
                    except _json.JSONDecodeError as e:
                        raise translate_json_error(e, stage) from e
                    except (ValidationError, ValueError, KeyError, TypeError) as e:
                        last_err = e
                        if attempt == 2:
                            raise UserFacingError(
                                f"distillation phase '{label}' produced output that fails validation, and the retry did not fix it.",
                                f"validation detail: {str(e)[:300]}",
                                "rerun (the interview resumes from its checkpoint) - or route distillation to a stronger model: "
                                "INTERVIEW_ENGINE_MODEL_DISTILLATION=...",
                            ) from e
                        if progress is not None:
                            import time as _t
                            progress.emit(ProgressEvent(stage, label, "started", _t.time(), None,
                                                        "retrying with schema feedback"))
                        fb = ("Your previous output failed validation:\n" + str(e)[:400]
                              + "\nReturn the corrected JSON only, matching the schema exactly.")

        def _valid_needs(data):
            needs = data.get("stakeholder_needs") or []
            if not needs:
                raise ValueError("stakeholder_needs must not be empty - an intent model with no needs is not an intent model")
            fixed = normalize_enums({"stakeholder_needs": needs}, IntentModel)["stakeholder_needs"]
            validated = [IntentNeed.model_validate(n) for n in fixed]
            from vmodel_contracts import Assumption
            items = data.get("assumption_items") or []
            return {"stakeholder_needs": [n.model_dump(mode="json") for n in validated],
                    "assumption_items": [Assumption.model_validate(a).model_dump(mode="json") for a in items],
                    "done_criteria": [str(d) for d in (data.get("done_criteria") or [])],
                    "operational_concepts": [str(o) for o in (data.get("operational_concepts") or [])],
                    "assumptions": data.get("assumptions") or []}

        def _valid_scope(data):
            required_str = ("problem_statement", "core_value_proposition")
            required_list = ("target_users", "in_scope", "out_of_scope")
            for k in required_str:
                if not isinstance(data.get(k), str) or not data[k].strip():
                    raise ValueError(f"{k} must be a non-empty string")
            for k in required_list:
                if not isinstance(data.get(k), list):
                    raise ValueError(f"{k} must be a list")
            if not data["out_of_scope"]:
                raise ValueError("out_of_scope is empty - derive at least one exclusion from the conversation")
            from vmodel_contracts import ScopeExclusion
            exclusions = [ScopeExclusion.model_validate(x).model_dump(mode="json")
                          for x in (data.get("scope_exclusions") or [])]
            keep = required_str + required_list + ("external_integrations", "constraints", "open_questions", "glossary", "source_turns")
            out = {k: data.get(k) for k in keep}
            out["scope_exclusions"] = exclusions
            return out

        parts = _phase("distill_needs", "Extracting stakeholder needs", "phase 1/4",
                       DISTILL_NEEDS_SYSTEM, "TRANSCRIPT:\n" + transcript, _valid_needs)
        parts.update(_phase("distill_scope", "Building scope and framing", "phase 2/4",
                            DISTILL_SCOPE_SYSTEM, "TRANSCRIPT:\n" + transcript, _valid_scope))
        parts["session_id"] = state["submission"]["submission_id"]

        # phase 3: requirement PREVIEW (presentation-only; never blocks the run)
        preview: list[dict] = []
        with track(progress, "distill_preview", "Drafting requirement preview", "phase 3/4"):
            try:
                _, raw = _call(DISTILL_PREVIEW_SYSTEM, "NEEDS:\n" + _json.dumps(parts["stakeholder_needs"]))
                if isinstance(raw, list):
                    preview = [p for p in raw
                               if isinstance(p, dict) and isinstance(p.get("statement"), str)][:3]
                    for p in preview:
                        if p.get("example_criterion") and is_solution_language(str(p["example_criterion"])):
                            p.pop("example_criterion")  # implementation mechanics are not a check (method pack 3)
            except Exception:  # noqa: BLE001 - decoration, not the system of record
                preview = []
        if not preview and progress is not None:
            import time as _t
            progress.emit(ProgressEvent("distill_preview", "Requirement preview", "completed", _t.time(), 0.0,
                                        "unavailable from this model - the full set derives in spec-generator"))

        # phase 4: deterministic assembly + full contract validation
        with track(progress, "distill_validate", "Validating intent model", "phase 4/4"):
            # mechanical QA (ADR-0015): grounding and dedup are ENFORCED, not prompted.
            import time as _t2
            turns = [str(getattr(m, "content", "")) for m in state["messages"]]
            grounded, dropped_a = ground_assumptions(
                parts.get("assumption_items") or [], turns,
                [DISTILL_NEEDS_SYSTEM, DISTILL_SCOPE_SYSTEM])
            parts["assumption_items"] = grounded
            # legacy flat assumptions from older prompts have no provenance - drop them too.
            if parts.get("assumptions"):
                dropped_a += [f"{a[:60]} (legacy ungrounded field)" for a in parts["assumptions"]]
                parts["assumptions"] = []
            kept_c, dropped_c = dedup_constraints(
                parts.get("constraints") or [],
                [n["statement"] for n in parts["stakeholder_needs"]])
            parts["constraints"] = kept_c
            if progress is not None:
                for d in dropped_a:
                    progress.emit(ProgressEvent("distill_validate", "Dropped ungrounded assumption",
                                                "completed", _t2.time(), 0.0, d))
                for d in dropped_c:
                    progress.emit(ProgressEvent("distill_validate", "Dropped duplicate constraint",
                                                "completed", _t2.time(), 0.0,
                                                d[:70] + " (already a classified need)"))
            data = normalize_enums(coerce_for_model(parts, IntentModel), IntentModel)
            try:
                intent = IntentModel.model_validate(data)
            except ValidationError as e:
                raise UserFacingError(
                    "the assembled intent model fails the contract schema.",
                    f"validation detail: {str(e)[:300]}",
                    "rerun (the interview resumes from its checkpoint) - or route distillation to a stronger model.",
                ) from e
        return {"intent_model": intent.model_dump(mode="json"), "requirement_preview": preview}

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
