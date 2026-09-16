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

import time as _time

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from vmodel_contracts import IntakeSubmission, ScopeVerdict

from . import model_config, scope_gate
from .coverage import (COVERED, THIN, SKIPPED, _LABELS, apply_reply, extract_turn_json, founder_control,
                         founder_steer, guard_question, initial_coverage, next_area, nfr_quantified, unique_question, same_question_substance, engineering_default, DELEGATED,
                         progress_line, question_topic, status_block, turn_output_schema)
from .progress import ProgressBus, ProgressEvent, track
from .prompts import INTERVIEWER_SYSTEM, STRUCTURED_TURN_FORMAT
from .telemetry import UsageLog


class InterviewState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    submission: dict          # IntakeSubmission as dict
    verdict: dict             # ScopeVerdict as dict
    intent_model: dict        # IntentModel as dict, after distillation
    conops_draft: dict        # validated context/scenarios/options, awaiting founder choice
    conops: dict              # founder-confirmed ConOpsDocument
    distill_warnings: list[str]  # non-blocking flags for the approval gate (e.g. unrecognized ecosystem name)
    requirement_preview: list  # sample derived requirements for the approval gate (presentation-only)
    coverage: dict[str, str]  # area -> uncovered|thin|covered|skipped (ADR-0016)
    coverage_target: str      # the area the last question aimed at ("skip" applies to it)
    parked_topics: list[str]  # founder-deferred sub-topics that must never be re-probed
    delegated_decisions: list[dict]  # founder-delegated choices, captured as research briefs
    question_history: list[dict]  # code-owned emitted questions: {area, question}
    turn_count: int


MAX_TURNS = 60  # hard stop: 30-60 min target; a runaway interview is a cost leak


def build_graph(usage_log: UsageLog, checkpointer=None, progress: ProgressBus | None = None, *, conops_enabled: bool = False, raw_output_dir=None):
    from .raw_output import RawOutputLog
    raw_log = RawOutputLog(raw_output_dir)
    probing_model = model_config.build_chat_model("probing", schema=turn_output_schema())
    distill_model = model_config.build_chat_model("distillation")
    conops_model = model_config.build_chat_model("conops") if conops_enabled else None
    tiebreak_model = model_config.build_chat_model("scope_tiebreak")

    def node_scope_gate(state: InterviewState) -> dict:
        if "verdict" in state:  # resuming a thread: never re-pay (or re-roll) the gate
            return {}
        with track(progress, "scope_gate", "Classifying scope"):
            sub = IntakeSubmission.model_validate(state["submission"])
            verdict = scope_gate.verdict_deterministic(sub)
            if verdict is None:
                verdict = scope_gate.verdict_with_llm(sub, tiebreak_model, usage_log, raw_log)
        return {"verdict": verdict.model_dump(mode="json")}

    def route_scope(state: InterviewState) -> str:
        if not state["verdict"]["accepted"]:
            return END
        if conops_enabled and state.get("intent_model") and not state.get("conops"):
            return "conops"
        return "interview_turn"

    def node_interview_turn(state: InterviewState) -> dict:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if "intent_model" in state:
            return {}  # already distilled: nothing left to do
        # marker ANYWHERE in AI history short-circuits, not only as the last
        # message: a founder message arriving after completion must not
        # re-open a saturated interview (2026-09-12 double-distillation)
        if any(getattr(m, "type", "") == "ai" and "[INTERVIEW COMPLETE]" in getattr(m, "content", "")
               for m in state.get("messages", [])):
            return {}

        last = state["messages"][-1] if state.get("messages") else None

        coverage = dict(state.get("coverage") or initial_coverage())
        parked_topics = list(state.get("parked_topics") or [])
        turn = state.get("turn_count", 0) + 1

        # Founder steers are terminal (2026-09-12 transcript: 'No. You do it.'
        # got rephrased and re-asked twice). 'delegate' parks the sub-topic and
        # records the delegation for the intent model; 'dismiss' closes the
        # whole target area. Neither may be rephrased and re-asked.
        steer = (founder_steer(getattr(last, "content", ""))
                 if last is not None and getattr(last, "type", "") == "human" else None)
        delegated = [dict(d) for d in (state.get("delegated_decisions") or [])]
        if steer:
            prior_question = next((getattr(m, "content", "") for m in reversed(state["messages"][:-1])
                                   if getattr(m, "type", "") == "ai"), "")
            parked = question_topic(prior_question)
            if parked and parked not in parked_topics:
                parked_topics.append(parked)
            area_label = state.get("coverage_target") or next_area(coverage) or "the current topic"
            words = getattr(last, "content", "").strip()[:120]
            turn_idx = len(state["messages"]) - 1  # the founder message's transcript index
            topic = parked or area_label
            delegated.append({
                "statement": f"FOUNDER-DELEGATED: {topic} - the founder explicitly delegated this "
                             f"decision ('{words}'); the spec/architecture stage owns the choice.",
                "research_brief": f"Research {topic} using current registries, vendor documentation, and web sources; propose a requirement-specific choice for founder review.",
                "source_turns": [turn_idx]})
            # Explicit delegation or dismissal resolves the current area as
            # covered-by-delegation. The assumption remains visible in the
            # intent artifact; this is neither a non-answer nor silent coverage.
            if area_label in coverage:
                coverage[area_label] = DELEGATED

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
        # Two unanswered attempts are enough. Close the stubborn area as a
        # visible assumption and move on rather than phrasing the same ask a
        # third way. This is code-owned and survives model drift.
        # Count only the trailing run for this target. An old question from an
        # earlier visit must not consume today's budget; conversely, two
        # consecutive questions with no coverage advance must close the thread.
        attempts = 0
        for asked in reversed(state.get("question_history") or []):
            if asked.get("area") != target:
                break
            attempts += 1
        if target and attempts >= 2 and coverage.get(target) in ("uncovered", "thin"):
            prior = [q.get("question", "") for q in (state.get("question_history") or [])[-attempts:]]
            # The current founder turn is the evidence for deferring this
            # unresolved detail. Keep the question history in graph state; the
            # contract assumption carries only its statement and real source.
            source_turn = len(state.get("messages") or []) - 1
            delegated.append({
                "statement": f"FOUNDER-DELEGATED: {_LABELS.get(target, target)} - after two interview attempts, "
                             "the remaining detail is left open for the spec/architecture stage instead of being re-asked. "
                             f"The founder's latest response was: '{getattr(last, 'content', '').strip()[:120]}'.",
                "research_brief": f"Research the unresolved {_LABELS.get(target, target)} question from current authoritative sources and present the method or choice at the review gate.",
                "source_turns": [source_turn]})
            coverage[target] = DELEGATED
            target = next_area(coverage)
        with track(progress, "interview_turn", "Interviewing", f"turn {turn}"):
            history = state["messages"]
            parked_block = ("\n\nPARKED SUB-TOPICS (founder deferred; never ask these again): "
                            + ", ".join(parked_topics)) if parked_topics else ""
            system = INTERVIEWER_SYSTEM + "\n\n" + status_block(coverage) + parked_block + "\n\n" + STRUCTURED_TURN_FORMAT
            _t0 = _time.time()
            resp = model_config.invoke_checked(probing_model, [SystemMessage(content=system), *history], "probing")
            usage_log.record("probing", model_config.for_stage("probing").provider, model_config.for_stage("probing").model,
                             (getattr(resp, "usage_metadata", None) or {}), duration_s=_time.time() - _t0)

        # structured reply (option a): {question, covered, thin, skipped, complete}.
        # Tolerant parse (fences/prose-wrapped JSON); on failure ONE retry with
        # feedback, then graceful degrade (run-4: the 14B ignored the format).
        import time as _t
        # terminal plain-text marker (older checkpoints / models): no parse, no retry
        if "[INTERVIEW COMPLETE]" in resp.content:
            return {"messages": [AIMessage(content=resp.content)], "coverage": coverage,
                    "coverage_target": target or "", "turn_count": turn}
        parsed = extract_turn_json(resp.content)
        if parsed is None:
            model_config.log_malformed_response("probing", "prose/non-JSON", resp, snippet=resp.content)
            if progress is not None:
                progress.emit(ProgressEvent("interview_turn", "Coverage", "started", _t.time(), None,
                                            "unstructured reply - retrying with format feedback"))
            retry_msgs = [SystemMessage(content=system), *history,
                          AIMessage(content=resp.content),
                          HumanMessage(content="Your reply was not valid JSON. Reply with ONLY the JSON object, "
                                               "no prose: " + STRUCTURED_TURN_FORMAT)]
            _t0 = _time.time()
            resp2 = model_config.invoke_checked(probing_model, retry_msgs, "probing")
            usage_log.record("probing", model_config.for_stage("probing").provider,
                             model_config.for_stage("probing").model,
                             (getattr(resp2, "usage_metadata", None) or {}), duration_s=_time.time() - _t0)
            parsed = extract_turn_json(resp2.content)
            if parsed is None:
                model_config.log_malformed_response("probing", "prose/non-JSON", resp2, snippet=resp2.content)
                from .errors import UserFacingError

                cfg = model_config.for_stage("probing")
                if cfg.provider == "ollama":
                    next_action = ("rerun with the same --session id; the checkpoint preserves prior answers. "
                                   "Confirm the server started with OLLAMA_CONTEXT_LENGTH=32768; if it repeats, "
                                   "route probing to DeepSeek with vmodel-spec-app/scripts/switch-model.sh deepseek.")
                else:
                    next_action = ("rerun with the same --session id; the checkpoint preserves prior answers. "
                                   f"If it repeats, check {cfg.provider}'s status and the configured model "
                                   f"({cfg.model}).")
                raise UserFacingError(
                    "stage 'probing': the model ignored the structured interview-turn contract twice.",
                    f"provider returned prose instead of the required JSON object; second reply began: {resp2.content.strip()[:160]!r}.",
                    next_action,
                )
        question, complete = resp.content, False
        if parsed is not None:
            coverage = apply_reply(coverage, parsed)
            if control == "skip" and target:
                coverage[target] = SKIPPED
            # Quantified-NFR guard (run-5): the model may not close "quantified
            # NFRs" while the founder's own words contain no numbers - unless the
            # founder explicitly said there are none (demo mode). Coded guardrail,
            # not prompt discipline.
            if coverage.get("nfrs") == COVERED and (parsed.get("covered") or []) and "nfrs" in (parsed.get("covered") or []):
                founder_text = "\n".join(getattr(m, "content", "") for m in history
                                          if getattr(m, "type", "") == "human")
                has_engineering_default = any(
                    "engineering-owned provisional default" in d.get("statement", "")
                    for d in delegated)
                if not nfr_quantified(founder_text) and not has_engineering_default:
                    coverage["nfrs"] = THIN
                    if progress is not None:
                        progress.emit(ProgressEvent(
                            "interview_turn", "Coverage", "completed", _t.time(), 0.0,
                            "NFRs claimed covered with no numbers given - holding thin; "
                            "say 'no numbers for the demo' to close it honestly"))
            next_target = next_area(coverage)
            candidate = parsed["question"].strip()
            owned = engineering_default(candidate)
            if owned:
                topic, default = owned
                source_turn = max(0, len(history) - 1)
                delegated.append({
                    "statement": f"FOUNDER-DELEGATED: {topic} - engineering-owned provisional default: "
                                 f"{default}. This remains founder-overridable.",
                    "research_brief": f"Validate the provisional {topic} against current industry guidance and "
                                      "map the founder's business-impact answer to measurable acceptance criteria.",
                    "source_turns": [source_turn],
                })
                if topic not in parked_topics:
                    parked_topics.append(topic)
            question = guard_question(candidate, parked_topics, next_target)
            question = unique_question(question, list(state.get("question_history") or []), next_target)
            # saturation guard (run-4 audit): the model may only declare the
            # interview complete when the coverage map actually is saturated -
            # self-grading cannot jump 0/11 -> done.
            # Saturation is CODE-OWNED (2026-09-12, session bd17: coverage hit
            # 11/11 and phi4 kept asking for three more turns because it never
            # set complete=true). A saturated map ends the interview whether or
            # not the model notices; the model's complete flag is a hint, not
            # the gate.
            complete = next_area(coverage) is None
            if parsed.get("complete") and not complete and progress is not None:
                open_areas = sum(1 for v in coverage.values() if v in ("uncovered", "thin"))
                progress.emit(ProgressEvent("interview_turn", "Coverage", "completed", _t.time(), 0.0,
                                            f"model tried to close with {open_areas} areas open - continuing"))
        if complete:
            question += "\n\n[INTERVIEW COMPLETE]"
        if progress is not None:
            import time as _t
            progress.emit(ProgressEvent("interview_turn", "Coverage", "completed", _t.time(), 0.0,
                                        progress_line(coverage)))
        history_log = list(state.get("question_history") or [])
        emitted_target = target
        history_log.append({"area": emitted_target or "", "question": question})
        return {"messages": [AIMessage(content=question)], "coverage": coverage,
                "coverage_target": emitted_target or "", "parked_topics": parked_topics,
                "delegated_decisions": delegated, "question_history": history_log, "turn_count": turn}

    def route_interview(state: InterviewState) -> str:
        last = state["messages"][-1]
        done = "[INTERVIEW COMPLETE]" in getattr(last, "content", "")
        if done or state.get("turn_count", 0) >= MAX_TURNS:
            return "distill_intent"
        return "await_founder"  # interrupt point: founder's reply arrives, graph resumes

    def node_distill(state: InterviewState) -> dict:
        if "intent_model" in state:  # distilled already - never re-pay
            return {}
        """PHASED distillation (ADR-0011): needs -> scope -> preview -> assemble/validate.
        Four visible steps instead of one black box; each LLM phase is small enough
        for a local 14B, validated on its own, and retried once with schema feedback.
        The preview is presentation-only (graceful degrade); needs/scope/assembly are
        the system of record and fail loudly."""
        import json as _json

        from langchain_core.messages import HumanMessage, SystemMessage
        from pydantic import ValidationError
        from vmodel_contracts import IntentModel, IntentNeed

        from .distill_qa import dedup_constraints, ground_assumptions, grounded_in, is_solution_language
        from .errors import UserFacingError, translate_json_error
        from .progress import ProgressEvent
        from .prompts import DISTILL_NEEDS_SYSTEM, DISTILL_PREVIEW_SYSTEM, DISTILL_SCOPE_SYSTEM
        from .repair import coerce_for_model, extract_json_value, normalize_enums

        transcript = "\n".join(
            f"[{i}] {getattr(m, 'type', '?')}: {getattr(m, 'content', '')}"
            for i, m in enumerate(state["messages"])
        )

        class _ParseFailure(Exception):
            """Model output was not valid JSON; carries the raw text for diagnosis."""

            def __init__(self, raw: str, err: Exception):
                self.raw, self.err = raw, err
                super().__init__(str(err))

        def _call(system: str, extra: str, feedback: str | None = None):
            messages = [SystemMessage(content=system + "\n\n" + extra)]
            if feedback:
                messages.append(HumanMessage(content=feedback))
            _t0 = _time.time()
            resp = model_config.invoke_checked(distill_model, messages, "distillation")
            usage_log.record("distillation", model_config.for_stage("distillation").provider,
                             model_config.for_stage("distillation").model,
                             (getattr(resp, "usage_metadata", None) or {}), duration_s=_time.time() - _t0)
            raw_label = ("needs" if system == DISTILL_NEEDS_SYSTEM else
                         "scope" if system == DISTILL_SCOPE_SYSTEM else
                         "preview" if system == DISTILL_PREVIEW_SYSTEM else "structured")
            raw_log.write("distillation", raw_label + ("-repair" if feedback else "-generation"), resp.content)
            try:
                return resp, _json.loads(extract_json_value(resp.content))
            except (_json.JSONDecodeError, ValueError) as e:
                raise _ParseFailure(resp.content, e) from e

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
                    except _ParseFailure as e:
                        last_err = e
                        if attempt == 2:
                            snippet = e.raw.strip()[:200].replace("\n", " ")
                            raise UserFacingError(
                                f"distillation phase '{label}' produced output that is not valid JSON (an automatic retry did not fix it).",
                                f"smaller/local models wrap JSON in prose or truncate. Raw output began: {snippet!r}",
                                "rerun (the interview resumes from its checkpoint) - or route distillation to a stronger model: "
                                "INTERVIEW_ENGINE_MODEL_DISTILLATION=...",
                            ) from e
                        if progress is not None:
                            import time as _t
                            progress.emit(ProgressEvent(stage, label, "started", _t.time(), None,
                                                        "model output was not valid JSON - retrying once with feedback"))
                        fb = ("Your previous reply was not valid JSON (prose-wrapped, unbalanced, or truncated). "
                              "Return ONLY the complete JSON payload, no prose, no commentary, balanced to the end.")
                        continue
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

        def _statements(items):
            # Model drift: done criteria and operational concepts can arrive as
            # {statement, source_turns} objects, while the contract carries strings.
            out = []
            for o in items:
                if isinstance(o, dict) and isinstance(o.get("statement"), str):
                    out.append(o["statement"])
                else:
                    out.append(str(o))
            return out

        def _valid_needs(data):
            needs = data.get("stakeholder_needs") or []
            if not needs:
                raise ValueError("stakeholder_needs must not be empty - an intent model with no needs is not an intent model")
            fixed = normalize_enums({"stakeholder_needs": needs}, IntentModel)["stakeholder_needs"]
            validated = [IntentNeed.model_validate(n) for n in fixed]
            from vmodel_contracts import Assumption
            items = data.get("assumption_items") or []
            return {"_glossary": data.get("glossary") if isinstance(data.get("glossary"), dict) else None,
                    "stakeholder_needs": [n.model_dump(mode="json") for n in validated],
                    "assumption_items": [Assumption.model_validate(a).model_dump(mode="json") for a in items],
                    "done_criteria": _statements(data.get("done_criteria") or []),
                    "operational_concepts": _statements(data.get("operational_concepts") or []),
                    "assumptions": data.get("assumptions") or []}

        def _as_str_list(v, *, field: str):
            """Normalize model-drifted list entries at the distillation boundary.

            Structured entries are accepted only through a field-specific text
            key. Never stringify an arbitrary dict: its repr would become
            accidental product data and hide schema drift.
            """
            text_key = "constraint" if field == "constraints" else "item"
            out = []
            for x in v or []:
                if isinstance(x, str):
                    text = x
                elif isinstance(x, dict) and isinstance(x.get(text_key), str):
                    text = x[text_key]
                elif field == "constraints" and isinstance(x, dict)                         and isinstance(x.get("statement"), str):
                    # Some models reuse the grounded-statement shape here.
                    text = x["statement"]
                else:
                    raise ValueError(f"{field} item must be text or an object with {text_key!r} text, "
                                     f"got {type(x).__name__}: {str(x)[:80]}")
                text = text.strip()
                if not text:
                    raise ValueError(f"{field} item text must not be empty")
                out.append(text)
            return out

        def _valid_scope(data):
            required_str = ("problem_statement", "core_value_proposition")
            required_list = ("target_users", "in_scope", "out_of_scope")
            for k in required_str:
                if not isinstance(data.get(k), str) or not data[k].strip():
                    raise ValueError(f"{k} must be a non-empty string")
            for k in required_list:
                if not isinstance(data.get(k), list):
                    raise ValueError(f"{k} must be a list")
                data[k] = _as_str_list(data[k], field=k)
            if not data["out_of_scope"]:
                raise ValueError("out_of_scope is empty - derive at least one exclusion from the conversation")
            from vmodel_contracts import ScopeExclusion
            for k in ("external_integrations", "constraints", "open_questions"):
                raw = data.get(k) or []
                if not isinstance(raw, list):
                    raise ValueError(f"{k} must be a list")
                data[k] = _as_str_list(raw, field=k)
            exclusions = [ScopeExclusion.model_validate(x).model_dump(mode="json")
                          for x in (data.get("scope_exclusions") or [])]
            keep = required_str + required_list + ("external_integrations", "constraints", "open_questions", "glossary", "source_turns")
            out = {k: data.get(k) for k in keep}
            # glossary drift: {term: meaning} sometimes arrives as [{term, meaning}, ...]
            gl = out.get("glossary")
            if isinstance(gl, list):
                fixed = {}
                for g in gl:
                    if isinstance(g, dict) and isinstance(g.get("term"), str) and isinstance(g.get("meaning"), str):
                        fixed[g["term"]] = g["meaning"]
                out["glossary"] = fixed
            out["scope_exclusions"] = exclusions
            return out

        parts = _phase("distill_needs", "Extracting interview-level needs", "phase 1/4",
                       DISTILL_NEEDS_SYSTEM, "TRANSCRIPT:\n" + transcript, _valid_needs)
        _needs_glossary = parts.pop("_glossary", None)
        parts.update(_phase("distill_scope", "Building scope and framing", "phase 2/4",
                            DISTILL_SCOPE_SYSTEM, "TRANSCRIPT:\n" + transcript, _valid_scope))
        if not parts.get("glossary") and isinstance(_needs_glossary, dict):
            parts["glossary"] = _needs_glossary
        parts["session_id"] = state["submission"]["submission_id"]

        # phase 3: requirement PREVIEW (presentation-only; never blocks the run)
        preview: list[dict] = []
        with track(progress, "distill_preview", "Drafting requirement preview", "phase 3/4"):
            try:
                _, raw = _call(DISTILL_PREVIEW_SYSTEM, "NEEDS:\n" + _json.dumps(parts["stakeholder_needs"]))
                if isinstance(raw, list):
                    preview = [p for p in raw
                               if isinstance(p, dict) and isinstance(p.get("statement"), str)][:3]
                    turns_text = [str(getattr(m, "content", "")) for m in state["messages"]
                                  if getattr(m, "type", "") == "human"]
                    for p in preview:
                        crit = str(p.get("example_criterion") or "")
                        if not crit:
                            continue
                        if is_solution_language(crit):
                            p.pop("example_criterion")  # implementation mechanics/named tech are not a check
                            if progress is not None:
                                progress.emit(ProgressEvent("distill_preview", "Dropped invented-tech criterion",
                                                            "completed", _time.time(), 0.0, crit[:70]))
                        elif not grounded_in(crit, turns_text):
                            p.pop("example_criterion")  # invented artifact - not from stakeholder language
                            if progress is not None:
                                progress.emit(ProgressEvent("distill_preview", "Dropped ungrounded criterion",
                                                            "completed", _time.time(), 0.0, crit[:70]))
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
                [str(n.get("statement", "")) for n in parts["stakeholder_needs"]
                 if isinstance(n, dict) and str(n.get("statement", "")).strip()])
            parts["constraints"] = kept_c
            # Founder-delegated decisions (steers like 'you do it'): appended by
            # CODE from interview state - grounded by construction, never left
            # for the distillation model to remember.
            for d in state.get("delegated_decisions") or []:
                # Remove an unvalidated copy the distillation model may have
                # echoed from transcript/state before appending the normalized
                # code-owned version below.
                parts["assumption_items"] = [a for a in parts["assumption_items"]
                    if str(a.get("statement", "")).strip() != str(d.get("statement", "")).strip()]
                # Repair checkpoints written by bb5b8d4: the automatic
                # two-attempt record had source_turns=[] and carried the
                # graph-only attempted_questions field into the contract.
                clean = {"statement": str(d.get("statement", "")).strip(),
                         "source_turns": list(d.get("source_turns") or [])}
                if not clean["source_turns"]:
                    attempted = list(d.get("attempted_questions") or [])
                    # The founder response immediately after the final emitted
                    # attempted question is the actual provenance turn.
                    messages = list(state.get("messages") or [])
                    for i, message in enumerate(messages[:-1]):
                        if (getattr(message, "type", "") == "ai" and attempted
                                and same_question_substance(getattr(message, "content", ""), attempted[-1])
                                and getattr(messages[i + 1], "type", "") == "human"):
                            clean["source_turns"] = [i + 1]
                    if not clean["source_turns"]:
                        # Conservative compatibility fallback for an already
                        # persisted invalid checkpoint: bind to the latest real
                        # founder turn rather than fabricating no provenance.
                        founder_turns = [i for i, m in enumerate(state.get("messages") or [])
                                         if getattr(m, "type", "") == "human"]
                        if founder_turns:
                            clean["source_turns"] = [founder_turns[-1]]
                if clean["statement"] and clean not in parts["assumption_items"]:
                    parts["assumption_items"].append(clean)
            if progress is not None:
                for d in dropped_a:
                    progress.emit(ProgressEvent("distill_validate", "Dropped ungrounded assumption",
                                                "completed", _t2.time(), 0.0, d))
                for d in dropped_c:
                    progress.emit(ProgressEvent("distill_validate", "Dropped duplicate constraint",
                                                "completed", _t2.time(), 0.0,
                                                d[:70] + " (already a classified need)"))
            from .distill_qa import verify_tech_preferences

            tprefs, dropped_tp, warnings_tp = verify_tech_preferences(parts.get("tech_preferences") or [])
            parts["tech_preferences"] = tprefs
            for tp in tprefs:  # dual form: the flat constraints list carries them too (ADR-0019)
                rendered = f"{tp.get('area', 'technology')}: {tp.get('preference')} (founder, {tp.get('strength', 'preferred')})"
                if rendered not in parts["constraints"]:
                    parts["constraints"].append(rendered)
            if progress is not None:
                for d in dropped_tp:
                    progress.emit(ProgressEvent("distill_validate", "Dropped ungrounded tech preference",
                                                "completed", _t2.time(), 0.0, d))
                for w in warnings_tp:
                    progress.emit(ProgressEvent("distill_validate", "Unrecognized ecosystem name",
                                                "completed", _t2.time(), 0.0, w))
            if (state.get("coverage") or {}).get("glossary") == "covered" and not parts.get("glossary"):
                import time as _t3
                if progress is not None:
                    progress.emit(ProgressEvent(
                        "distill_validate", "Glossary gap", "completed", _t3.time(), 0.0,
                        "founder defined terms in the interview but distillation captured none - "
                        "the intent model has no glossary; re-run or add terms manually"))
            parts["research_briefs"] = [
                {"topic": str(d.get("statement", "")).split(" - ", 1)[0].removeprefix("FOUNDER-DELEGATED: "),
                 "brief": str(d.get("research_brief") or "Research this delegated decision using current authoritative sources and present it for founder review."),
                 "source_turns": list(d.get("source_turns") or [])}
                for d in state.get("delegated_decisions") or [] if d.get("source_turns")
            ]
            data = normalize_enums(coerce_for_model(parts, IntentModel), IntentModel)
            try:
                intent = IntentModel.model_validate(data)
            except ValidationError as e:
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']} (got {str(err.get('input'))[:60]})"
                    for err in e.errors())[:1500]
                raise UserFacingError(
                    "the assembled intent model fails the contract schema.",
                    f"validation detail: {detail}",
                    "rerun (the interview resumes from its checkpoint) - or route distillation to a stronger model.",
                ) from e
        return {"intent_model": intent.model_dump(mode="json"), "requirement_preview": preview,
                "distill_warnings": warnings_tp}


    def node_conops(state: InterviewState) -> dict:
        """One resumable stage: draft once, pause for an exact founder choice,
        then finalize without another model call."""
        if state.get("conops"):
            return {}
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from .conops_stage import (CONOPS_SYSTEM, comparison_question, finalize,
                                   derive_scenario, derive_stakeholder_contexts,
                                   scenario_seeds_from_intent, validate_draft)
        from .errors import UserFacingError
        from .repair import extract_json_value, normalize_model_item
        import json as _json

        draft = state.get("conops_draft")
        last = state.get("messages", [])[-1] if state.get("messages") else None
        if draft:
            if last is None or getattr(last, "type", "") != "human":
                return {}
            source_turn = len(state["messages"]) - 1
            try:
                doc = finalize(draft, state["submission"]["submission_id"],
                               getattr(last, "content", ""), source_turn)
            except ValueError:
                return {"messages": [AIMessage(content=comparison_question(draft)
                    + "\n\nI couldn't match an exact option plus rationale. Reply as `<number or exact name> - <why you chose it>`.")]}
            if progress is not None:
                progress.emit(ProgressEvent("conops", "Concept of Operations decision", "completed",
                                            _time.time(), 0.0, doc.decision.selected_concept))
            return {"conops": doc.model_dump(mode="json")}

        transcript = "\n".join(
            f"[{i}] {getattr(m, 'type', '?')}: {getattr(m, 'content', '')}"
            for i, m in enumerate(state.get("messages", [])))
        founder_turns = {i for i, m in enumerate(state.get("messages", []))
                         if getattr(m, "type", "") == "human"}
        intent = state.get("intent_model") or {}
        context_seeds = derive_stakeholder_contexts(intent, founder_turns)
        scenario_seeds = [derive_scenario(x, intent) for x in scenario_seeds_from_intent(intent, founder_turns)]

        raw_paths: dict[str, str] = {}
        def call_json(system_text: str, body: str, label: str, detail: str) -> dict | list:
            if progress is not None:
                progress.emit(ProgressEvent("conops", label, "started", _time.time(), None, detail))
            t0 = _time.time()
            resp = model_config.invoke_checked(conops_model,
                [SystemMessage(content=system_text), HumanMessage(content=body)], "conops")
            usage_log.record("conops", model_config.for_stage("conops").provider,
                             model_config.for_stage("conops").model,
                             (getattr(resp, "usage_metadata", None) or {}), duration_s=_time.time()-t0)
            raw_path = raw_log.write("conops", label, resp.content)
            if raw_path:
                raw_paths[label] = str(raw_path)
            try:
                value = _json.loads(extract_json_value(resp.content))
            except Exception as exc:
                raise UserFacingError(
                    f"stage 'conops': {label.lower()} returned unreadable JSON.",
                    f"parse detail: {str(exc)[:300]}",
                    "inspect the raw response at " + (str(raw_path) if raw_path else "the configured debug output")
                    + ", then rerun with the same --session id; completed intent work is checkpointed.") from exc
            if progress is not None:
                progress.emit(ProgressEvent("conops", label, "completed", _time.time(), _time.time()-t0, detail))
            return value

        def generate_item(kind: str, identity: dict, model_cls, prompt: str, index: int, total: int) -> dict:
            derived_fields = identity.pop("_derived_fields", [])
            candidate = normalize_model_item(identity, model_cls)
            candidate.update({k:v for k,v in identity.items() if k in model_cls.model_fields})
            missing=[]
            for name, field in model_cls.model_fields.items():
                if name == "contract_version" or not field.is_required():
                    continue
                if candidate.get(name) in (None,"",[]):
                    missing.append(name)
            for field_name in missing:
                label=f"Building CONOPS {kind} {index}/{total} field {field_name}"
                body=(f"Fill ONLY the `{field_name}` field for this {kind}. Return JSON exactly as "
                      f"{{\"{field_name}\": <value>}}. Do not return null.\n"
                      f"CODE-DERIVED FIELDS (do not rewrite): {_json.dumps(candidate)}\n"
                      f"VALIDATED INTENT: {_json.dumps(intent)}\nTRANSCRIPT:\n{transcript}\n{prompt}")
                raw=call_json(CONOPS_SYSTEM,body,label,f"field {field_name}; item {index} of {total}")
                try:
                    # A field call is cleaned against a synthetic missing-field
                    # error contract: normalize wrapper, whitelist, take only it.
                    normalized=normalize_model_item(raw,model_cls)
                    value=normalized.get(field_name)
                    if value in (None,"",[]):
                        raise ValueError(f"field {field_name} was not provided")
                    candidate[field_name]=value
                except ValueError as field_error:
                    path=raw_paths.get(label)
                    raise UserFacingError(
                        f"stage 'conops': {kind} item {index}/{total} field '{field_name}' was not provided.",
                        str(field_error), "inspect the raw response at "+(path or "the configured debug output")
                        +", then rerun with the same --session id.") from field_error
            try:
                item=model_cls.model_validate(candidate)
            except Exception as final_error:
                path=raw_paths.get(label) if missing else None
                raise UserFacingError(
                    f"stage 'conops': {kind} item {index}/{total} failed after per-field generation.",
                    f"validation detail: {str(final_error)[:350]}; code-derived fields: {derived_fields}",
                    "inspect the raw response at "+(path or "the configured debug output")
                    +", then rerun with the same --session id.") from final_error
            if hasattr(item,"source_turns"):
                invalid=sorted(set(item.source_turns)-founder_turns)
                if invalid:
                    raise UserFacingError(f"stage 'conops': {kind} item {index}/{total} cites invalid provenance.",
                        f"non-founder transcript turns: {invalid}",
                        "rerun with the same --session id; the invalid item was not checkpointed.")
            if progress is not None and derived_fields:
                progress.emit(ProgressEvent("conops",f"Derived CONOPS {kind} {index}/{total}","completed",
                    _time.time(),0.0,"code-derived from validated intent: "+", ".join(derived_fields)))
            return item.model_dump(mode="json")

        from vmodel_contracts import ConceptOption, OperationalScenario, StakeholderContext
        with track(progress, "conops", "Building Concept of Operations",
                   f"{len(context_seeds)} stakeholders, {len(scenario_seeds)} scenarios, comparison + options"):
            contexts = [generate_item("stakeholder", seed, StakeholderContext,
                "Fill role, goals, and operating_environment from founder evidence; retain source_turns.",
                i, len(context_seeds)) for i, seed in enumerate(context_seeds,1)]
            scenarios = [generate_item("scenario", seed, OperationalScenario,
                "Fill trigger, actors, ordered operational_flow, success_outcome, exceptions; retain source_turns.",
                i, len(scenario_seeds)) for i, seed in enumerate(scenario_seeds,1)]

            comparison = call_json(CONOPS_SYSTEM,
                "VALIDATED INTENT: " + _json.dumps(intent) + "\nPropose 2-4 genuinely distinct product-level operating concepts. "
                "Return a JSON array of objects with name and summary only. Do not include architecture components.",
                "Planning CONOPS concept comparison", "known next step: one validated call per option")
            if not isinstance(comparison,list) or not 2 <= len(comparison) <= 4:
                raise UserFacingError(
                    "stage 'conops': concept comparison plan must contain 2-4 options.",
                    f"received: {str(comparison)[:300]}",
                    "rerun with the same --session id; stakeholder and scenario work will rerun from intent.")
            identities=[]
            for raw in comparison:
                if not isinstance(raw,dict) or not str(raw.get("name","")).strip() or not str(raw.get("summary","")).strip():
                    raise UserFacingError("stage 'conops': comparison plan contains an invalid option identity.",
                                          f"received: {str(raw)[:250]}",
                                          "rerun with the same --session id; intent is checkpointed.")
                from vmodel_contracts import canonical_concept_id
                name = str(raw["name"]).strip()
                identities.append({"concept_id": canonical_concept_id(name), "name": name,
                                   "summary": str(raw["summary"]).strip()})
            if len({x["name"].casefold() for x in identities}) != len(identities):
                raise UserFacingError("stage 'conops': comparison plan repeated an option name.",
                                      f"names: {[x['name'] for x in identities]}",
                                      "rerun with the same --session id; intent is checkpointed.")
            options=[generate_item("concept option", ident, ConceptOption,
                "Fill concise advantages and disadvantages grounded in the intended operation.",
                i,len(identities)) for i,ident in enumerate(identities,1)]
            draft=validate_draft({"stakeholder_context":contexts,"operational_scenarios":scenarios,
                                  "concept_options":options},founder_turns)
        return {"conops_draft": draft,
                "messages": [AIMessage(content=comparison_question(draft))]}

    g = StateGraph(InterviewState)
    g.add_node("scope_gate", node_scope_gate)
    g.add_node("interview_turn", node_interview_turn)
    g.add_node("distill_intent", node_distill)
    g.add_node("conops", node_conops)
    g.set_entry_point("scope_gate")
    g.add_conditional_edges("scope_gate", route_scope, {"interview_turn": "interview_turn", "conops": "conops", END: END})
    g.add_conditional_edges("interview_turn", route_interview,
                            {"interview_turn": "interview_turn", "await_founder": END, "distill_intent": "distill_intent"})
    g.add_edge("distill_intent", "conops" if conops_enabled else END)
    g.add_edge("conops", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())

