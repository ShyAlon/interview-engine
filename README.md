# interview-engine

The adaptive systems-engineering interview: a founder's rough idea in, a
versioned `IntentModel` out. **This stage is the moat** - the LLM budget is
spent here on purpose (see ARCHITECTURE.md in `vmodel-spec-app`, ADR-0004).

## What it does

1. **Scope gate** (`scope_gate.py`) - web applications only. Deterministic
   signal matching first; an LLM tiebreaker runs only when signals are mixed
   or silent. Out-of-scope ideas get a polite, specific decline and the run
   ends. The gate never re-runs (and never re-pays) on a resumed thread.
2. **Interview loop** (`graph.py`) - a LangGraph `StateGraph`: stateful,
   resumable via checkpointer, human-in-the-loop by construction. The
   interviewer persona (`prompts.py`) chases what founders forget: failure
   modes, privacy obligations, acceptance criteria. Hard stop at 60 turns.
3. **Distillation** - transcript condenses into an `IntentModel`
   (vmodel-contracts), every field traced to transcript turns, open questions
   carried, never silently resolved.
4. **Founder review gate** - the intent model renders to markdown for
   approval/edit before any generation. Cheapest place to catch a
   misunderstanding.

## Model routing

`model_config.py` is the model-agnostic layer: per-stage provider+model,
env-overridable (`INTERVIEW_ENGINE_MODEL_PROBING=openai:gpt-5`). Default is
DeepSeek (demo prices). Swapping providers is a config change, never a
refactor (ADR-0003).

## Telemetry

Every LLM call records a `StageTokenUsage` (vmodel-contracts) with the
versioned price table in `telemetry.py`. The $5-per-run claim is a belief
until these records exist.

## Run the demo

```bash
pip install -e .[dev]
export DEEPSEEK_API_KEY=...
python -m interview_engine.cli "a marketplace for surf lessons in Tel Aviv"
```

Artifacts land in `out/<session>/`: `intent_model.json`, `intent_model.md`,
`usage.jsonl`.

## Boundary rule

Imports `vmodel-contracts` (pinned release) and nothing from sibling modules.
Output crosses to `spec-generator` only as an `IntentModel` payload.
