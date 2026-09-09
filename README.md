# interview-engine

The adaptive systems-engineering interview: a founder's rough idea in, a
versioned `IntentModel` out. **This stage is the moat** - the LLM budget is
spent here on purpose (see ARCHITECTURE.md in `vmodel-spec-app`, ADR-0004).

Pipeline map: [PROCESS.md in the meta repo](https://github.com/ShyAlon/vmodel-spec-app/blob/main/docs/PROCESS.md) shows what each stage consumes and produces (and why stage 1 is "Formalizing", not "Extracting").

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
   (vmodel-contracts) in four visible phases: extract needs -> build scope ->
   requirement preview -> assemble/validate (ADR-0013). Every field traced to
   transcript turns; open questions carried, never silently resolved; founder
   assertions captured as assumptions, not needs. Each phase validates
   independently with mechanical shape-drift repair + one schema-feedback
   retry (ADR-0011).
4. **Founder review gate** - the intent model renders to markdown for
   approval/edit before any generation. Cheapest place to catch a
   misunderstanding.

## Running, progress, and crash-resume

`python -m interview_engine.cli "your idea"` prints a live progress line per
stage (`-> Interviewing (turn 3)...` / `ok ... [12.4s]`) driven by structured
`ProgressEvent`s (progress.py) - the same stream the UI will consume
(ADR-0010). Checkpoints persist to `out/<session>/checkpoints.sqlite`: if
anything crashes, `python -m interview_engine.cli --session <id>` resumes
exactly where it stopped - completed stages are never re-paid (ADR-0011).

## Model routing

`model_config.py` is the model-agnostic layer: per-stage provider+model,
env-overridable (`INTERVIEW_ENGINE_MODEL_PROBING=openai:gpt-5`). Default is
LOCAL Ollama ($0, no key) - the product owner's standing call until production
(ADR-0018); DeepSeek/OpenAI/Anthropic are opt-in per stage. Swapping providers
is a config change, never a refactor (ADR-0003).

## Telemetry

Every LLM call records a `StageTokenUsage` (vmodel-contracts) with the
versioned price table in `telemetry.py`. The $5-per-run claim is a belief
until these records exist.

## Run the demo

```bash
pip install -e .[dev]
ollama pull qwen2.5-coder:14b   # the default model (qwen2.5-coder:7b is the lighter fallback)
export OLLAMA_NUM_PARALLEL=1   # never load two models at once on 16GB
python -m interview_engine.cli "a marketplace for surf lessons in Tel Aviv"
```

Remote providers are opt-in per stage, e.g.
`INTERVIEW_ENGINE_MODEL_PROBING=deepseek:deepseek-chat` + `export DEEPSEEK_API_KEY=...`.

Artifacts land in `out/<session>/`: `intent_model.json`, `intent_model.md`,
`usage.jsonl`.

## Boundary rule

Imports `vmodel-contracts` (pinned release) and nothing from sibling modules.
Output crosses to `spec-generator` only as an `IntentModel` payload.

## Technology preferences (ADR-0019)

The interview asks openly about technology constraints (language/ecosystem,
cloud affinity - "agnostic" is recorded as an answer, not silence). Stated
preferences land in the intent model as `tech_preferences` with transcript
provenance; ungrounded ones are dropped, unrecognized ecosystem names are kept
but flagged at the approval gate. They confine the architecture stage in
spec-generator - the SRS stays solution-agnostic. Cloud is captured, never
forced: it binds only when the founder explicitly mandates it.

## Model size

`qwen2.5-coder:14b` is the standing default - proven on a 16GB M4 with
`OLLAMA_NUM_PARALLEL=1` set. `qwen2.5-coder:7b` is the lighter fallback for
tighter machines - downgrade per stage:

```bash
for s in PROBING DISTILLATION PHRASING SCOPE_TIEBREAK; do
  export INTERVIEW_ENGINE_MODEL_$s=ollama:qwen2.5-coder:7b
done
```
```

Ollama calls are logged at $0.00 in `usage.jsonl` (local compute). Quality
note: the interview IS the moat - judge demo quality on the real provider
before showing founders; Ollama is for the dev loop.
