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

## Canonical artifact lifecycle vocabulary

Lifecycle terms come from `vmodel_contracts.artifact_lifecycle`: `missing`,
`partial`, `complete`, `stale`, `invalidated`, and `migrated`. During phase 1,
LangGraph still owns orchestration; these words describe artifacts, not graph
nodes. Interview turns and per-phase work are `partial`; a schema-valid sealed
IntentModel or CONOPS is `complete`; a contract-transformed disk artifact is
`migrated` until `adopt_migration` updates canonical storage and checkpoint
atomically. A stale checkpoint must never overwrite a `migrated` artifact.
Resume is legal only with a named next artifact and outgoing arc. See the
contracts README for the guarded transition table and diagram exporters.

## Jev shadow comparison

When `TYPESAFE_API_KEY` resolves, Jev shadow mode is on by default. The existing
LLM/rules chain remains authoritative; Jev runs in parallel and cannot change
interview state. Disable it with `INTERVIEW_ENGINE_JEV_SHADOW=off`.

Each interview appends real-time comparison records to
`categorization-shadow.jsonl` in its interview artifact directory. Records use
a SHA-256 input reference rather than copying founder text. Regenerate a report:

```bash
./interview-engine/scripts/report-categorization-shadow.py \
  /path/to/interview/categorization-shadow.jsonl \
  --output /path/to/interview/categorization-shadow.md
```

Several runs can be combined by listing several JSONL paths before `--output`.


### Adjudication-ready Jev shadow data

New records retain the SHA-256 `input_ref` and also write the complete decision-time
input package: request state (founder utterance plus relevant question), structured
context, versioned rubric/questions, both verdicts/confidences, and provider/model
versions. The append-only JSONL remains authoritative.

Create a blind-ready labeling bundle containing every disagreement and a deterministic,
stratified agreement sample (slot + baseline label + Jev confidence band):

```bash
./scripts/export-categorization-labels.py \
  out/*/interview/categorization-shadow.jsonl \
  --output /tmp/jev-adjudication.csv
# Keep /tmp/jev-adjudication.csv.key.json away from reviewers until labels are sealed.
```

For an older session, export its transcript as a JSON list of `{role, content}` turns,
then recover any matching inputs by their existing hash without rewriting the source:

```bash
./scripts/enrich-categorization-shadow.py OLD.jsonl \
  --transcript transcript.json --output OLD.enriched.jsonl
```

Unrecoverable hashes remain explicit legacy rows. They are never guessed or silently
dropped.

### Rubric and testbed status (September 2026)

Production still defaults to intent-v1 (v1.3 five-label criteria) and
relevance-v1. Explicit opt-in replay variants intent-v1.4, intent-v1.5,
intent-v1.6, intent-v1.7, and relevance-trees-v1..v5 are **research only**. The new
`proceed` and `non-answer` outputs from intent-v1.5/v1.6/v1.7 have no production
graph behavior yet. Replaying a rubric does not enable it at runtime.

JEV-L2 is the 195-case real question/reply label set. For the September 25
run, eight reviewer-marked summaries and two clarification requests were
split out after export, leaving 185 five-label cases. The original export,
filtered CSV, clarification audit, and replay JSONLs should be kept together
outside git. JEV-L3 is a separate blind 34-case addressing review: 20
contested real calls, eight real controls, six synthetic wrong-area replies.
The review CSV and its audit manifest must be kept with the corresponding
replay; two boundary verdicts remain unscored. Cat probes are synthetic and
are reported separately from real cases. The strict merged/v1.5/v1.6 reports
have a different scoring contract from the historical `score-rubric-variants.py`
answer/delegate compatibility score.

The separate opt-in intent-v1.7 request tests question-target addressing and
partial-answer precedence after v1.6. Its offline scorer is
`scripts/score-jev-l3-v17.py`; pass the filtered 185-case real CSV, matching
v1.7 real replay, 12 cat cases/replay, reviewed JEV-L3 CSV and audit, and six
swap cases/replay, in that order. It rejects missing/duplicate/wrong-variant
replay IDs. The two reviewed boundary cases remain unscored (183 real scored),
and real versus synthetic scores remain separate. Run Jev calls on Shy's Mac;
a dry run here only checks request construction. Keep the v1.6 and v1.7 replay
files separately.

Research scripts remain under `scripts/` with their original CLI paths so
past runs can be reproduced. Input datasets under `benchmarks/` are versioned;
generated runs belong under ignored `runs/` or `out/`. Do not infer
production readiness from synthetic pass rates or historical variant names.
