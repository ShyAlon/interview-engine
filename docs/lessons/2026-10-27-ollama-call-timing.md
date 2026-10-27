# 2026-10-27 - Per-call model timing: what the wire actually gives you

The ollama client goes through the OpenAI-compatible endpoint
(`ChatOpenAI` at `localhost:11434/v1`). That response carries ONLY token
usage - `load_duration`/`prompt_eval_duration`/`eval_duration` exist solely
on the native `/api/chat` (verified in ollama main `openai/openai.go`:
`ToChatCompletion` emits `usage` and nothing else; zero occurrences of the
duration fields). `keep_alive` IS honored on the compat path (passed through
to `api.ChatRequest`, openai.go:815). So: record what the wire gives
(latency, token counts, keep_alive, derived tok/s), and if a future client
surfaces the native split in `response_metadata`, log it in ms - honest
absence, never zeros. Owner ruled (2026-10-27): no infrastructure switch
just for instrumentation; call type is deduced from the stage/slot, which
the log event already carries.

Observability never breaks a run: the per-call progress event publishes
through a ContextVar sink that defaults to None, and the publish itself is
wrapped - same pattern as the heartbeat.

Sandbox note: `tests/test_pg_application_wiring.py` spins up embedded
Postgres via `pgserver`; `initdb` flakes in this sandbox (CalledProcessError
from initdb), red on clean origin/main too, pass/fail varies run to run.
Unrelated to any change here.
