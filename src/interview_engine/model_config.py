"""Per-stage model routing. THE model-agnostic layer: swapping providers is a
config change here, never a refactor. Providers:

- ollama    (DEFAULT - local-first, the product owner's standing call:
             "It should all be ollama by default. We'll change it once we go
             to production." $0, no API key, no data leaves the machine.
             Needs a running Ollama server + a pulled model.)
- deepseek  (opt-in - demo prices, OpenAI-compatible)
- openai    (opt-in)
- anthropic (opt-in)

Every stage's override is deliberate (see ARCHITECTURE.md, ADR-0004):
- probing / distillation earn the strong model for founder-facing runs -
  they are the moat: INTERVIEW_ENGINE_MODEL_PROBING=deepseek:deepseek-chat etc.
- phrasing / tiebreaks can drop to a cheaper tier without visible quality loss.
Local Ollama is the default until production (ADR-0018); judge demo quality on
the real provider before showing founders.
"""
from __future__ import annotations

import json
import os
import re as _re
import sys
import threading
import time
from dataclasses import dataclass

_DOTENV: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    """Config files, parsed once per process, most-shared first:

    1. <require-root>/.env - the SHARED config one directory up
       (vmodel-spec-app/scripts/switch-model.sh writes there; one file drives
       every process). Present only in the sibling-repo layout; absent in a
       standalone checkout and nothing changes.
    2. ./.env - repo-local overrides for the rare per-repo deviation.

    Real environment variables always win over both files (see _env), and
    os.environ is never mutated - no dependency, no side effects."""
    global _DOTENV
    if _DOTENV is not None:
        return _DOTENV
    _DOTENV = {}
    from pathlib import Path

    for p in (Path("..") / ".env", Path(".env")):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            _DOTENV[key.strip()] = value.strip().strip('"').strip("'")
    return _DOTENV


def _env(key: str) -> str | None:
    """One setting lookup: the real environment wins, the repo-local .env
    fills the gap. Every external config read goes through here so a .env
    written by switch-model.sh works for API keys too, not just routing."""
    return os.environ.get(key) or _dotenv().get(key)


# ---------------------------------------------------------------- guardrails
# Every LLM call passes through these limits. A hung ollama generation costs a
# hang at full GPU; the same hang on a paid API is a bill. All four knobs are
# env/.env-overridable; the defaults match "demo on a 16GB laptop, or cents".
_CALL_TIMEOUT_S = 600   # wall-clock budget per call, client retries included
_STALL_TIMEOUT_S = 180  # a stream this quiet is hung, not thinking
_MAX_CALLS = 200        # what ONE process run may spend, in LLM calls
_CALLS_USED = 0


_MAX_TOKENS_ENV = "VMODEL_LLM_MAX_TOKENS"  # per-call generation ceiling override (unset: stage default)


def _max_tokens_limit(stage_default: int) -> int:
    """Per-call generation-length ceiling. Each stage ships its own default
    (StageModel.max_tokens); VMODEL_LLM_MAX_TOKENS overrides it globally - a
    looping generation hits a length ceiling, not only a clock ceiling."""
    raw = _env(_MAX_TOKENS_ENV)
    try:
        v = int(raw) if raw else stage_default
    except ValueError:
        v = stage_default
    return v if v > 0 else stage_default


def _truncation_reason(resp) -> str | None:
    meta = getattr(resp, "response_metadata", None) or {}
    for key in ("done_reason", "finish_reason", "stop_reason"):  # ollama / openai-compat / anthropic
        if meta.get(key) in ("length", "max_tokens"):
            return str(meta[key])
    return None


def _check_truncated(resp, stage: str, cfg) -> None:
    """A generation cut at the token ceiling is a failure, not an answer: name
    it here, or it surfaces downstream as a confusing parse/validation error."""
    if _truncation_reason(resp) is None:
        return
    from .errors import UserFacingError

    ceiling = _max_tokens_limit(cfg.max_tokens)
    _log_llm({"event": "llm_call_error", "provider": cfg.provider, "model": cfg.model, "stage": stage,
              "error": {"kind": "token_ceiling", "ceiling_tokens": ceiling}})
    ollama_context = (" For Ollama's OpenAI-compatible endpoint, per-request num_ctx may be ignored; "
                      "restart the server with OLLAMA_CONTEXT_LENGTH=32768 if the prompt itself is being cut."
                      if cfg.provider == "ollama" else "")
    raise UserFacingError(
        f"stage '{stage}': the {cfg.provider}/{cfg.model} call hit the generation-length "
        f"ceiling ({ceiling} tokens) and its output was cut off.",
        "the model had more to emit than one call allows - the payload is truncated." + ollama_context,
        "rerun the same command - the checkpoint resumes at this stage. If it repeats, raise "
        f"{_MAX_TOKENS_ENV} only deliberately, or route the stage to another provider with "
        "vmodel-spec-app/scripts/switch-model.sh.",
    )


def _guard_limits() -> tuple[int, int, int]:
    def _num(name: str, default: int) -> int:
        raw = _env(name)
        try:
            v = int(raw) if raw else default
        except ValueError:
            v = default
        return v if v > 0 else default

    return (_num("VMODEL_LLM_CALL_TIMEOUT_S", _CALL_TIMEOUT_S),
            _num("VMODEL_LLM_STALL_TIMEOUT_S", _STALL_TIMEOUT_S),
            _num("VMODEL_LLM_MAX_CALLS", _MAX_CALLS))


def _llm_log_sink() -> str | None:
    """VMODEL_LLM_LOG: unset = off (default, zero cost); 'stderr' = console;
    any other value = a file path, one JSON object per line."""
    return _env("VMODEL_LLM_LOG")


def _mask_secrets(value: str) -> str:
    return _re.sub(r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+", r"\1...", str(value))


def _log_llm(event: dict) -> None:
    """One JSON line per LLM call event. Never logs message content or keys -
    metadata, token usage, and provider error payloads only. Logging must
    never break a run."""
    sink = _llm_log_sink()
    if not sink:
        return
    line = json.dumps({**event, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=str)
    try:
        if sink == "stderr":
            _console_line(line)
        else:
            with open(sink, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


_CONSOLE_LOCK = threading.Lock()


def _console_line(msg: str) -> None:
    """One stderr line, atomically - the heartbeat thread and the progress
    printer share this lock so lines never interleave mid-call."""
    with _CONSOLE_LOCK:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def _heartbeat_interval() -> float:
    """VMODEL_LLM_HEARTBEAT_S: seconds between 'still working' lines while a
    call runs without producing output. Default 30; 0 disables."""
    try:
        return float(_env("VMODEL_LLM_HEARTBEAT_S") or "30")
    except ValueError:
        return 30.0


def _start_heartbeat(label: str):
    """Print '  ... still working - <label>, Ns elapsed' every
    VMODEL_LLM_HEARTBEAT_S seconds until the returned stop() is called.
    Daemon thread; zero threads when disabled."""
    interval = _heartbeat_interval()
    if interval <= 0:
        return lambda: None
    t0 = time.monotonic()
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval):
            if stop.is_set():  # a beat already past the wait must not print after the call returned
                break
            _console_line(f"  ... still working - {label}, {time.monotonic() - t0:.0f}s elapsed")

    threading.Thread(target=_beat, daemon=True).start()
    return stop.set


def _provider_error_detail(exc) -> dict:
    """Best-effort raw provider error for logs and the error taxonomy: HTTP
    status, error type/code/message, request id, retry-after. The openai SDK
    shape (openai, deepseek) carries .status_code/.body/.response; local
    ollama errors degrade to class + message. Secret-shaped strings masked."""
    out = {"class": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if status is not None:
        out["http_status"] = status
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for h, key in (("x-request-id", "request_id"), ("retry-after", "retry_after_s")):
        try:
            v = headers.get(h)
        except Exception:
            v = None
        if v:
            out[key] = v
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        for k in ("type", "code", "param", "message"):
            if err.get(k) is not None:
                out[k] = _mask_secrets(str(err[k]))[:600]
    if "message" not in out:
        out["message"] = _mask_secrets(str(exc))[:600]
    return out


def _usage_of(resp) -> dict:
    usage = getattr(resp, "usage_metadata", None) or {}
    return {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens") if k in usage}


def _rerun_guidance(stage: str) -> str:
    """The rerun instruction that is TRUE for this invocation: an explicit
    --from <stage> forces a re-run and discards preserved per-item work, so
    after a failure the right command drops --from."""
    frm = os.environ.get("SPEC_GENERATOR_FROM")
    if frm:
        return (f"rerun the command WITHOUT '--from {frm}' ('--from' forces a re-run from '{frm}' and "
                "discards finished per-item work) - the checkpoint resumes at this stage and finished "
                "items are skipped")
    return "rerun the same command - the checkpoint resumes at this stage"

def _guard_enter(stage: str) -> None:
    """One LLM call is starting: charge it against the run budget."""
    global _CALLS_USED
    _CALLS_USED += 1
    if _CALLS_USED > _guard_limits()[2]:
        cfg = for_stage(stage)
        from .errors import UserFacingError

        raise UserFacingError(
            f"stage '{stage}': this run hit its LLM call budget ({_guard_limits()[2]} calls) and was stopped.",
            "the budget bounds what one run can cost and catches retry loops; the "
            f"{cfg.provider}/{cfg.model} calls before it all completed.",
            "rerun the same command - the checkpoint resumes at this stage. If the run keeps "
            "hitting the budget a stage is looping: report it. Raise the ceiling with "
            "VMODEL_LLM_MAX_CALLS only deliberately.",
        )


def _guard_killed(stage: str, kind: str, waited: float, limit: int):
    cfg = for_stage(stage)
    from .errors import UserFacingError

    return UserFacingError(
        f"stage '{stage}': the {cfg.provider}/{cfg.model} call was killed - {kind} "
        f"(limit {limit}s, waited {waited:.0f}s).",
        "a runaway generation or a hung connection; the call was making no useful progress.",
        "rerun the same command - the checkpoint resumes at this stage. If it repeats, check "
        "the provider is healthy (for ollama: ollama ps + server logs), raise "
        "VMODEL_LLM_CALL_TIMEOUT_S / VMODEL_LLM_STALL_TIMEOUT_S, or route the stage to another "
        "provider with vmodel-spec-app/scripts/switch-model.sh.",
    )


def _run_with_deadline(fn, timeout_s: int, stage: str):
    """Wall-clock deadline around one blocking call. The worker thread cannot
    be force-killed in Python, so the client ALSO gets the same timeout (see
    build_chat_model) - the socket dies, the thread unwinds, and the caller
    gets an informative error instead of a hang."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError from None


@dataclass(frozen=True)
class StageModel:
    provider: str          # 'deepseek' | 'openai' | 'anthropic' | 'ollama'
    model: str
    temperature: float = 0.4
    max_tokens: int = 2048


_STAGE_DEFAULTS: dict[str, StageModel] = {
    "probing":       StageModel("ollama", "qwen2.5-coder:14b", temperature=0.5),
    "distillation":  StageModel("ollama", "qwen2.5-coder:14b", temperature=0.2),
    # CONOPS is intentionally not configured independently. It follows the
    # shared distillation route below, so adding a stage can never fall back to
    # a different model than the interview configured in the shared .env.
    "conops":         StageModel("ollama", "qwen2.5-coder:14b", temperature=0.2),
    "phrasing":      StageModel("ollama", "qwen2.5-coder:14b", temperature=0.7),
    "scope_tiebreak": StageModel("ollama", "qwen2.5-coder:14b", temperature=0.0),
}

_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": None,  # SDK default
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}
_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,  # no key; ChatOpenAI just needs a non-empty string
}


def for_stage(stage: str) -> StageModel:
    if stage not in _STAGE_DEFAULTS:
        raise KeyError(f"unknown stage {stage!r}; add it explicitly - silent fallback models are how budgets leak")
    cfg = _STAGE_DEFAULTS[stage]
    # Concept of Operations is part of interview understanding, not its own
    # model tier. It MUST follow the shared distillation route. The first live
    # run exposed why: all configured stages used phi4, while a new CONOPS key
    # silently fell back to qwen. There is deliberately no
    # INTERVIEW_ENGINE_MODEL_CONOPS setting.
    route = "distillation" if stage == "conops" else stage
    override = _env(f"INTERVIEW_ENGINE_MODEL_{route.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return _STAGE_DEFAULTS[route] if stage == "conops" else cfg


def _preflight(cfg: "StageModel", stage: str) -> None:
    """Fail BEFORE the call, with the fix named. An empty remote key would
    otherwise surface as a raw SDK 401 ten calls deep."""
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not _env(env):
        raise UserFacingError(
            f"stage '{cfg.model}' is routed to provider '{cfg.provider}', but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... - remote providers are opt-in (the default is local Ollama); "
            f"or route this stage back to local: INTERVIEW_ENGINE_MODEL_{stage.upper()}.=ollama:qwen2.5-coder:14b.",
        )


def describe_models() -> str:
    """What will actually run, resolved per stage after env/.env layering -
    printed at startup so the log always records which provider/model ran."""
    return ", ".join(f"{s}={for_stage(s).provider}/{for_stage(s).model}" for s in _STAGE_DEFAULTS)


def invoke_checked(chat_model, messages, stage: str, item: str | None = None):
    """chat_model.invoke under the guardrails (budget + deadline), with the
    known pains translated to human language. Every call is logged when
    VMODEL_LLM_LOG is set (start/end/error, latency, token usage, raw provider
    error payload - never message content, never keys)."""
    from .errors import UserFacingError, translate_llm_error

    cfg = for_stage(stage)
    _guard_enter(stage)
    call_timeout = _guard_limits()[0]
    base = {"provider": cfg.provider, "model": cfg.model, "stage": stage,
            **({"item": item} if item else {})}
    _log_llm({"event": "llm_call_start", **base})
    stop_heartbeat = _start_heartbeat(f"stage '{stage}'" + (f" [{item}]" if item else "") + " call")
    t0 = time.monotonic()
    try:
        resp = _run_with_deadline(lambda: chat_model.invoke(messages), call_timeout, stage)
        _check_truncated(resp, stage, cfg)
        _log_llm({"event": "llm_call_end", **base, "latency_s": round(time.monotonic() - t0, 2),
                  "usage": _usage_of(resp)})
        return resp
    except TimeoutError:
        _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                  "error": {"kind": "timeout", "limit_s": call_timeout}})
        raise _guard_killed(stage, "no completion within the call budget",
                            call_timeout, call_timeout) from None
    except UserFacingError:
        raise
    except Exception as e:  # noqa: BLE001
        _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                  "error": _provider_error_detail(e)})
        friendly = translate_llm_error(e, stage, cfg.provider, cfg.model, _BASE_URLS.get(cfg.provider))
        if friendly is not None:
            raise friendly from e
        raise
    finally:
        stop_heartbeat()


def _ollama_keep_alive() -> str:
    """Pin the model between calls: VMODEL_OLLAMA_KEEP_ALIVE, default 30m.
    Without it the idle timer (5m default) can unload the model out from
    under a long in-flight request - and a model stuck 'Stopping...' hangs
    the client until the wall-clock budget (ollama#14364/#12179/#8178;
    2026-09-10: T-1 hung 6+ minutes while `ollama ps` showed 'Stopping...')."""
    return _env("VMODEL_OLLAMA_KEEP_ALIVE") or "30m"


def _ollama_num_ctx() -> int:
    """Explicit context window for ollama calls, default 32768. Ollama's
    server default is 4096 and it silently TRUNCATES prompts past that
    (2026-09-10: `ollama ps` showed CONTEXT 4096 while stage prompts run
    larger - the model never saw whole requirements, which surfaced as
    contract failures/retries). Override with VMODEL_OLLAMA_NUM_CTX.
    Note: the native /api/chat path honors this per request; the
    OpenAI-compat endpoint only forwards num_ctx once ollama PR #16825
    lands - until then the server-side lever for that path is
    OLLAMA_CONTEXT_LENGTH=32768 on `ollama serve`."""
    try:
        return int(_env("VMODEL_OLLAMA_NUM_CTX") or "32768")
    except ValueError:
        return 32768


def _ollama_think() -> bool:
    """qwen3-class ollama models think by default - minutes of silent <think>
    before any output, dead time for structured generation (2026-09-10: a
    qwen3:8b items-plan call sat silent 4+ minutes and read as hung). Off by
    default; VMODEL_OLLAMA_THINK=1 restores provider-default thinking."""
    return _env("VMODEL_OLLAMA_THINK") == "1"


class _StructuredOutputAdapter:
    """Normalize LangChain native structured output back to an AIMessage.

    Anthropic returns ``{raw, parsed, parsing_error}`` when include_raw is on.
    Keeping the raw message preserves finish reasons and token usage for the
    existing truncation and telemetry path; serializing parsed makes provider
    behavior identical to the OpenAI-compatible JSON-schema route.
    """
    def __init__(self, runnable):
        self.runnable = runnable

    def invoke(self, messages):
        import json
        result = self.runnable.invoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        error = result.get("parsing_error") if isinstance(result, dict) else None
        if error is not None or parsed is None:
            return raw if raw is not None else result
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump(mode="json")
        raw.content = json.dumps(parsed)
        return raw


def build_chat_model(stage: str, schema: dict | None = None):
    """Return a LangChain chat model for the stage.

    `schema` is enforced provider-side. Ollama's OpenAI-compatible endpoint
    supports JSON Schema through response_format; DeepSeek/OpenAI use the same
    interface. The post-parse checks remain the safety net, not the formatter.
    """
    cfg = for_stage(stage)
    _preflight(cfg, stage)
    if cfg.provider in ("deepseek", "openai", "ollama"):
        from langchain_openai import ChatOpenAI

        env = _API_KEY_ENV[cfg.provider]
        extra = {}
        if cfg.provider == "ollama":
            body = {"num_ctx": _ollama_num_ctx(),  # honored once ollama PR #16825 lands
                    "keep_alive": _ollama_keep_alive()}
            if not _ollama_think():
                body["think"] = False
            extra = {"extra_body": body}
        if schema is not None:
            extra["model_kwargs"] = {"response_format": {"type": "json_schema", "json_schema":
                                     {"name": f"interview_{stage}_turn", "strict": True, "schema": schema}}}
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=_guard_limits()[0],  # the socket dies with the guardrail deadline
            max_tokens=_max_tokens_limit(cfg.max_tokens),
            base_url=_BASE_URLS[cfg.provider],
            api_key=(_env(env) or "" if env else "") or "ollama",
            **extra,
        )
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=_guard_limits()[0],  # the socket dies with the guardrail deadline
            max_tokens=_max_tokens_limit(cfg.max_tokens),
            api_key=_env(_API_KEY_ENV["anthropic"]) or "",
        )
        if schema is not None:
            return _StructuredOutputAdapter(model.with_structured_output(schema, include_raw=True))
        return model
    raise ValueError(f"unsupported provider {cfg.provider!r}")

