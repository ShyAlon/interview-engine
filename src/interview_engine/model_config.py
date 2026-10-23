"""Console-stage model defaults and shared model client configuration.
HTTP interviews pin a session-owned route; .env cannot reroute a live session. Providers:

- ollama    (DEFAULT - local-first, the product owner's standing call:
             "It should all be ollama by default. We'll change it once we go
             to production." $0, no API key, no data leaves the machine.
             Needs a running Ollama server + a pulled model.)
- deepseek  (opt-in - demo prices, OpenAI-compatible)
- openai    (opt-in)
- anthropic (opt-in)

Console-stage overrides remain for non-HTTP workflows (see ARCHITECTURE.md, ADR-0004):
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
from contextvars import ContextVar
from dataclasses import dataclass

_DOTENV: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    """Config files, parsed once per process, most-shared first:

    1. <require-root>/.env - the SHARED config one directory up
       (system keys and endpoints live there; HTTP model routing belongs
       to each persisted session). Present only in the sibling-repo layout; absent in a
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
    from the shared config remains available for API keys and endpoints."""
    return os.environ.get(key) or _dotenv().get(key)


# ---------------------------------------------------------------- guardrails
# Every LLM call passes through these limits. A hung ollama generation costs a
# hang at full GPU; the same hang on a paid API is a bill. All four knobs are
# env/.env-overridable; the defaults match "demo on a 16GB laptop, or cents".
_CALL_TIMEOUT_S = 600   # wall-clock budget per call, client retries included
_STALL_TIMEOUT_S = 180  # a stream this quiet is hung, not thinking
_MAX_CALLS = 200        # what ONE process run may spend, in LLM calls
_CALLS_USED = 0

# Shared conservative policy: 8x the same-stage running median, never less
# than four minutes, including cold start and unusually quick early calls.
from vmodel_contracts.runtime import AdaptiveCallWatchdog, AdaptiveWatchdogTimeout
_ADAPTIVE_WATCHDOG = AdaptiveCallWatchdog(multiplier=8.0, floor_s=240.0)


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
        "the interview task model picker; console runs use explicit CLI settings.",
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


_session_llm_log: ContextVar[str | None] = ContextVar("session_llm_log", default=None)
_progress_heartbeat: ContextVar[object | None] = ContextVar("progress_heartbeat", default=None)

def _llm_log_sink() -> str | None:
    """VMODEL_LLM_LOG: unset = off (default, zero cost); 'stderr' = console;
    any other value = a file path, one JSON object per line."""
    return _session_llm_log.get() or _env("VMODEL_LLM_LOG")


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
_DEBUG_HTTP_ANNOUNCED: set[str] = set()


def _console_level(msg: str) -> str | None:
    """Conservative severity classification for shared terminal colors."""
    lower = msg.lower()
    if any(token in lower for token in ("error", "failed", "failure", "!!")):
        return "error"
    if any(token in lower for token in ("warning", "fallback", "retry", "still working", "simulated")):
        return "warning"
    if " ok " in f" {lower} " or "success" in lower or "approved" in lower:
        return "success"
    return None


def _console_line(msg: str) -> None:
    """One stderr line, atomically - the heartbeat thread and the progress
    printer share this lock so lines never interleave mid-call."""
    from vmodel_contracts.runtime import format_console_line

    with _CONSOLE_LOCK:
        sys.stderr.write(format_console_line(msg, level=_console_level(msg), stream=sys.stderr) + "\n")
        sys.stderr.flush()


def _debug_http_path() -> str | None:
    """Current payload-bearing debug sink, only when VMODEL_DEBUG_HTTP opts in."""
    from vmodel_contracts.runtime import debug_http_enabled, debug_http_path
    return str(debug_http_path(_llm_log_sink())) if debug_http_enabled() else None


def log_supported_prose_response(stage: str, resp) -> None:
    """Visible degraded mode: a usable single question arrived without JSON."""
    response_id = getattr(resp, "id", None) or (getattr(resp, "response_metadata", None) or {}).get("id") or "unknown"
    _console_line(f"  WARNING response mode [{stage}]: clean question without JSON envelope; "
                  f"response_id={response_id}; coverage metadata requested separately")


def log_malformed_response(stage: str, kind: str, resp, *, snippet: str | None = None) -> None:
    """Make a malformed provider reply visible immediately, without prompts."""
    response_id = getattr(resp, "id", None) or (getattr(resp, "response_metadata", None) or {}).get("id") or "unknown"
    text = f"  ERROR malformed response [{stage}]: {kind}; response_id={response_id}"
    if snippet:
        text += f"; reply began: {snippet.strip()[:160]!r}"
    debug_path = _debug_http_path()
    if debug_path:
        text += f"; full HTTP exchange: {debug_path}"
    _console_line(text)


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
    sink = _progress_heartbeat.get()  # capture before the daemon thread starts

    def _beat() -> None:
        while not stop.wait(interval):
            if stop.is_set():  # a beat already past the wait must not print after the call returned
                break
            elapsed=time.monotonic() - t0
            _console_line(f"  ... still working - {label}, {elapsed:.0f}s elapsed")
            if sink is not None:
                try:
                    from .stop_fence import guarded_publish
                    guarded_publish(lambda:sink(label, label, elapsed))
                except Exception:pass

    import contextvars
    context=contextvars.copy_context()
    threading.Thread(target=lambda:context.run(_beat), daemon=True).start()
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
                out[k] = _mask_secrets(str(err[k]))
    if "message" not in out:
        out["message"] = _mask_secrets(str(exc))
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
    from .session_runtime import current
    runtime=current()
    if runtime is not None:
        with runtime.lock:
            runtime.calls+=1;calls=runtime.calls
    else:
        _CALLS_USED+=1;calls=_CALLS_USED
    if calls > _guard_limits()[2]:
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
        "provider using the interview task model picker.",
    )


def _run_with_deadline(fn, timeout_s: int, stage: str):
    """Wall-clock deadline around one blocking call. The worker thread cannot
    be force-killed in Python, so the client ALSO gets the same timeout (see
    build_chat_model) - the socket dies, the thread unwinds, and the caller
    gets an informative error instead of a hang."""
    import concurrent.futures
    import contextvars

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    context = contextvars.copy_context()
    from .stop_fence import track_inflight_worker
    release=track_inflight_worker()
    try:future = pool.submit(context.run, fn)
    except BaseException:
        release();pool.shutdown(wait=False,cancel_futures=True);raise
    future.add_done_callback(lambda done:release())
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _set_model_timeout(chat_model, timeout_s: float) -> None:
    """Put the adaptive deadline on the live provider socket."""
    model = getattr(chat_model, "runnable", chat_model)
    try:
        model.request_timeout = timeout_s
        root = getattr(model, "root_client", None)
        if root is not None:
            root.timeout = timeout_s
            if getattr(root, "_client", None) is not None:
                import httpx
                root._client.timeout = httpx.Timeout(timeout_s)
    except Exception:
        pass
    try:
        import httpx
        model._client._client.timeout = httpx.Timeout(timeout_s)
    except Exception:
        pass


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
    "expert_opinion": StageModel("ollama", "qwen2.5-coder:14b", temperature=0.2),
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
    from .session_models import selected
    pinned = selected(stage)
    if pinned:
        return StageModel(*pinned, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    # Console-era stage keys remain as CLI defaults only. HTTP sessions use
    # the pinned map above, so no env key can override a session choice.
    # ConOps follows distillation in the legacy CLI path.
    route = "distillation" if stage == "conops" else stage
    override = _env(f"INTERVIEW_ENGINE_MODEL_{route.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return _STAGE_DEFAULTS[route] if stage == "conops" else cfg


def _preflight(cfg: "StageModel", stage: str) -> None:
    """Fail BEFORE the call, with the fix named. An empty remote key would
    otherwise surface as a raw SDK 401 ten calls deep."""
    _check_db_dispatch()
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not _env(env):
        raise UserFacingError(
            f"stage '{cfg.model}' is routed to provider '{cfg.provider}', but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... - remote providers are opt-in (the default is local Ollama); "
            "or select an available local model in the interview task. For console runs, fix the named CLI model setting.",
        )


def describe_models() -> str:
    """What will actually run, resolved per stage after env/.env layering -
    printed at startup so the log always records which provider/model ran."""
    return ", ".join(f"{s}={for_stage(s).provider}/{for_stage(s).model}" for s in _STAGE_DEFAULTS)


import contextvars as _contextvars

# PR B staged provider responses: set by the application when a DB-backed session
# runs. Holds a StagingContext; when unset, behavior is exactly the pre-staging path.
# The dispatch guard (_db_dispatch_block) is unchanged; staging is what will make
# lifting it safe, under a separate approved enable.
_response_staging = _contextvars.ContextVar("response_staging", default=None)


class StagingContext:
    """Per-application-instance staging state.

    Logical call identity = (session, generation, execution_point, node, stage,
    item, ordinal). `execution_point` is the DURABLE checkpoint id the graph
    execution resumed from ('bootstrap' before the first checkpoint): the
    application sets it from its pinned config before each advance, and the
    checkpoint saver advances it as super-steps commit, so a crash-resume pins
    the same published checkpoint and re-executes the interrupted node under the
    SAME point (its calls replay), while a new turn resumes from a NEW committed
    checkpoint (identical prompts are new paid calls, never replays). `node` is
    the issuing graph node from the LangGraph runnable config, so two parallel
    nodes under one execution point cannot collide. `ordinal` is the position of
    the call within that node execution's (stage,item) sequence: `cursors` maps
    (execution_point, node, stage, item) to the next ordinal. A fresh instance
    after a crash starts its cursors at 0 and replays the durable log under the
    pinned execution point; nothing about the identity is process-local."""

    def __init__(self, session_id: str, store, claim):
        self.session_id = session_id
        self.store = store
        self.claim = claim
        self.generation = claim.generation
        self.anchor = 'bootstrap'
        self.step = 0
        self.cursors: dict = {}

    @property
    def execution_point(self) -> str:
        """anchor:super-step - both halves deterministic across re-execution.
        The anchor is the advance's durable checkpoint anchor (pinned on restore,
        published pointer while serving, 'bootstrap' first). The step is
        LangGraph's super-step counter read from the pregel scratchpad AT CALL
        TIME (checkpoint commits complete asynchronously and race the next
        super-step's calls; the scratchpad step does not). Resuming from the
        same anchor re-derives the same steps."""
        return f"{self.anchor}:{self.current_step()}"

    def set_execution_point(self, anchor, step: int = 0):
        """Seed the durable anchor for an advance (and the fallback step for
        calls made outside graph execution, e.g. direct test drives)."""
        self.anchor = str(anchor) if anchor else 'bootstrap'
        self.step = step

    # The super-step and node identity rely on LangGraph private internals,
    # pinned and probed: langgraph 1.1.10 / langchain-core 1.6.3 (uv.lock).
    # tests/test_pg_staged_loop_replay.py proves the contract against real graph
    # executions; re-run it before any version bump. A missing/malformed
    # scratchpad IN GRAPH is a loud failure - a fallback counter would silently
    # re-key paid-call identity.
    @staticmethod
    def _runnable_config() -> dict:
        from langchain_core.runnables.config import var_child_runnable_config
        return var_child_runnable_config.get(None) or {}

    def current_step(self) -> int:
        """The current super-step from LangGraph's runnable config; the seeded
        value only for calls made entirely outside graph execution."""
        configurable = self._runnable_config().get('configurable') or {}
        if not configurable:
            return self.step  # outside graph execution (direct test drives)
        scratchpad = configurable.get('__pregel_scratchpad')
        step = getattr(scratchpad, 'step', None)
        if not isinstance(step, int) or step < 0:
            raise RuntimeError(
                'LangGraph super-step unreadable: in-graph provider call without '
                f'a valid __pregel_scratchpad.step (scratchpad={scratchpad!r}). '
                'Pinned/probed at langgraph 1.1.10 + langchain-core 1.6.3; '
                're-probe the private API before any upgrade - no fallback is '
                'safe for paid-call identity.')
        return step

    @staticmethod
    def current_node() -> str:
        """The issuing graph node from LangGraph's runnable config contextvar
        (checkpoint_ns prefix). '' only outside graph execution."""
        configurable = StagingContext._runnable_config().get('configurable') or {}
        if not configurable:
            return ''  # outside graph execution (direct calls)
        ns = configurable.get('checkpoint_ns') or ''
        node = ns.split(':')[0] if ns else ''
        if not node:
            raise RuntimeError(
                'LangGraph node identity unreadable: in-graph provider call '
                'without checkpoint_ns. Pinned/probed at langgraph 1.1.10 + '
                'langchain-core 1.6.3; re-probe before any upgrade.')
        return node

    def next_cursor(self, node: str, stage: str, item: str):
        key = (self.execution_point, node, stage, item)
        idx = self.cursors.get(key, 0)
        return key, idx


def _staging_request_sha(cfg, messages, stage: str, item, params: dict | None = None) -> str:
    """Hash the FULL effective request: provider, model, generation ceiling and the
    serialized messages. A replay matches only the identical paid call."""
    import hashlib, json
    from langchain_core.messages import BaseMessage, message_to_dict
    def _wire(message):
        # Semantic request content only: the provider never sees the locally
        # generated message id, and a crash-resume rebuilds messages with fresh
        # ids - including it would fingerprint-pause a legitimate replay.
        if not isinstance(message, BaseMessage):
            return str(message)
        d = message_to_dict(message)
        d.get('data', d).pop('id', None)
        return d
    body = json.dumps({"provider": cfg.provider, "model": cfg.model, "stage": stage,
                       "item": item, "max_tokens": _max_tokens_limit(cfg.max_tokens),
                       # Every provider-significant generation parameter the call
                       # was built with (temperature, tools, response_format,
                       # extra_body/provider route settings): a replay must match
                       # the identical paid call, not merely the same prompt.
                       "params": params or {},
                       "messages": [_wire(m) for m in messages]},
                      sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def invoke_checked(chat_model, messages, stage: str, item: str | None = None):
    """Invoke under guardrails, retrying one provider-empty completion.

    DeepSeek documents occasional empty content in JSON mode. Treat that as a
    retryable provider signal, log the response envelope, and never let it
    masquerade as a downstream contract failure.
    """
    _check_db_dispatch()
    from .errors import UserFacingError, translate_llm_error

    cfg = for_stage(stage)
    call_timeout = _guard_limits()[0]
    for attempt in (1, 2):
        adaptive_deadline, stage_median = _watchdog().deadline(stage)
        effective_timeout = min(call_timeout, adaptive_deadline)
        _guard_enter(stage)
        base = {"provider": cfg.provider, "model": cfg.model, "stage": stage,
                "attempt": attempt, **({"item": item} if item else {}),
                **({"retry_reason": "empty_response"} if attempt == 2 else {})}
        _log_llm({"event": "llm_call_start", **base})
        stop_heartbeat = _start_heartbeat(f"stage '{stage}'" + (f" [{item}]" if item else "") + " call")
        t0 = time.monotonic()
        try:
            _set_model_timeout(chat_model, effective_timeout)
            resp = _run_with_deadline(lambda: __import__("interview_engine.attempt_admission",fromlist=["checked_model_invoke"]).checked_model_invoke(chat_model,messages,stage=stage,item=item,provider=cfg.provider,model=cfg.model,max_output_tokens=_max_tokens_limit(cfg.max_tokens)), effective_timeout, stage)
            elapsed = time.monotonic() - t0
            _watchdog().complete(stage, elapsed)
            _check_truncated(resp, stage, cfg)
            content = getattr(resp, "content", None)
            if isinstance(content, str) and not content.strip():
                log_malformed_response(stage, "empty content", resp)
                _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                          "error": {"kind": "empty_response", "raw_payload": __import__("vmodel_contracts.runtime", fromlist=["empty_llm_response_detail"]).empty_llm_response_detail(resp, _mask_secrets)}})
                if attempt == 1:
                    continue
                guidance = (_rerun_guidance(stage) + "; DeepSeek documents occasional empty JSON-mode "
                            "completions, so a fresh call may succeed" if cfg.provider == "deepseek" else
                            _rerun_guidance(stage) + "; if it repeats, check the selected provider/model")
                raise UserFacingError(
                    f"stage '{stage}': '{cfg.provider}/{cfg.model}' returned empty content twice.",
                    "the provider completed both requests without an answer; the raw response envelopes "
                    "are recorded in llm-calls.jsonl.", guidance)
            _log_llm({"event": "llm_call_end", **base, "latency_s": round(time.monotonic() - t0, 2),
                      "usage": _usage_of(resp)})
            return resp
        except TimeoutError:
            elapsed = time.monotonic() - t0
            adaptive = effective_timeout < call_timeout or adaptive_deadline <= call_timeout
            _log_llm({"event": "llm_call_error", **base, "latency_s": round(elapsed, 2),
                      "error": {"kind": "adaptive_watchdog" if adaptive else "timeout", "limit_s": effective_timeout,
                                "stage_median_s": stage_median}})
            if adaptive and attempt == 1:
                ratio = elapsed / stage_median if stage_median else None
                basis = f"{ratio:.1f}x stage median" if ratio is not None else "cold-start floor"
                _console_line(f"  WARNING boundary=watchdog stage={stage} item={item or '(none)'} "
                              f"elapsed={elapsed:.1f}s median={stage_median if stage_median is not None else 'none'}s "
                              f"multiplier={getattr(_watchdog(), 'multiplier', 8):g} floor={getattr(_ADAPTIVE_WATCHDOG, 'floor_s', 240):g}s "
                              f"absolute_cap={call_timeout}s deadline={effective_timeout:.1f}s "
                              f"decision=kill-retry reason={basis}")
                continue
            if adaptive:
                raise _guard_killed(stage, "adaptive watchdog retry exhausted", elapsed, effective_timeout) from None
            raise _guard_killed(stage, "no completion within the call budget", elapsed, call_timeout) from None
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



class _DeepSeekToolOutputAdapter:
    """Expose one forced DeepSeek function call as ordinary JSON content.

    Function parameters must be an object, while our artifact schemas can also
    be arrays, so every schema is wrapped in a code-owned ``payload`` property.
    The rest of the pipeline continues to parse and validate plain JSON.
    """
    def __init__(self, model, schema: dict, name: str):
        self.name = name
        tool = {"type": "function", "function": {"name": name,
                "description": "Return the requested structured artifact.",
                "parameters": {"type": "object", "properties": {"payload": schema},
                               "required": ["payload"], "additionalProperties": False}}}
        self.model = model
        self.tool = tool
        self.runnable = model.bind_tools([tool], tool_choice=name)

    def __getattr__(self, name):
        # Preserve observability/test access to the underlying transport.
        return getattr(self.model, name)

    def _content(self, message):
        import json
        calls = getattr(message, "tool_calls", None) or []
        if len(calls) != 1 or calls[0].get("name") != self.name:
            # Empty content enters the existing bounded human-readable retry;
            # never disguise a missing/wrong tool call as a valid artifact.
            return ""
        args = calls[0].get("args") or {}
        if "payload" not in args:
            return ""
        return json.dumps(args["payload"])

    def invoke(self, messages):
        message = self.runnable.invoke(messages)
        message.content = self._content(message)
        return message

    def stream(self, messages):
        # LangChain assembles streamed tool-call argument fragments via chunk
        # addition. Emit one normalized final chunk so downstream sees exactly
        # one JSON artifact, never partial function syntax.
        from langchain_core.messages import AIMessageChunk
        combined = None
        for chunk in self.runnable.stream(messages):
            combined = chunk if combined is None else combined + chunk
        if combined is None:
            yield AIMessageChunk(content="")
            return
        yield AIMessageChunk(content=self._content(combined),
                             response_metadata=getattr(combined, "response_metadata", {}) or {},
                             usage_metadata=getattr(combined, "usage_metadata", None))

_db_dispatch_block: ContextVar[bool] = ContextVar('db_dispatch_block', default=False)


class db_provider_dispatch_blocked:
    """Block every provider call while set. Database session persistence (PR A)
    never dispatches providers; staged responses land in PR B. The registry
    sets this around program construction and advances whenever a session
    runs on database persistence without the explicit internal test opt-in.
    This is not a feature flag and must not be exposed as one."""

    def __enter__(self):
        self._token = _db_dispatch_block.set(True)
        return self

    def __exit__(self, *exc):
        _db_dispatch_block.reset(self._token)
        return False


def _check_db_dispatch() -> None:
    if _db_dispatch_block.get():
        from .errors import UserFacingError
        raise UserFacingError(
            "database session mode does not call providers yet.",
            "Staged responses land in PR B; this guard is not a feature flag.",
            "Run the session in file persistence mode.")


def build_chat_model(stage: str, schema: dict | None = None):
    """Return a LangChain chat model for the stage.

    `schema` is enforced provider-side. Ollama's OpenAI-compatible endpoint
    supports JSON Schema through response_format; DeepSeek/OpenAI use the same
    interface. The post-parse checks remain the safety net, not the formatter.
    """
    _check_db_dispatch()
    cfg = for_stage(stage)
    from .wire_binding import validate_provider
    validate_provider(cfg.provider)
    _preflight(cfg, stage)
    if cfg.provider in ("deepseek", "openai", "ollama"):
        from langchain_openai import ChatOpenAI

        env = _API_KEY_ENV[cfg.provider]
        extra = {}
        if cfg.provider == "deepseek":
            # Classification and plain-text artifact calls need the same policy
            # as schema calls; the provider otherwise enables hidden thinking.
            extra["extra_body"] = {"thinking": {"type": "disabled"}}
        if cfg.provider == "ollama":
            body = {"num_ctx": _ollama_num_ctx(),  # honored once ollama PR #16825 lands
                    "keep_alive": _ollama_keep_alive()}
            if not _ollama_think():
                body["think"] = False
            extra = {"extra_body": body}
        if schema is not None:
            if cfg.provider == "deepseek":
                _console_line(f"  output mode [{stage}]: DeepSeek forced function call; field shape provider-constrained, semantic validation+retry retained")
            else:
                extra["model_kwargs"] = {"response_format": {"type": "json_schema", "json_schema":
                                         {"name": f"interview_{stage}_turn", "strict": True, "schema": schema}}}
        from .wire_binding import model_clients,register_model
        installed_clients=model_clients(stage=stage,provider=cfg.provider,model=cfg.model)
        if installed_clients is not None:extra.update(installed_clients)
        if cfg.provider in ("deepseek", "openai") and installed_clients is None:
            from vmodel_contracts.runtime import debug_http_client
            client = debug_http_client(llm_log=_llm_log_sink(), mask=_mask_secrets)
            if client is not None:
                extra["http_client"] = client
                debug_path = _debug_http_path()
                if debug_path not in _DEBUG_HTTP_ANNOUNCED:
                    _DEBUG_HTTP_ANNOUNCED.add(debug_path)
                    _console_line(f"  WARNING full LLM HTTP debug enabled: {debug_path} (request messages included; no authorization headers)")
        model = ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=_guard_limits()[0],
            **({"max_retries":0} if installed_clients is not None or os.environ.get("VMODEL_TENANT_OWNER") is not None or os.environ.get("INTERVIEW_ENGINE_TENANTS")=="1" else {}),  # the socket dies with the guardrail deadline
            max_tokens=_max_tokens_limit(cfg.max_tokens),
            base_url=_BASE_URLS[cfg.provider],
            api_key=(_env(env) or "" if env else "") or "ollama",
            **extra,
        )
        if cfg.provider == "deepseek" and schema is not None:
            model=_DeepSeekToolOutputAdapter(model, schema, f"return_{stage}_artifact")
        return register_model(model,stage=stage,provider=cfg.provider,model=cfg.model)
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=_guard_limits()[0],
            **({"max_retries":0} if os.environ.get("VMODEL_TENANT_OWNER") is not None or os.environ.get("INTERVIEW_ENGINE_TENANTS")=="1" else {}),  # the socket dies with the guardrail deadline
            max_tokens=_max_tokens_limit(cfg.max_tokens),
            api_key=_env(_API_KEY_ENV["anthropic"]) or "",
        )
        if schema is not None:
            return _StructuredOutputAdapter(model.with_structured_output(schema, include_raw=True))
        return model
    raise ValueError(f"unsupported provider {cfg.provider!r}")



def _watchdog():
    from .session_runtime import current
    runtime=current()
    return runtime.watchdog if runtime is not None else _ADAPTIVE_WATCHDOG
