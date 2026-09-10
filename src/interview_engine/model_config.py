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

import os
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
# hang at full GPU; the same hang on a paid API is a bill. All three knobs are
# env/.env-overridable; the defaults match "demo on a 16GB laptop, or cents".
_CALL_TIMEOUT_S = 600   # wall-clock budget per call, client retries included
_STALL_TIMEOUT_S = 180  # a stream this quiet is hung, not thinking
_MAX_CALLS = 200        # what ONE process run may spend, in LLM calls
_CALLS_USED = 0


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
    # Env override: INTERVIEW_ENGINE_MODEL_PROBING=ollama:qwen2.5-coder:14b etc.
    override = _env(f"INTERVIEW_ENGINE_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


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


def invoke_checked(chat_model, messages, stage: str):
    """chat_model.invoke under the guardrails (budget + deadline), with the
    known pains translated to human language."""
    from .errors import translate_llm_error

    cfg = for_stage(stage)
    _guard_enter(stage)
    call_timeout = _guard_limits()[0]
    try:
        return _run_with_deadline(lambda: chat_model.invoke(messages), call_timeout, stage)
    except TimeoutError:
        raise _guard_killed(stage, "no completion within the call budget",
                            call_timeout, call_timeout) from None
    except Exception as e:  # noqa: BLE001 - translated or re-raised
        friendly = translate_llm_error(e, stage, cfg.provider, cfg.model, _BASE_URLS.get(cfg.provider))
        if friendly is not None:
            raise friendly from e
        raise


def build_chat_model(stage: str):
    """Return a LangChain chat model for the stage. Import-time lazy so the
    deterministic parts of the module never pay for SDK imports."""
    cfg = for_stage(stage)
    _preflight(cfg, stage)
    if cfg.provider in ("deepseek", "openai", "ollama"):
        from langchain_openai import ChatOpenAI

        env = _API_KEY_ENV[cfg.provider]
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=_guard_limits()[0],  # the socket dies with the guardrail deadline
            max_tokens=cfg.max_tokens,
            base_url=_BASE_URLS[cfg.provider],
            api_key=(_env(env) or "" if env else "") or "ollama",
        )
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=_guard_limits()[0],  # the socket dies with the guardrail deadline
            max_tokens=cfg.max_tokens,
            api_key=_env(_API_KEY_ENV["anthropic"]) or "",
        )
    raise ValueError(f"unsupported provider {cfg.provider!r}")

