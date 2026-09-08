"""Per-stage model routing. THE model-agnostic layer: swapping providers is a
config change here, never a refactor. Providers:

- deepseek  (default - demo prices, OpenAI-compatible)
- openai
- anthropic
- ollama    (LOCAL: offline test runs at $0 - no API key, no data leaves the
             machine. Needs a running Ollama server + a pulled model.)

Every stage's choice is deliberate (see ARCHITECTURE.md, ADR-0004):
- probing / distillation earn the strong model - they are the moat.
- phrasing / tiebreaks can drop to a cheaper tier without visible quality loss.
Local Ollama runs are for development and tests; judge demo quality on the
real provider before showing founders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StageModel:
    provider: str          # 'deepseek' | 'openai' | 'anthropic' | 'ollama'
    model: str
    temperature: float = 0.4
    max_tokens: int = 2048


_STAGE_DEFAULTS: dict[str, StageModel] = {
    "probing":       StageModel("deepseek", "deepseek-chat", temperature=0.5),
    "distillation":  StageModel("deepseek", "deepseek-chat", temperature=0.2),
    "phrasing":      StageModel("deepseek", "deepseek-chat", temperature=0.7),
    "scope_tiebreak": StageModel("deepseek", "deepseek-chat", temperature=0.0),
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
    override = os.environ.get(f"INTERVIEW_ENGINE_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def _preflight(cfg: "StageModel", stage: str) -> None:
    """Fail BEFORE the call, with the fix named. An empty remote key would
    otherwise surface as a raw SDK 401 ten calls deep."""
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not os.environ.get(env):
        raise UserFacingError(
            f"stage '{cfg.model}' is routed to provider '{cfg.provider}', but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... or run this stage offline via Ollama: "
            f"INTERVIEW_ENGINE_MODEL_{stage.upper()}=ollama:qwen2.5-coder:7b (see README 'Local runs with Ollama').",
        )


def invoke_checked(chat_model, messages, stage: str):
    """chat_model.invoke with the known pains translated to human language."""
    from .errors import UserFacingError, translate_llm_error

    cfg = for_stage(stage)
    try:
        return chat_model.invoke(messages)
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
            max_tokens=cfg.max_tokens,
            base_url=_BASE_URLS[cfg.provider],
            api_key=(os.environ.get(env, "") if env else "") or "ollama",
        )
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=os.environ.get(_API_KEY_ENV["anthropic"], ""),
        )
    raise ValueError(f"unsupported provider {cfg.provider!r}")
