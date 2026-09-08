"""Per-stage model routing. THE model-agnostic layer: swapping providers is a
config change here, never a refactor. DeepSeek first (demo prices), any
OpenAI-compatible or Anthropic model after.

Every stage's choice is deliberate (see ARCHITECTURE.md, ADR-0004):
- probing / distillation earn the strong model - they are the moat.
- phrasing / tiebreaks can drop to a cheaper tier without visible quality loss.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StageModel:
    provider: str          # 'deepseek' | 'openai' | 'anthropic'
    model: str
    temperature: float = 0.4
    max_tokens: int = 2048


# Defaults: DeepSeek everywhere at demo prices. 'strong'/'cheap' split exists
# so a later quality pass can move probing/distillation to Sonnet-class by
# editing two lines, not two modules.
_STAGE_DEFAULTS: dict[str, StageModel] = {
    "probing":       StageModel("deepseek", "deepseek-chat", temperature=0.5),
    "distillation":  StageModel("deepseek", "deepseek-chat", temperature=0.2),
    "phrasing":      StageModel("deepseek", "deepseek-chat", temperature=0.7),
    "scope_tiebreak": StageModel("deepseek", "deepseek-chat", temperature=0.0),
}

_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": None,  # SDK default
}
_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def for_stage(stage: str) -> StageModel:
    if stage not in _STAGE_DEFAULTS:
        raise KeyError(f"unknown stage {stage!r}; add it explicitly - silent fallback models are how budgets leak")
    cfg = _STAGE_DEFAULTS[stage]
    # Env override: INTERVIEW_ENGINE_MODEL_PROBING=openai:gpt-5 etc. Config, not code.
    override = os.environ.get(f"INTERVIEW_ENGINE_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def build_chat_model(stage: str):
    """Return a LangChain chat model for the stage. Import-time lazy so the
    deterministic parts of the module never pay for SDK imports."""
    cfg = for_stage(stage)
    if cfg.provider in ("deepseek", "openai"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            base_url=_BASE_URLS[cfg.provider],
            api_key=os.environ.get(_API_KEY_ENV[cfg.provider], ""),
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
