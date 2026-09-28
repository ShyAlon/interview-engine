"""Pin the session model map resolved from shared stage env at HTTP start."""
from __future__ import annotations
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Mapping
from .errors import UserFacingError

CATALOG = {"deepseek": ("deepseek", "deepseek-flash"), "ollama": ("ollama", "qwen2.5-coder:14b")}
STAGES = ("probing", "distillation", "phrasing", "scope_tiebreak", "expert_opinion", "conops")
_active: ContextVar[Mapping[str,tuple[str,str]] | None] = ContextVar("session_model_map", default=None)

def resolve(route:str)->dict[str,tuple[str,str]]:
    """Resolve stage env first, catalog second, then pin it forever.

    Real process environment wins over .env via model_config._env. If a stage
    key is absent, other configured stages provide the one shared model. If
    all stage keys are absent, DELIVERY_OLLAMA_MODEL (Ollama only) precedes
    catalog. The selected dropdown route must match every configured stage provider.
    """
    if route not in CATALOG:
        raise UserFacingError(f"unsupported interview model route {route!r}.","Only DeepSeek and Ollama are available.","Choose a supported model route.")
    from .model_config import _env
    fallback=CATALOG[route]
    if route=="ollama":
        fallback=("ollama",_env("DELIVERY_OLLAMA_MODEL") or fallback[1])
    models={}
    chosen={}
    for stage in STAGES:
        key=f"INTERVIEW_ENGINE_MODEL_{'DISTILLATION' if stage=='conops' else stage.upper()}"
        raw=_env(key)
        if raw is not None:
            provider,sep,model=raw.strip().partition(":")
            provider,model=provider.strip(),model.strip()
            if not sep or not provider or not model:
                raise UserFacingError(f"{key}={raw!r} is not provider:model.","The shared model configuration is invalid.","Fix the named .env entry before starting.")
            if provider != route:
                raise UserFacingError(f"selected route {route!r} conflicts with {key}={raw!r}.","The dropdown and shared .env choose different providers.","Choose the matching provider or edit the named .env setting before starting.")
            models[stage]=(provider,model)
            chosen[key]=raw
    # An incomplete .env is intentional: one uniquely configured model fills
    # the missing stages. Divergent configured values are rejected by name.
    configured=set(models.values())
    if len(configured)==1:
        inherited=next(iter(configured))
        for stage in STAGES:
            models.setdefault(stage,inherited)
    elif not configured:
        models={stage:fallback for stage in STAGES}
    distinct=set(models.values())
    if len(distinct)!=1:
        configuration=", ".join(f"{key}={value}" for key,value in sorted(chosen.items()))
        raise UserFacingError(f"interview stage models disagree: {configuration}.","Configured stage keys disagree on provider or model.","Set the named stage keys to the same model before starting.")
    return models

def validate(models:Mapping[str,tuple[str,str]], selected_route:str|None=None)->None:
    from . import model_config
    routes={tuple(pair) for pair in models.values() if isinstance(pair,(tuple,list)) and len(pair)==2}
    if len(routes)!=1:
        raise UserFacingError("session stages do not share one effective route.","The pinned route is inconsistent.","Start a new session.")
    if selected_route and (selected_route not in CATALOG or next(iter(routes))[0]!=selected_route):
        raise UserFacingError("pinned model does not match the selected route.","The session route is inconsistent.","Start a new session.")
    for stage in STAGES:
        pair=models.get(stage)
        if not isinstance(pair,(tuple,list)) or len(pair)!=2 or (pair[0] not in CATALOG or pair[0] != CATALOG[pair[0]][0] or not isinstance(pair[1],str) or not pair[1].strip()):
            raise UserFacingError(f"session model for {stage} is missing or unsupported.","The pinned route is incomplete.","Start a new session with an available model route.")
        provider,model=pair
        if provider != "ollama" and stage==STAGES[0]:
            # A valid API route with no price must fail before its first paid call.
            # Check the effective table (including VMODEL_PRICE_TABLE overrides),
            # not a duplicate model list that can drift from contracts.
            from vmodel_contracts import price_for
            try:price_for(model,provider)
            except (KeyError,ValueError) as exc:
                raise UserFacingError(f"No price for selected model {provider}/{model}.",
                    "The shared vmodel-contracts price table has no entry for this paid route.",
                    "Choose a supported model or update VMODEL_PRICE_TABLE with verified prices before starting.") from exc
        model_config._preflight(model_config.StageModel(provider,model),stage)
        if provider=="ollama" and stage==STAGES[0]:
            import httpx
            endpoint=model_config._BASE_URLS["ollama"].rstrip("/")+"/models"
            try:
                response=httpx.get(endpoint,timeout=2)
                response.raise_for_status()
                found={entry.get("id") for entry in response.json().get("data",[]) if isinstance(entry,dict)}
            except (httpx.HTTPError,ValueError,KeyError) as exc:
                raise UserFacingError(f"Ollama preflight could not list models at {endpoint}: {exc}","The selected local route cannot be verified.","Start Ollama and retry.") from exc
            if model not in found:
                raise UserFacingError(f"Ollama model {model!r} is unavailable at {endpoint}; available: {sorted(found)}","The selected local model is not installed.",f"Pull {model} or choose DeepSeek, then retry.")

@contextmanager
def use(models:Mapping[str,tuple[str,str]]):
    token=_active.set(models)
    try:yield
    finally:_active.reset(token)

def selected(stage:str):
    models=_active.get()
    if models is None:return None
    return models["distillation" if stage=="conops" else stage]
