"""A session-owned model map, never inferred from mutable process defaults."""
from __future__ import annotations
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Mapping
from .errors import UserFacingError

CATALOG = {"deepseek": ("deepseek", "deepseek-chat"), "ollama": ("ollama", "qwen2.5-coder:14b")}
STAGES = ("probing", "distillation", "phrasing", "scope_tiebreak", "expert_opinion", "conops")
_active: ContextVar[Mapping[str,tuple[str,str]] | None] = ContextVar("session_model_map", default=None)

def resolve(route:str)->dict[str,tuple[str,str]]:
    if route not in CATALOG:
        raise UserFacingError(f"unsupported interview model route {route!r}.","Only DeepSeek and Ollama are available.","Choose a supported model route.")
    pair=CATALOG[route]
    if route=="ollama":
        from .model_config import _env
        pair=("ollama", _env("DELIVERY_OLLAMA_MODEL") or pair[1])
    return {stage:pair for stage in STAGES}

def validate(models:Mapping[str,tuple[str,str]])->None:
    from . import model_config
    routes={tuple(pair) for pair in models.values() if isinstance(pair,(tuple,list)) and len(pair)==2}
    if len(routes)!=1:
        raise UserFacingError("session stages do not share one effective route.","The pinned route is inconsistent.","Start a new session.")
    for stage in STAGES:
        pair=models.get(stage)
        if not isinstance(pair,(tuple,list)) or len(pair)!=2 or (pair[0] not in CATALOG or pair[0] != CATALOG[pair[0]][0] or not isinstance(pair[1],str) or not pair[1].strip()):
            raise UserFacingError(f"session model for {stage} is missing or unsupported.","The pinned route is incomplete.","Start a new session with an available model route.")
        provider,model=pair
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
