"""Existing general-LLM and rules adapters for categorization slots."""
from __future__ import annotations
import json,time
from . import model_config
from .categorization import ClassificationAnswer,ClassificationRequest
from .coverage import founder_control,founder_steer
from .repair import extract_json_value


def llm_answers(request: ClassificationRequest, chat_model, usage_log):
    labels = {}
    for key,q in request.questions.items():
        if hasattr(q,"criteria") and isinstance(q.criteria,dict): labels[key]={"choice":list(q.criteria),"instructions":q.instructions}
        else: labels[key]={"noul":"number 0..1","instructions":q.instructions}
    prompt=("Classify the supplied state. Return one strict JSON object with exactly these keys. "
            "Choice values must be one listed label; noul values are numbers 0..1. Include a parallel "
            "_confidence object with a 0..1 value for each key.\nCONTRACT:\n"+json.dumps(labels)+"\nSTATE:\n"+request.state)
    t=time.time();resp=model_config.invoke_checked(chat_model,prompt,"probing")
    cfg=model_config.for_stage("probing");usage_log.record("categorization_"+request.slot,cfg.provider,cfg.model,getattr(resp,"usage_metadata",{}) or {},duration_s=time.time()-t)
    data=json.loads(extract_json_value(resp.content));conf=data.pop("_confidence",{})
    out={}
    for key,value in data.items():
        q=request.questions[key]
        # Providers sometimes honor the semantic question type by returning a
        # typed envelope ({"choice": ...} / {"noul": ...}) even though this
        # adapter asks for raw values. Normalize before label membership: a
        # dict is unhashable and must never become a control-flow crash.
        nested_conf=None
        if isinstance(value,dict):
            nested_conf=value.get("confidence")
            candidates=("choice","noul","score","value")
            present=[name for name in candidates if name in value]
            if len(present)!=1:raise ValueError(f"invalid wrapped {key} answer keys {sorted(value)}")
            value=value[present[0]]
        if hasattr(q,"criteria") and isinstance(q.criteria,dict) and value not in q.criteria:raise ValueError(f"invalid {key} label {value!r}")
        confidence=conf.get(key,nested_conf)
        out[key]=ClassificationAnswer(value,float(confidence) if confidence is not None else None)
    if set(out)!=set(request.questions):raise ValueError("categorization response keys do not match request")
    return out


def intent_rules(request: ClassificationRequest):
    answer=request.state.split("Founder answer:",1)[-1].strip()
    control=founder_control(answer);steer=founder_steer(answer)
    intent=control or steer or "answer"
    return {"intent":ClassificationAnswer(intent,1.0)}


def unavailable_rules(request: ClassificationRequest):
    raise ValueError(f"no reliable deterministic rules for {request.slot}")
