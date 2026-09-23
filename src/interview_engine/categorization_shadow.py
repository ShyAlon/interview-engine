"""Non-authoritative Jev shadow comparison and adjudication-ready reporting."""
from __future__ import annotations
import hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,is_dataclass
from datetime import datetime,timezone
from pathlib import Path

RECORD_SCHEMA_VERSION=2
RUBRIC_VERSION="categorization-request-v1"


def _side(result=None, *, elapsed=None, usage=None, error=None):
    if error is not None:
        return {"status":"error","error":{"type":type(error).__name__,"message":str(error)},
                "provider":getattr(error,"provider",None),"model":None,"verdict":{},"confidence":{},
                "input_tokens":0,"output_tokens":0,"latency_s":elapsed,"cost_usd":None}
    tokens_in=getattr(result,"input_tokens",0);tokens_out=getattr(result,"output_tokens",0)
    cost=getattr(result,"cost_usd",None)
    if result.provider == "rules": cost=0.0
    if usage is not None:
        tokens_in=usage.prompt_tokens;tokens_out=usage.completion_tokens;cost=usage.cost_usd
    return {"status":"ok","provider":result.provider,"model":result.model,
            "verdict":{k:v.value for k,v in result.answers.items()},
            "confidence":{k:v.confidence for k,v in result.answers.items()},
            "input_tokens":tokens_in,"output_tokens":tokens_out,
            "latency_s":getattr(result,"latency_s",None) or elapsed,"cost_usd":cost,
            "pricing_status":getattr(result,"pricing_status","priced" if cost is not None else "unpriced")}


def _agreement(a,b):
    return a.get("status")==b.get("status")=="ok" and a.get("verdict")==b.get("verdict")


def _jsonable(value):
    if is_dataclass(value): return {k:_jsonable(v) for k,v in asdict(value).items()}
    if isinstance(value,dict): return {str(k):_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    return value


def input_package(request):
    """The complete, versioned evidence/rubric package used by both deciders."""
    questions=_jsonable(request.questions)
    rubric_json=json.dumps(questions,sort_keys=True,separators=(",",":"),default=str)
    return {"state":request.state,"context":_jsonable(request.context),"questions":questions,
            "rubric_version":getattr(request,"rubric_version","") or RUBRIC_VERSION,
            "rubric_ref":"sha256:"+hashlib.sha256(rubric_json.encode()).hexdigest()}


def valid_for_request(result, request) -> bool:
    """Status/schema check: every choice question answered with a listed label."""
    if result is None or not getattr(result,"answers",None):return False
    for key,q in (request.questions or {}).items():
        criteria=getattr(q,"criteria",None)
        if not isinstance(criteria,dict):continue
        a=result.answers.get(key)
        if a is None or str(a.value) not in criteria:return False
    return True


class ShadowCategorizer:
    """Run baseline and Jev concurrently on every decision and record both.

    By default the baseline is authoritative and Jev only observes. Slots
    listed in ``authoritative`` (2026-09-23: scope) return Jev's decision
    when it is ok and schema-valid; the baseline is the fallback on Jev
    status/schema errors only. ``last_jev[slot]`` exposes Jev's latest
    verdict so code-owned guards can compare the two sides."""
    def __init__(self, baseline, jev, jsonl_path, usage_log=None, report_path=None, *, session_id=None, schema_id=None, schema_version=None,
                 authoritative=()):
        self.baseline,self.jev,self.path,self.usage_log=baseline,jev,Path(jsonl_path),usage_log
        self.authoritative=frozenset(authoritative);self.last_jev={}
        self.report_path=Path(report_path) if report_path else self.path.with_suffix(".md")
        self.session_id,self.schema_id,self.schema_version=session_id,schema_id,schema_version
        self.path.parent.mkdir(parents=True,exist_ok=True)
    def active(self,slot): return self.baseline.active(slot)
    def classify(self,request):
        before=len(self.usage_log.records) if self.usage_log else 0
        def call(fn):
            t=time.monotonic()
            try:return fn(request),time.monotonic()-t,None
            except Exception as e:return None,time.monotonic()-t,e
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix="categorization-shadow") as pool:
            base_f=pool.submit(call,self.baseline.classify);jev_f=pool.submit(call,self.jev.classify)
            base,base_s,base_err=base_f.result();shadow,shadow_s,shadow_err=jev_f.result()
        usage=None
        if self.usage_log:
            new=self.usage_log.records[before:]
            usage=next((r for r in new if r.stage=="categorization_"+request.slot and r.provider!="typesafe"),None)
        baseline_side=_side(base,elapsed=base_s,usage=usage,error=base_err)
        jev_side=_side(shadow,elapsed=shadow_s,error=shadow_err)
        package=input_package(request)
        record={"record_schema_version":RECORD_SCHEMA_VERSION,"schema_id":self.schema_id,"schema_version":self.schema_version,"session_id":self.session_id,
                "timestamp":datetime.now(timezone.utc).isoformat(),"slot":request.slot,
                "input_ref":"sha256:"+hashlib.sha256(request.state.encode()).hexdigest(),"input":package,
                "baseline":baseline_side,"jev":jev_side,"agreement":_agreement(baseline_side,jev_side),
                "jev_fallback_count":0,"jev_error_count":int(shadow_err is not None)}
        use_jev=False
        if request.slot in self.authoritative:
            use_jev=shadow_err is None and valid_for_request(shadow,request)
            record["authority"]="jev" if use_jev else "baseline"
            record["jev_fallback_count"]=int(not use_jev)
        else:
            record["authority"]="baseline"
        self.last_jev[request.slot]=shadow if shadow_err is None else None
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(record,separators=(",",":"),default=str)+"\n");f.flush();os.fsync(f.fileno())
            tmp=self.report_path.with_suffix(self.report_path.suffix+".tmp")
            tmp.write_text(render_report([self.path])+"\n");tmp.replace(self.report_path)
        except Exception:
            pass
        if use_jev: return shadow
        if base_err: raise base_err
        return base


def read_records(paths):
    records=[]
    for path in paths:
        with Path(path).open() as f: records += [json.loads(line) for line in f if line.strip()]
    return records


def confidence_band(side):
    vals=[v for v in side.get("confidence",{}).values() if isinstance(v,(int,float))]
    if not vals:return "missing"
    value=min(vals)
    return "high" if value>=.85 else "medium" if value>=.6 else "low"


def adjudication_sample(records, agreements_per_stratum=2):
    """All disagreements plus deterministic agreement samples by slot/label/confidence."""
    picked=[r for r in records if not r.get("agreement")]
    strata={}
    for r in records:
        if not r.get("agreement"):continue
        label=json.dumps(r.get("baseline",{}).get("verdict",{}),sort_keys=True)
        key=(r.get("slot"),label,confidence_band(r.get("jev",{})))
        strata.setdefault(key,[]).append(r)
    for rs in strata.values():
        picked += sorted(rs,key=lambda r:r.get("input_ref",""))[:agreements_per_stratum]
    return picked


def render_report(paths):
    records=read_records(paths);generated=datetime.now(timezone.utc).isoformat()
    lines=["# Jev shadow report","",f"Generated: {generated}",f"Decisions: {len(records)}",""]
    for slot in sorted({r["slot"] for r in records}):
        rs=[r for r in records if r["slot"]==slot];ok=[r for r in rs if r["baseline"]["status"]==r["jev"]["status"]=="ok"]
        agreements=sum(r["agreement"] for r in ok)
        lines += [f"## {slot}","",f"- Comparable decisions: {len(ok)}",f"- Agreement: {agreements}/{len(ok)} ({agreements/len(ok):.1%})" if ok else "- Agreement: n/a",
                  f"- Jev errors: {sum(r['jev_error_count'] for r in rs)}",f"- Jev fallbacks: {sum(r['jev_fallback_count'] for r in rs)}",""]
        for side in ("baseline","jev"):
            valid=[r[side] for r in rs if r[side]["status"]=="ok"]
            lines += [f"### {side.title()} telemetry","",f"- Calls: {len(valid)}",f"- Input tokens: {sum(x['input_tokens'] for x in valid)}",
                      f"- Output tokens: {sum(x['output_tokens'] for x in valid)}",f"- Total latency: {sum((x['latency_s'] or 0) for x in valid):.3f}s",
                      f"- Estimated/priced cost: ${sum((x['cost_usd'] or 0) for x in valid):.6f}",
                      f"- Estimated calls (observed dashboard rate): {sum(x.get('pricing_status')=='estimated_observed_dashboard' for x in valid)}",
                      f"- Unpriced calls: {sum(x['cost_usd'] is None for x in valid)}",""]
        disagreements=[r for r in rs if not r["agreement"]]
        lines += ["### Disagreements for human review",""]
        if not disagreements: lines += ["None.",""]
        for r in disagreements:
            evidence=r.get("input",{}).get("state")
            lines += [f"- `{r['input_ref']}` at {r['timestamp']}",*( [f"  - Input: {json.dumps(evidence)}"] if evidence else []),
                      f"  - Baseline: `{json.dumps(r['baseline']['verdict'],sort_keys=True)}` ({r['baseline']['confidence']})",
                      f"  - Jev: `{json.dumps(r['jev']['verdict'],sort_keys=True)}` ({r['jev']['confidence']})",
                      f"  - Jev status: {r['jev']['status']}" + (f" - {r['jev']['error']['type']}: {r['jev']['error']['message']}" if r['jev']['status']=='error' else "")]
        lines.append("")
    return "\n".join(lines)
