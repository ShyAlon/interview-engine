"""Non-authoritative Jev shadow comparison and regenerable reporting."""
from __future__ import annotations
import hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path


def _side(result=None, *, elapsed=None, usage=None, error=None):
    if error is not None:
        return {"status":"error","error":{"type":type(error).__name__,"message":str(error)},
                "provider":getattr(error,"provider",None),"model":None,"verdict":{},"confidence":{},
                "input_tokens":0,"output_tokens":0,"latency_s":elapsed,"cost_usd":None}
    tokens_in=getattr(result,"input_tokens",0);tokens_out=getattr(result,"output_tokens",0)
    cost=getattr(result,"cost_usd",None)
    if usage is not None:
        tokens_in=usage.prompt_tokens;tokens_out=usage.completion_tokens;cost=usage.cost_usd
    return {"status":"ok","provider":result.provider,"model":result.model,
            "verdict":{k:v.value for k,v in result.answers.items()},
            "confidence":{k:v.confidence for k,v in result.answers.items()},
            "input_tokens":tokens_in,"output_tokens":tokens_out,
            "latency_s":getattr(result,"latency_s",None) or elapsed,"cost_usd":cost}


def _agreement(a,b):
    return a.get("status")==b.get("status")=="ok" and a.get("verdict")==b.get("verdict")


class ShadowCategorizer:
    """Return baseline decisions; run Jev concurrently and only observe it."""
    def __init__(self, baseline, jev, jsonl_path, usage_log=None):
        self.baseline,self.jev,self.path,self.usage_log=baseline,jev,Path(jsonl_path),usage_log
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
        record={"schema_version":1,"timestamp":datetime.now(timezone.utc).isoformat(),"slot":request.slot,
                "input_ref":"sha256:"+hashlib.sha256(request.state.encode()).hexdigest(),
                "baseline":baseline_side,"jev":jev_side,"agreement":_agreement(baseline_side,jev_side),
                "jev_fallback_count":0,"jev_error_count":int(shadow_err is not None)}
        with self.path.open("a") as f:
            f.write(json.dumps(record,separators=(",",":"),default=str)+"\n");f.flush();os.fsync(f.fileno())
        if base_err: raise base_err
        return base


def render_report(paths):
    records=[]
    for path in paths:
        with Path(path).open() as f:
            records += [json.loads(line) for line in f if line.strip()]
    generated=datetime.now(timezone.utc).isoformat()
    lines=["# Jev shadow report","",f"Generated: {generated}",f"Decisions: {len(records)}",""]
    for slot in sorted({r["slot"] for r in records}):
        rs=[r for r in records if r["slot"]==slot];ok=[r for r in rs if r["baseline"]["status"]==r["jev"]["status"]=="ok"]
        agreements=sum(r["agreement"] for r in ok)
        lines += [f"## {slot}","",f"- Comparable decisions: {len(ok)}",f"- Agreement: {agreements}/{len(ok)} ({agreements/len(ok):.1%})" if ok else "- Agreement: n/a",
                  f"- Jev errors: {sum(r['jev_error_count'] for r in rs)}",f"- Jev fallbacks: {sum(r['jev_fallback_count'] for r in rs)}",""]
        for side in ("baseline","jev"):
            valid=[r[side] for r in rs if r[side]["status"]=="ok"]
            lines += [f"### {side.title()} telemetry","",
                      f"- Calls: {len(valid)}",f"- Input tokens: {sum(x['input_tokens'] for x in valid)}",
                      f"- Output tokens: {sum(x['output_tokens'] for x in valid)}",
                      f"- Total latency: {sum((x['latency_s'] or 0) for x in valid):.3f}s",
                      f"- Priced cost: ${sum((x['cost_usd'] or 0) for x in valid):.6f}",
                      f"- Unpriced calls: {sum(x['cost_usd'] is None for x in valid)}",""]
        disagreements=[r for r in rs if not r["agreement"]]
        lines += ["### Disagreements for human review",""]
        if not disagreements: lines += ["None.",""]
        for r in disagreements:
            lines += [f"- `{r['input_ref']}` at {r['timestamp']}",f"  - Baseline: `{json.dumps(r['baseline']['verdict'],sort_keys=True)}` ({r['baseline']['confidence']})",
                      f"  - Jev: `{json.dumps(r['jev']['verdict'],sort_keys=True)}` ({r['jev']['confidence']})",
                      f"  - Jev status: {r['jev']['status']}" + (f" - {r['jev']['error']['type']}: {r['jev']['error']['message']}" if r['jev']['status']=='error' else "")]
        lines.append("")
    return "\n".join(lines)
