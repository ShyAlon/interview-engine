"""Verbose, secret-safe focused diagnostics for categorization providers."""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from .categorization import (ChoiceQuestion,ClassificationRequest,FunctionProvider,
    ProviderChain,TypeSafeProvider)
from .model_config import _env


def emit(event, **fields): print(json.dumps({"event":event,**fields},sort_keys=True),flush=True)

def scope_request():
    return ClassificationRequest("scope","A web marketplace where surf schools offer lessons and customers book them.",{
      "accepted":ChoiceQuestion("Is this interactive web/mobile software?",{"accepted":"interactive web/mobile software","rejected":"not an interactive web/mobile application"}),
      "archetype":ChoiceQuestion("Choose the best product archetype",{"saas_crud":"SaaS workflow/CRUD","marketplace":"buyers and sellers transact","content_site":"content publishing","dashboard":"analytics/reporting","ecommerce":"single-merchant shop","other_web":"other interactive web/mobile app","not_web_app":"outside interactive web/mobile software"})})

def contracts():
    audits=[]
    def fail(_): raise TimeoutError("injected provider timeout")
    fallback=FunctionProvider("rules","diagnostic-rules",lambda _:{"accepted":__import__('interview_engine.categorization',fromlist=['ClassificationAnswer']).ClassificationAnswer("accepted",1.0),"archetype":__import__('interview_engine.categorization',fromlist=['ClassificationAnswer']).ClassificationAnswer("marketplace",1.0)})
    chain=ProviderChain("scope",[FunctionProvider("typesafe","injected-jev",fail),fallback],audits.append)
    request=scope_request();emit("request",slot=request.slot,state=request.state,questions=list(request.questions),provider_chain=[p.name for p in chain.providers])
    result=chain.classify(request);emit("response",slot=request.slot,provider=result.provider,model=result.model,answers={k:asdict(v) for k,v in result.answers.items()},fallback_path=list(audits[-1].fallback_path))
    if result.provider!="rules" or tuple(audits[-1].fallback_path)!=("typesafe","rules"):raise RuntimeError("fallback contract deviated")

def live():
    api_key=_env("TYPESAFE_API_KEY")
    if not api_key:
        emit("skip",reason="TYPESAFE_API_KEY is absent")
        return
    request=scope_request();provider=TypeSafeProvider(api_key=api_key);emit("request",slot=request.slot,state=request.state,questions={k:asdict(v) for k,v in request.questions.items()},provider_chain=[provider.name],model=provider.model)
    try: result=provider.classify(request)
    except Exception as exc:
        emit("failure",slot=request.slot,provider_chain=[provider.name],request={"state":request.state,"questions":list(request.questions)},response={"type":type(exc).__name__,"message":str(exc)})
        raise
    emit("response",slot=request.slot,provider=result.provider,model=result.model,answers={k:asdict(v) for k,v in result.answers.items()},fallback_path=[provider.name])
    if result.answers["accepted"].value!="accepted" or result.answers["archetype"].value!="marketplace":raise RuntimeError("live Jev control case returned unexpected verdict")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--mode",choices=("contracts","live"),required=True);args=parser.parse_args()
    (contracts if args.mode=="contracts" else live)()
if __name__=="__main__":main()
