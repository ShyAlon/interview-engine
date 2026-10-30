#!/usr/bin/env python3
"""Backfill v1 shadow rows from an explicitly exported transcript JSON."""
import argparse,hashlib,json
from pathlib import Path


def _package(request,version='recovered-v1'):
    from interview_engine.categorization_shadow import input_package
    package=input_package(request);package['rubric_version']=version
    return package


def _candidates(turns,submission=None):
    from interview_engine.categorization import coverage_request,intent_request
    candidates={}
    for i,t in enumerate(turns):
        if t.get('role') not in {'user','human','founder'}:continue
        answer=t.get('content','');prior=next((x.get('content','') for x in reversed(turns[:i]) if x.get('role') in {'assistant','ai'}),'')
        for request in (coverage_request(prior,answer),intent_request(prior,answer)):
            candidates['sha256:'+hashlib.sha256(request.state.encode()).hexdigest()]=_package(request)
    first=next((x.get('content','') for x in turns if x.get('role') in {'user','human','founder'}),'')
    if first:
        try:
            from vmodel_contracts import IntakeSubmission
            from interview_engine.scope_gate import categorization_request
            payload=dict(submission or {});payload.setdefault('free_text',first);payload.setdefault('founder_id','recovered');payload.setdefault('submission_id','recovered')
            request=categorization_request(IntakeSubmission.model_validate(payload))
            candidates['sha256:'+hashlib.sha256(request.state.encode()).hexdigest()]=_package(request)
        except Exception:
            # Oldest scope records predate the canonical "Product idea" state.
            candidates['sha256:'+hashlib.sha256(first.encode()).hexdigest()]={'state':first,'context':{},'questions':{},'rubric_version':'recovered-legacy-scope','rubric_ref':None}
    return candidates


def enrich(jsonl,output,turns,submission=None):
    candidates=_candidates(turns,submission);found=0;out=[]
    for line in Path(jsonl).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line);package=candidates.get(row.get('input_ref'))
        if package and not row.get('input'):
            row['input']=package;row['record_schema_version']=2;found+=1
        out.append(json.dumps(row,separators=(',',':')))
    Path(output).write_text('\n'.join(out)+'\n')
    print(f'{output}: enriched {found}/{len(out)} records')
    return found,len(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('jsonl');source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--transcript',help='JSON list of {role,content} turns')
    p.add_argument('-o','--output',required=True);a=p.parse_args()
    turns=json.loads(Path(a.transcript).read_text());submission={}
    enrich(a.jsonl,a.output,turns,submission)

if __name__=='__main__':main()
