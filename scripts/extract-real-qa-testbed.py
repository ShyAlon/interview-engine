#!/usr/bin/env python3
"""Extract real interview Q/A into replay-ready intent adjudication CSV.

Defaults to out/ and runs/ under repo root. Reads interview/checkpoints.sqlite,
transcript JSON lists, shadow/replay JSONL with a complete input package, and
adjudication CSVs. Does not infer Q/A from decision-only logs. Synthetic bench
artifacts (bench*.csv, synthetic-benchmark) are excluded.

The blind CSV contains neither candidate hints nor system verdicts. A separate
manifest gives source and deterministic control-phrase flags for sampling;
these are retrieval hints, NOT ground truth. Have a human label the blind CSV
before scoring. --dry-run reads/counts but writes nothing.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,re,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from interview_engine.categorization import intent_request  # noqa: E402
from interview_engine.categorization_shadow import input_package  # noqa: E402
from interview_engine.intent_taxonomy import INTENT_ORDER  # noqa: E402

COLUMNS=['case_id','input_ref','slot','session_id','input_state','input_context','rubric','rubric_version',
         'system_a_verdict','system_a_confidence','system_b_verdict','system_b_confidence','sample_reason',
         'human_verdict','winner','reviewer_id','review_notes']
# Conservative, high-precision retrieval cues. They do not assign intent truth.
CUES={
 'skip': r'\b(?:skip (?:it|this|that)|next question|move (?:on|along)|rather not (?:talk|discuss|get into)|don.t want to (?:talk|discuss|get into)|not relevant|leave (?:this|that|it) (?:out|aside))\b',
 'delegate': r'\b(?:you decide|your call|you choose|pick (?:something|whatever|for us)|choose for (?:me|us)|whatever you.d recommend|go with what (?:you think|makes sense))\b',
 'defer': r'\b(?:come back to (?:you|this|that)|get back to (?:you|this)|check with (?:my|our)|find out (?:and|then)|answer (?:this|that) later)\b',
 'enough': r'\b(?:no more questions|end the interview|done with the interview|stop (?:the )?questions|write (?:the spec|it) up|wrap (?:it|this|the interview) up)\b',
}
# Explicit summary/review prompts are not single-question intent cases. A
# length alone is not evidence: long ordinary questions remain eligible and
# can be removed after review with filter-real-qa-summary.py.
SUMMARY_QUESTION=re.compile(r"\b(?:summari[sz]e|summary of|review (?:everything|all (?:the|your) (?:answers|requirements))|recap of|here(?:'| i)s (?:what|a summary) (?:we|i) (?:heard|covered))\b",re.I)
QA=re.compile(r'(?:^|\n)Question:\s*(.*?)\n(?:Founder answer|Answer):\s*(.*)\Z',re.S|re.I)


def text(v):
    if isinstance(v,str):return v.strip()
    if isinstance(v,list):return ''.join(str(z.get('text','')) if isinstance(z,dict) else str(z) for z in v).strip()
    return ''


def from_turns(turns):
    question=''
    for t in turns:
        role=(t.get('role') or t.get('type') or '').lower()
        body=text(t.get('content'))
        if role in ('assistant','ai'): question=body
        elif role in ('user','human','founder') and question and body:
            yield question,body


def checkpoint_turns(path):
    spec=importlib.util.spec_from_file_location('enrich_shadow',ROOT/'scripts/enrich-categorization-shadow.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    turns,_=module._turns_from_checkpoint(path)
    return turns


def parse_state(v):
    if not isinstance(v,str):return None
    m=QA.search(v.strip())
    if not m:return None
    question,answer=(part.strip() for part in m.groups())
    return (question,answer) if question and answer else None


def candidates(path):
    suffix=path.suffix.lower();name=path.name.lower()
    if name=='checkpoints.sqlite':
        for i,(q,a) in enumerate(from_turns(checkpoint_turns(path)),1):yield q,a,i
    elif suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:
            for i,row in enumerate(csv.DictReader(f),2):
                if (row.get('sample_reason') or '').startswith('synthetic-') or (row.get('session_id') or '').startswith('bench-'):continue
                pair=parse_state(row.get('input_state',''))
                if pair and row.get('slot')=='intent':yield *pair,i
    elif suffix=='.jsonl':
        with path.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                try:record=json.loads(line)
                except json.JSONDecodeError:continue
                if record.get('slot')!='intent':continue
                if str(record.get('session_id') or '').startswith('bench-') or str((record.get('replay') or {}).get('source_file') or '').endswith('bench.csv'):continue
                inp=record.get('input') or {}
                pair=parse_state(inp.get('state') if isinstance(inp,dict) else None)
                if pair:yield *pair,i
    elif suffix=='.json' and ('transcript' in name or 'turns' in name):
        turns=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(turns,list):
            for i,(q,a) in enumerate(from_turns(turns),1):yield q,a,i


def sources(roots):
    seen=set()
    for root in roots:
        if not root.exists():continue
        files=[root] if root.is_file() else root.rglob('*')
        for p in files:
            if not p.is_file() or p.is_symlink():continue
            if p.name.lower().startswith('bench') or 'benchmarks' in p.parts:continue
            if p.name=='decisions.jsonl' or (p.suffix.lower()=='.jsonl' and not ('shadow' in p.name or 'replay' in p.name)):continue
            if p.suffix.lower()=='.json' and not ('transcript' in p.name.lower() or 'turns' in p.name.lower()):continue
            if p.name!='checkpoints.sqlite' and p.suffix.lower() not in ('.csv','.jsonl','.json'):continue
            resolved=p.resolve()
            if resolved not in seen:seen.add(resolved);yield p


def extract(roots):
    entries={};stats=Counter();files=0
    for path in sources(roots):
        files+=1
        try:items=list(candidates(path))
        except (OSError,ValueError,ImportError) as e:
            stats['unreadable']+=1;print(f'warning: skipped {path}: {type(e).__name__}: {e}',file=sys.stderr);continue
        for q,a,n in items:
            stats['observed_pairs']+=1
            if SUMMARY_QUESTION.search(q):
                stats['excluded_summary']+=1;continue
            req=intent_request(q,a)
            pkg=input_package(req)
            # Identity is the exact request state; duplicates across old exports,
            # checkpoints and replays must not inflate a test's denominator.
            ref='sha256:'+hashlib.sha256(pkg['state'].encode()).hexdigest()
            if ref in entries:
                entries[ref]['sources'].append(f'{path}:{n}');stats['duplicate_pairs']+=1;continue
            flags=[label for label,pattern in CUES.items() if re.search(pattern,a,re.I)]
            entries[ref]={'package':pkg,'source':f'{path}:{n}','sources':[f'{path}:{n}'],'flags':flags,'question':q,'answer':a}
            stats['candidate' if flags else 'ordinary']+=1
    return entries,stats,files


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots',nargs='*',type=Path,help='artifact files/directories; defaults to repo out/ and runs/')
    p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    roots=a.roots or [ROOT/'out',ROOT/'runs']
    entries,stats,files=extract(roots)
    print(f'1. Scanned {files} files under {", ".join(map(str,roots))}; observed {stats["observed_pairs"]} Q/A pairs; {stats["duplicate_pairs"]} duplicate occurrences.')
    print(f'2. Unique: {len(entries)}; control-phrase candidates: {stats["candidate"]}; ordinary controls: {stats["ordinary"]}; summary question occurrences excluded: {stats["excluded_summary"]}; unreadable files: {stats["unreadable"]}.')
    print('3. Flags are retrieval hints in a SEPARATE manifest, never labels; the adjudication CSV has no flags or model verdicts.')
    if not entries:print('4. No Q/A pairs found. Supply an artifact root with checkpoints, transcript JSON, enriched shadows, or real adjudication CSV.');return 2
    if a.dry_run:print('4. Dry run: no files written.');return 0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=a.output.with_name(a.output.stem+'-manifest.csv')
    with a.output.open('w',newline='',encoding='utf-8') as f,manifest.open('w',newline='',encoding='utf-8') as m:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader()
        mw=csv.DictWriter(m,fieldnames=['case_id','candidate_flags','source','source_count','question','answer']);mw.writeheader()
        for ref,e in sorted(entries.items()):
            pkg=e['package'];cid=hashlib.sha256((ref+'|intent').encode()).hexdigest()[:16]
            w.writerow(dict(case_id=cid,input_ref=ref,slot='intent',session_id='',input_state=pkg['state'],
                input_context=json.dumps(pkg['context'],sort_keys=True),rubric=json.dumps(pkg['questions'],sort_keys=True),
                rubric_version=pkg['rubric_version'],system_a_verdict='',system_a_confidence='',system_b_verdict='',
                system_b_confidence='',sample_reason='artifact-extract',human_verdict='',winner='',reviewer_id='',review_notes=''))
            mw.writerow(dict(case_id=cid,candidate_flags=';'.join(e['flags']),source=e['source'],source_count=len(e['sources']),question=e['question'],answer=e['answer']))
    print(f'4. Wrote blind replay/label CSV: {a.output}; private sampling manifest: {manifest}.')
    print('5. Human labels required: in the labeling tool choose Neither, then a manual intent label; no A/B verdicts exist. Do not score candidate flags as truth.')
    return 0

if __name__=='__main__':sys.exit(main())
