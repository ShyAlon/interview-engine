#!/usr/bin/env python3
"""One-command, fail-closed pre-promotion regression matrix (run on Shy's Mac).

Usage: .venv/bin/python scripts/run-production-regression.py \
  --candidate-commit COMMIT --intent-variant intent-v1.4 \
  --tree-variant relevance-trees-v5 --real-csv runs/JEV-L2-filtered.csv \
  --review-csv ~/Downloads/JEV-L3-reviewed.csv --audit-csv runs/JEV-L3-audit.csv \
  --cat-csv runs/JEV-L2-cat-probe.csv --swaps-csv runs/JEV-L3-swaps.csv \
  --out runs/regression-COMMIT

The candidate commit must be checked out with a clean tree. Modes: --dry-run
builds all requests but makes no Jev calls; --check only validates existing
replays in OUT without running Jev. A new full run uses fresh OUT. It covers
four pinned intent pools, pool-v4 component-tree coverage, and JEV-L2 real
with the reviewed L3 addressing overlay only for research-only v1.5-v1.7.
v1.4 uses the original five-label L2 truth (L3 non-answer is out of taxonomy).
No current numeric approval bars
exist for four intent pools or real data. A complete run is evidence for a
human quality ruling, NOT automatic permission to promote or deploy. Six
pre-registered component bars must pass; absence/error/mismatch fails closed.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from interview_engine.categorization_rubrics import INTENT_VARIANTS,COMPONENT_VARIANTS

def load_script(name):
    s=importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/'scripts'/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(*argv):subprocess.run([sys.executable,str(ROOT/'scripts'/argv[0])+'.py',*map(str,argv[1:])],check=True)
def fail(text):raise ValueError(text)

def records(path,cases,variant,slot):
    expected={r['case_id']:r for r in cases if r['slot']==slot}
    if len(expected)!=sum(r['slot']==slot for r in cases):fail('duplicate input case IDs: '+str(path))
    seen={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);cid=r['replay']['case_id']
            if cid in seen:fail(f'duplicate replay ID {cid}: {path}')
            if r['replay']['variant_id']!=variant or r['slot']!=slot or r['jev']['status']!='ok':fail(f'variant/slot/status mismatch {cid}: {path}')
            if r['input_ref']!=expected.get(cid,{}).get('input_ref'):fail(f'input ref mismatch {cid}: {path}')
            if r['input'].get('rubric_version')!=(INTENT_VARIANTS[variant] if slot=='intent' else COMPONENT_VARIANTS[variant]).rubric_version:
                fail(f'rubric version mismatch {cid}: {path}')
            if r['replay'].get('repeat')!=1:fail(f'unexpected repeat {cid}: {path}')
            seen[cid]=r
    if set(seen)!=set(expected):fail(f'ID mismatch for {path}: {len(set(expected)-set(seen))} missing, {len(set(seen)-set(expected))} extra')
    return seen

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ('candidate-commit','intent-variant','tree-variant','real-csv','review-csv','audit-csv','cat-csv','swaps-csv','out'):
        p.add_argument('--'+key,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--check',action='store_true')
    p.add_argument('--workers',type=int,default=4);a=p.parse_args(argv)
    if a.intent_variant not in INTENT_VARIANTS or a.tree_variant not in COMPONENT_VARIANTS:fail('unknown candidate variant')
    if a.workers<1:fail('workers must be >=1')
    if a.intent_variant not in ('intent-v1.4','intent-v1.5','intent-v1.6','intent-v1.7'):
        fail('this gate requires a candidate with supported truth/scoring contract (v1.4-v1.7)')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','--verify',a.candidate_commit+'^{commit}'],cwd=ROOT,text=True).strip()
    if head!=commit:fail(f'candidate commit {commit} is not checked out (HEAD {head})')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):fail('working tree must be clean')
    paths={name:Path(getattr(a,name.replace('-','_'))).expanduser().resolve() for name in ('real-csv','review-csv','audit-csv','cat-csv','swaps-csv')}
    for name,path in paths.items():
        if not path.is_file():fail(f'missing {name}: {path}')
    out=Path(a.out).expanduser().resolve()
    if not a.check and out.exists() and any(out.iterdir()):fail('output directory must be fresh/empty: '+str(out))
    if a.check and not out.is_dir():fail('missing existing regression output: '+str(out))
    if a.check and not (out/'manifest.json').is_file():fail('missing completed regression manifest: '+str(out))
    builder=load_script('build-intent-benchmark');replayer=load_script('replay-jev-rubric-variants')
    pool_meta={}
    for n in range(1,5):
        src=ROOT/f'benchmarks/intent-pool-v{n}.json';pool=json.loads(src.read_text())
        rows=builder.build_rows(pool);csvpath=out/f'pool-v{n}.csv'
        if a.check:
            if not csvpath.is_file() or csvpath.read_text()!=builder.to_csv(rows):fail('generated benchmark mismatch: '+str(csvpath))
        elif not a.dry_run:
            out.mkdir(parents=True,exist_ok=True);csvpath.write_text(builder.to_csv(rows))
        pool_meta[n]=(src,rows,csvpath)
    with paths['real-csv'].open(encoding='utf-8-sig',newline='') as f:real=list(csv.DictReader(f))
    if len(real)!=185 or len({r['case_id'] for r in real})!=185 or any(r['slot']!='intent' for r in real):fail('real source must have 185 distinct intent cases')
    legacy_real={r['case_id']:json.loads(r['human_verdict'])['intent'] for r in real}
    review=load_script('score-jev-l3-v16');rtruth,_,_,boundary,_=review.truth_sets(paths['real-csv'],paths['cat-csv'],paths['review-csv'],paths['audit-csv'])
    if len(rtruth)!=183 or len(boundary)!=2:fail('real truth overlay must have 183 scored and two boundary cases')
    modern=a.intent_variant in ('intent-v1.5','intent-v1.6','intent-v1.7')
    if not modern and any(not set(labels)<= {'answer','delegate','defer','skip','enough'} for labels in legacy_real.values()):
        fail('legacy real truth has out-of-taxonomy labels')
    manifest={'candidate_commit':commit,'intent_variant':a.intent_variant,'tree_variant':a.tree_variant,
              'runner_sha256':sha(ROOT/'scripts/run-production-regression.py'),
              'sources':{str(x):sha(x) for x in [*paths.values(),*(v[0] for v in pool_meta.values())]},
              'runs':{},'production_approval':'NOT GRANTED; quality bars and operator review required'}
    for name,source,rows,slot,variant in [
        *((f'pool-v{n}-intent',v[2],v[1],'intent',a.intent_variant) for n,v in pool_meta.items()),
        ('pool-v4-trees',pool_meta[4][2],pool_meta[4][1],'coverage',a.tree_variant),
        ('jev-l2-real',paths['real-csv'],real,'intent',a.intent_variant)]:
        dest=out/name
        if a.dry_run:
            cases=[{'slot':r['slot'],'state':r['input_state'],'context':json.loads(r['input_context'] or '{}'),
                    'questions':json.loads(r['rubric'] or '{}')} for r in rows]
            variant_obj=INTENT_VARIANTS[variant] if slot=='intent' else COMPONENT_VARIANTS[variant]
            for c in cases:
                if c['slot']==slot:replayer.build_request(c,variant_obj)
            print(f'DRY {name}: {sum(r["slot"]==slot for r in rows)} requests built')
            continue
        if not a.check:
            run('replay-jev-rubric-variants',source,'--intent-variants',variant if slot=='intent' else 'none',
                '--relevance-variants',variant if slot=='coverage' else 'none','--workers',a.workers,'-o',dest)
        replay=dest/f'replay-{variant}.jsonl';got=records(replay,rows,variant,slot)
        manifest['runs'][name]={'n':len(got),'replay_sha256':sha(replay)}
        if slot=='intent':
            truth=({r['case_id']:json.loads(r['human_verdict'])['intent'] for r in rows} if name!='jev-l2-real'
                   else rtruth if modern else legacy_real)
            scored={c:r for c,r in got.items() if c in truth}
            if name=='jev-l2-real' and len(scored)!=(183 if modern else 185):fail('unexpected scored real count')
            strict=load_script('_jev_strict_scoring')
            labels=('answer','proceed','skip','enough','non-answer') if modern else ('answer','delegate','defer','skip','enough')
            from interview_engine.intent_taxonomy import compatible
            preds={c:r['jev']['verdict']['intent'] for c,r in scored.items()}
            if any(v not in labels for v in preds.values()):fail('unexpected intent label in '+name)
            # Keep raw strict metrics AND legacy answer/delegate compatibility visible.
            metrics=strict.label_stats(truth,preds,labels)
            manifest['runs'][name]['strict']={k:metrics[k] for k in ('n','correct','accuracy','macro_f1','per_label')}
            manifest['runs'][name]['compatible_correct']=sum(compatible(preds[c],truth[c]) for c in truth) if not modern else None
            if name=='jev-l2-real':manifest['runs'][name]['truth_contract']='reviewed L3 overlay; two boundary rows excluded' if modern else 'original L2 five-label truth; L3 addressing overlay incompatible with v1.4 taxonomy and not applied'
        else:
            scorer=load_script('score-component-bench');cov={r['case_id']:r for r in rows if r['slot']=='coverage'}
            s,fy,fn,models,extra=scorer.score(cov,list(got.values()))
            pool=json.loads(pool_meta[4][0].read_text());from interview_engine.component_trees import derive_relevance
            concrete={r['id'] for r in pool['responses'] for area in r.get('areas',[]) if derive_relevance(area,set(r.get('components',{}).get(area,[])),r.get('skipped',[]),set(r.get('delegated',{}).get(area,[])),personas=pool.get('personas'))=='addressed'}
            wqok,wqn=scorer.wrong_question(cov,list(got.values()),concrete)
            vals={'dodged_recall':(s['dodged_ok'],s['dodged_n']),'wrong_question':(wqok,wqn),'false_yes':(s['comp_fp'],s['comp_neg']),
                  'component_recall':(s['comp_tp'],s['comp_pos']),'partials':(s['partial_ok'],s['partial_n']),'addressed':(s['addr_ok'],s['addr_n'])}
            bars={}
            for key,_,bar in scorer.GATES:
                k,n=vals[key];passed=n>0 and ((k/n<=bar) if key=='false_yes' else (k/n>=bar))
                bars[key]={'n':n,'numerator':k,'bar':bar,'pass':passed}
            manifest['runs'][name]['component_bars']=bars
    if a.dry_run:return 0
    if a.check:
        original=json.loads((out/'manifest.json').read_text())
        if original!=manifest:
            fail('existing manifest differs from recomputed evidence')
    else:(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,v in manifest['runs'].items():
        if 'strict' in v:print(f'{name}: {v["strict"]["correct"]}/{v["strict"]["n"]} strict'+(f'; compatible {v["compatible_correct"]}/{v["strict"]["n"]}' if v['compatible_correct'] is not None else ''))
        else:print(f'{name}: '+', '.join(f'{k}={"PASS" if x["pass"] else "FAIL"} ({x["numerator"]}/{x["n"]})' for k,x in v['component_bars'].items()))
    if not all(x['pass'] for x in manifest['runs']['pool-v4-trees']['component_bars'].values()):fail('component bars FAIL; report remains at '+str(out/'manifest.json'))
    print('Complete evidence at',out/'manifest.json','; production approval is NOT granted')
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,KeyError,FileNotFoundError) as e:print('REGRESSION BLOCKED:',e,file=sys.stderr);sys.exit(2)
