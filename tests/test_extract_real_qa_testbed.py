"""The wide testbed must not turn synthetic or hint text into a ground truth label."""
import csv, importlib.util, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('extract_real',ROOT/'scripts/extract-real-qa-testbed.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_extract_mixed_real_artifacts_and_blind_output(tmp_path):
    root=tmp_path/'out';root.mkdir()
    turns=[{'role':'human','content':'idea'}, {'role':'ai','content':'Who decides?'},
           {'role':'human','content':'Your call.'}, {'role':'ai','content':'What happens?'},
           {'role':'human','content':'It stores dates.'}]
    (root/'transcript.json').write_text(json.dumps(turns))
    (root/'categorization-shadow.jsonl').write_text('\n'.join(json.dumps({'slot':'intent','session_id':'real',
        'input':{'state':f'Question: {q}\nFounder answer: {a}'}}) for q,a in mod.from_turns(turns))+'\n')
    (root/'bench1.csv').write_text('garbage')
    (root/'decisions.jsonl').write_text('garbage')
    (root/'replay-synthetic.jsonl').write_text(json.dumps({'slot':'intent','session_id':'bench-123',
       'input':{'state':'Question: fake\nFounder answer: Skip it'}})+'\n')
    entries,stats,_=mod.extract([root]);assert len(entries)==2 and stats['duplicate_pairs']==2
    assert {tuple(e['flags']) for e in entries.values()}=={('delegate',),()}
    out=tmp_path/'real.csv'; assert mod.main([str(root),'-o',str(out),'--dry-run'])==0 and not out.exists()
    assert mod.main([str(root),'-o',str(out)])==0
    rows=list(csv.DictReader(out.open()));assert len(rows)==2
    assert all(not r['human_verdict'] and not r['winner'] and not r['system_a_verdict'] for r in rows)
    man=list(csv.DictReader((tmp_path/'real-manifest.csv').open()));assert len(man)==2
    assert sum(bool(r['candidate_flags']) for r in man)==1
    assert not any('candidate' in r['sample_reason'] for r in rows)


def test_full_founder_answer_not_question_as_hint():
    assert mod.parse_state('Question: would you skip?\nFounder answer: I have names and dates.')==('would you skip?','I have names and dates.')
    assert mod.parse_state('Question: x\nFounder answer:') is None
