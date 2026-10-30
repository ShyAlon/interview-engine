import csv,json,subprocess,sys
from pathlib import Path

def test_blind_label_export_and_validation(tmp_path):
    root=Path(__file__).parents[1];source=tmp_path/'shadow.jsonl';out=tmp_path/'labels.csv'
    source.write_text(json.dumps({'slot':'coverage','input_ref':'sha256:a','session_id':'s','agreement':False,
      'input':{'state':'Question: q\nAnswer: a','context':{'target':'goal'},'questions':{'relevance':{'instructions':'judge'}},'rubric_version':'v1'},
      'baseline':{'verdict':{'relevance':'addressed'},'confidence':{'relevance':.8},'model':'base'},
      'jev':{'verdict':{'relevance':'partial'},'confidence':{'relevance':.7},'model':'jev'}})+'\n')
    subprocess.run([sys.executable,root/'scripts/export-categorization-labels.py',source,'-o',out],check=True)
    rows=list(csv.DictReader(out.open()));assert rows[0]['input_state'].endswith('Answer: a')
    assert rows[0]['system_a_verdict'] and rows[0]['system_b_verdict'] and 'baseline' not in rows[0] and 'jev' not in rows[0]
    key=json.loads(Path(str(out)+'.key.json').read_text());assert {key[0]['system_a'],key[0]['system_b']}=={'baseline','jev'}

def test_transcript_enrichment_keeps_turns_and_submission_explicit(tmp_path):
    import importlib.util
    root=Path(__file__).parents[1];script=root/'scripts/enrich-categorization-shadow.py'
    spec=importlib.util.spec_from_file_location('enrich',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    turns=[{'role':'human','content':'idea'},{'role':'ai','content':'question'},{'role':'human','content':'answer'}]
    candidates=mod._candidates(turns,{'free_text':'idea','founder_id':'f','submission_id':'s'})
    assert candidates and all(key.startswith('sha256:') for key in candidates)
    assert not hasattr(mod,'_turns_from_checkpoint')
