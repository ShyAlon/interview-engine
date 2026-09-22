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

def test_checkpoint_reader_uses_latest_langgraph_state(tmp_path):
    import importlib.util,sqlite3
    from langchain_core.messages import AIMessage,HumanMessage
    from langgraph.checkpoint.sqlite import SqliteSaver
    root=Path(__file__).parents[1];script=root/'scripts/enrich-categorization-shadow.py'
    spec=importlib.util.spec_from_file_location('enrich',script);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    db=tmp_path/'checkpoints.sqlite';saver=SqliteSaver(sqlite3.connect(db,check_same_thread=False))
    cfg={'configurable':{'thread_id':'s','checkpoint_ns':'','checkpoint_id':'c1'}}
    checkpoint={'v':1,'id':'c1','ts':'2026-09-22T00:00:00+00:00','channel_values':{
      'messages':[HumanMessage(content='idea'),AIMessage(content='question'),HumanMessage(content='answer')],
      'submission':{'free_text':'idea','founder_id':'f','submission_id':'s'}},'channel_versions':{},'versions_seen':{},'updated_channels':[]}
    saver.put(cfg,checkpoint,{},{});saver.conn.close()
    turns,submission=mod._turns_from_checkpoint(db)
    assert [x['content'] for x in turns]==['idea','question','answer']
    assert submission['free_text']=='idea'
