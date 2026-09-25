"""Cleanup invariants: strict reports and both browser export shapes stay stable."""
import importlib.util,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/name)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_shared_strict_stats_multi_label_and_score_shape():
    m=script('_jev_strict_scoring.py')
    r=m.label_stats({'a':['answer','skip'],'b':['proceed'],'c':['non-answer']},
                    {'a':'skip','b':'answer','c':'non-answer'},('answer','proceed','skip','enough','non-answer'))
    assert r['correct']==2 and r['accuracy']==2/3
    assert r['per_label']['answer']['support']==1 and r['per_label']['answer']['recall']==0
    assert r['per_label']['skip']['support']==1 and r['per_label']['skip']['recall']==1
    assert r['truth_distribution']=={'answer+skip':1,'proceed':1,'non-answer':1}
    assert r['mistakes']==[{'case_id':'b','truth':['proceed'],'prediction':'answer'}]

def test_shared_browser_csv_is_one_parser_and_preserves_storage_keys():
    label=script('build-labeling-tool.py');l3=script('build-jev-l3-review.py')
    row={c:'' for c in label.COLUMNS};row.update(case_id='id1',slot='intent',input_state='Question: Q?\nFounder answer: A')
    for manual in (False,True):
        html,key=label.build([row],title='Test',subtitle='Test',export_name='labels.csv',manual_intent=manual)
        assert html.count('function parseCSV(')==html.count('function q(')==1
        assert key.startswith('labeling-tool-v2:') and 'localStorage.getItem(STORE)' in html
    review=[{'case_id':'id2','question':'Q?','founder_answer':'A'}]
    html,key=l3.html(review)
    assert html.count('function parseCSV(')==html.count('function q(')==1
    assert key.startswith('jev-l3:') and "KEY='"+key+"'" in html
    # Run the exact embedded JS functions in Node, including Unicode, quotes,
    # commas and CRLF; export shape is not changed by sharing the parser.
    shared=(ROOT/'scripts/templates/label-csv.js').read_text()
    js=shared+'''\nconst cells=['case_id','question','founder_answer','verdict','review_notes'];
const values=['id2','Q, and "more"?','A\\ncat','answer',''];
const text=cells.map(q).join(',')+'\\r\\n'+values.map(q).join(',')+'\\r\\n';
console.log(JSON.stringify(parseCSV(text)));
'''
    run=subprocess.run(['node','-e',js],capture_output=True,text=True,check=True)
    assert json.loads(run.stdout)==[['case_id','question','founder_answer','verdict','review_notes'],['id2','Q, and "more"?','A\ncat','answer','']]

def test_reports_match_pre_cleanup_goldens(tmp_path):
    fixture=ROOT/'tests/fixtures/cleanup'
    source=tmp_path/'truth.csv';replay=tmp_path/'replay.jsonl'
    rows=[('708e80db09a22198',['defer'],'answer'),
          ('6fc6ef8c785d964a',['defer'],'defer'),
          ('a90ef9d67e072b60',['answer','skip'],'answer'),
          ('x',['delegate'],'answer')]
    with source.open('w',newline='') as f:
        import csv
        w=csv.DictWriter(f,fieldnames=['case_id','human_verdict']);w.writeheader()
        w.writerows({'case_id':cid,'human_verdict':json.dumps({'intent':label})} for cid,label,_ in rows)
    with replay.open('w') as f:
        for cid,_,pred in rows:
            f.write(json.dumps({'replay':{'case_id':cid,'variant_id':'intent-v1'},
                                'jev':{'status':'ok','verdict':{'intent':pred}}})+'\n')
    merged=script('score-jev-l2-merged.py')
    assert json.dumps(merged.analyze(source,[replay]),indent=2)+'\n'==(fixture/'merged-report.json').read_text()
    v15=script('score-jev-l2-v15.py')
    result=v15.score({'a':['answer','skip'],'b':['proceed'],'c':['non-answer']},
                     {'a':'skip','b':'answer','c':'non-answer'})
    assert json.dumps(result,indent=2)+'\n'==(fixture/'v15-strict.json').read_text()
