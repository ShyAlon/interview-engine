import csv,json,subprocess,sys
from pathlib import Path

def test_label_export_and_validation(tmp_path):
    root=Path(__file__).parents[1];source=tmp_path/'shadow.jsonl';out=tmp_path/'labels.csv'
    source.write_text(json.dumps({'slot':'coverage','input_ref':'sha256:a','session_id':'s','schema_id':'interview.coverage','schema_version':'1.0.0','baseline':{'verdict':{'relevance':'addressed'}},'jev':{'verdict':{'relevance':'partial'}}})+'\n')
    subprocess.run([sys.executable,root/'scripts/export-categorization-labels.py',source,'-o',out],check=True)
    rows=list(csv.DictReader(out.open()));assert rows[0]['input_ref']=='sha256:a' and rows[0]['human_relevance']==''
    rows[0].update(human_relevance='addressed',human_covered_facets='goal,users',reviewer_id='reviewer-1')
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([sys.executable,root/'scripts/validate-categorization-labels.py',out],check=True)
