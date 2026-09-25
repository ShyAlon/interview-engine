import csv,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('filter_qa',ROOT/'scripts/filter-real-qa-summary.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_filter_preserves_export_and_case_ids(tmp_path):
    source=tmp_path/'export.csv';out=tmp_path/'filtered.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','slot','human_verdict','review_notes','input_state']);w.writeheader()
        w.writerows([{'case_id':'a','slot':'intent','human_verdict':'','review_notes':'EXCLUDE_SUMMARY','input_state':'blob'},
                     {'case_id':'c','slot':'intent','human_verdict':'','review_notes':'CLARIFY','input_state':'What do you mean?'},
                     {'case_id':'b','slot':'intent','human_verdict':'{"intent":["answer"]}','review_notes':'ordinary','input_state':'short'}])
    before=source.read_bytes()
    assert mod.main([str(source),'-o',str(out),'--dry-run'])==0 and not out.exists()
    assert mod.main([str(source),'-o',str(out)])==0 and source.read_bytes()==before
    assert [r['case_id'] for r in csv.DictReader(out.open())]==['b']
    audit=tmp_path/'filtered-clarifications.csv'
    assert [r['case_id'] for r in csv.DictReader(audit.open())]==['c']


def test_future_extractor_summary_marker_not_generic_length():
    spec=importlib.util.spec_from_file_location('extract_qa',ROOT/'scripts/extract-real-qa-testbed.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    assert e.SUMMARY_QUESTION.search('Please summarize all requirements and confirm this recap.')
    assert not e.SUMMARY_QUESTION.search('Why is your phone booking process slow? '*50)
