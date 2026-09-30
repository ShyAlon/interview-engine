from langchain_core.messages import HumanMessage
from interview_engine.completeness import inventory

def test_missing_data_classes_are_named_open_not_defaulted():
 r=inventory('s',{'open_questions':['Which curriculum?']},[HumanMessage(content='A music app on iPad')])
 assert {x['data_class'] for x in r['items']} >= {'flows','styling','error_states','empty_states'}
 assert all(x['status']=='open' and x['owner']=='founder' for x in r['items'])
 assert {x['advisory_class'] for x in r['advisories']}=={'rights','real_device'}

def test_recorded_data_suppresses_absence_and_conops_supplies_flows():
 r=inventory('s',{},[HumanMessage(content='Blue colors, retry errors, empty state and screen reader support')],{'operational_scenarios':[{'flow':['start','finish']}]})
 assert r['items']==[]
