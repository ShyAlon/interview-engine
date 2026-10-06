from langchain_core.messages import HumanMessage
from interview_engine.completeness import inventory

def test_missing_data_classes_are_named_open_not_defaulted():
 r=inventory('s',{'open_questions':['Which curriculum?']},[HumanMessage(content='A music app on iPad')])
 assert {x['data_class'] for x in r['items']} >= {'flows','styling','error_states','empty_states'}
 assert all(x['status']=='open' and x['owner']=='founder' for x in r['items'])
 assert {x['advisory_class'] for x in r['advisories']}=={'rights','real_device'}

def test_recorded_data_suppresses_absence_and_conops_supplies_flows():
 r=inventory('s',{},[HumanMessage(content='Blue colors, retry errors, empty state and screen reader support')],{'operational_scenarios':[{'name':'Begin','operational_flow':['start','finish'],'exceptions':['offline']}]})
 assert r['items']==[]

def test_shallow_conops_scenario_does_not_hide_flow_or_exception_findings():
 r=inventory('s',{},[HumanMessage(content='Blue colors, a flow, retry errors, empty state and screen reader support')],{'operational_scenarios':[{'name':'Log habit','operational_flow':['Log habit'],'exceptions':[]}]})
 ids={i['item_id'] for i in r['items']}
 assert ids=={'FD-missing-flows','FD-missing-scenario-exceptions'}
