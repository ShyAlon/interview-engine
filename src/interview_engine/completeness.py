"""Conservative recorded-data inventory, not a claim about the founder's world."""
import re
from .conops_stage import scenario_depth_problems
from vmodel_contracts.founder_register import FounderRegister

_CLASSES={
 'flows':(r'\b(flow|steps|journey|scenario|workflow|operational)\b','Which user flows and steps are required?','user flows'),
 'styling':(r'\b(style|styling|visual|design|colors?|colours?|fonts?|theme|brand|screenshot)\b','What styling or visual references should guide the app?','visual identity'),
 'error_states':(r'\b(error|failure|retry|invalid|exception)\b','What should users see and do when something fails?','failure handling'),
 'empty_states':(r'\b(empty|no data|first use|first run|onboarding)\b','What should users see before data exists?','empty and first-use states'),
 'accessibility':(r'\b(accessib|screen reader|keyboard navigation|contrast)', 'What accessibility needs must the app meet?','accessibility'),
}

# Earliest pipeline stage that cannot be built correctly without the missing data.
# Free-text open questions have no known owning stage, so they stay empty and only block the final handoff.
_BLOCKED_STAGES={'flows':['srs'],'error_states':['srs'],'styling':['items'],'empty_states':['items'],
 'accessibility':['items'],'content':['items'],'deployment':['architecture']}

def inventory(session_id,intent,messages,conops=None):
 turns=[(i,str(getattr(m,'content',''))) for i,m in enumerate(messages) if getattr(m,'type',None)=='human']
 corpus=' '.join(text for _,text in turns)
 items=[]
 for cls,(pattern,question,scope) in _CLASSES.items():
  # ConOps scenarios are concrete flow data; do not report missing flows once
  # a reviewed flow exists. Keyword evidence suppresses uncertain absence.
  # Flows: a ConOps scenario with 2+ real steps is concrete flow data. A shallow
  # scenario must not hide the gap, so the check ignores keyword evidence then.
  if cls=='flows' and conops and conops.get('operational_scenarios'):
   if any('operational_flow' not in scenario_depth_problems(x) for x in conops['operational_scenarios']):continue
   items.append({'item_id':'FD-missing-flows','data_class':'flows','question':_CLASSES['flows'][1],'owner':'founder','status':'open','scope':'user flows',
    'source':'Every Concept of Operations scenario has fewer than 2 real steps; completeness finding, not proof no flow exists.','blocks':['user flows']});continue
  if re.search(pattern,corpus,re.I):continue
  items.append({'item_id':f'FD-missing-{cls}','data_class':cls,'question':question,
    'owner':'founder','status':'open','scope':scope,'blocked_stages':_BLOCKED_STAGES.get(cls,[]),'source':'Recorded founder interview has no explicit data for this class; completeness finding, not proof no external material exists.',
    'blocks':[scope]})
 shallow=[x.get('name','scenario') for x in (conops or {}).get('operational_scenarios') or [] if 'exceptions' in scenario_depth_problems(x)]
 if shallow:
  items.append({'item_id':'FD-missing-scenario-exceptions','data_class':'error_states','question':'What should happen when these scenarios fail or deviate: '+'; '.join(shallow)+'?','owner':'founder','status':'open','scope':'scenario exceptions','blocked_stages':_BLOCKED_STAGES['error_states'],
   'source':'Concept of Operations scenarios without any recorded exception; completeness finding, not proof none exist.','blocks':['scenario exceptions']})
 for i,question in enumerate(intent.get('open_questions') or [],1):
  items.append({'item_id':f'FD-open-{i}','data_class':'other','question':question,'owner':'founder','scope':'Unresolved interview question','source':'intent_model.json#open_questions','blocks':['work depending on this answer']})
 advisories=[]
 if re.search(r'\b(song|music|duolingo|copy|clone|licensed|copyright|branding)\b',corpus,re.I):
  advisories.append({'advisory_id':'CA-rights','advisory_class':'rights','statement':'Confirm rights to third-party content, arrangements, branding and visual references before using them.','owner':'founder','source':'Interview domain/content references','validation_expectation':'Record permission, license, public-domain status or original replacement. This is not a legal conclusion.'})
 if re.search(r'\b(iphone|ipad|mobile|touch|audio|microphone|pwa)\b',corpus,re.I):
  advisories.append({'advisory_id':'CA-device','advisory_class':'real_device','statement':'Validate the specified touch/audio/install/storage behavior on the intended physical devices.','owner':'founder','source':'Interview device/runtime references','validation_expectation':'Record device/browser and observed results. Browser automation does not prove physical-device behavior.'})
 return FounderRegister.model_validate({'session_id':session_id,'items':items,'advisories':advisories}).model_dump(mode='json')
