import json
from interview_engine.interviewee import load_persona,question_id,scripted_answer

def test_scripted_answer_is_stable_and_keyed(tmp_path):
 p=tmp_path/"p.json";p.write_text(json.dumps({"idea":"x","answers":[{"keywords":["users","who"],"answer":"Families"}],"fallback":"enough"}))
 persona=load_persona(p); q="Who are the users?"
 assert scripted_answer(persona,q)==(question_id(q),"Families")
 assert scripted_answer(persona,"Unspecified detail")[1]=="enough"
