import json
from interview_engine.interviewee import load_persona,question_id,scripted_answer

def test_scripted_answer_is_stable_and_keyed(tmp_path):
 p=tmp_path/"p.json";p.write_text(json.dumps({"idea":"x","answers":[{"keywords":["users","who"],"answer":"Families"}],"fallback":"enough"}))
 persona=load_persona(p); q="Who are the users?"
 assert scripted_answer(persona,q)==(question_id(q),"Families")
 assert scripted_answer(persona,"Unspecified detail")[1]=="enough"

def test_missing_persona_is_human_readable(tmp_path):
 from interview_engine.errors import UserFacingError
 import pytest
 with pytest.raises(UserFacingError) as ei: load_persona(tmp_path/"missing.json")
 rendered=ei.value.render()
 assert rendered.startswith("Error: hands-free persona file not found:")
 assert "Likely cause:" in rendered and "Next:" in rendered

def test_malformed_persona_is_human_readable(tmp_path):
 from interview_engine.errors import UserFacingError
 import pytest
 p=tmp_path/"bad.json";p.write_text("{")
 with pytest.raises(UserFacingError) as ei: load_persona(p)
 assert all(x in ei.value.render() for x in ("Error:","Likely cause:","Next:"))
