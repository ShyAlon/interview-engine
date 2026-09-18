"""Hands-free interviewees seeded by one ground-truth persona file."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

def load_persona(path: str|Path) -> dict:
 d=json.loads(Path(path).read_text())
 if not str(d.get("idea") or "").strip() or not isinstance(d.get("answers"),list):
  raise ValueError("persona requires non-empty idea and answers[]")
 return d

def question_id(question: str) -> str:
 normalized=re.sub(r"[^a-z0-9]+"," ",question.casefold()).strip()
 return "q-"+hashlib.sha256(normalized.encode()).hexdigest()[:12]

def scripted_answer(persona: dict, question: str) -> tuple[str,str]:
 text=question.casefold(); ranked=[]
 for index,row in enumerate(persona["answers"]):
  keys=[str(k).casefold() for k in row.get("keywords") or []]
  score=sum(1 for k in keys if k in text)
  if score: ranked.append((score,-index,row))
 row=max(ranked,key=lambda x:(x[0],x[1]))[2] if ranked else None
 answer=str((row or {}).get("answer") or persona.get("fallback") or "enough").strip()
 return question_id(question),answer

def llm_answer(persona: dict, question: str, history: list[dict]) -> tuple[str,str]:
 from langchain_core.messages import HumanMessage
 from . import model_config
 model=model_config.build_chat_model("probing")
 prompt=("Act as the product founder in a test interview. Answer only the latest question, in first person, "
         "using the persona as ground truth. Never invent facts; say 'you decide' for unspecified technical choices. "
         "Return only the answer.\nPERSONA:\n"+json.dumps(persona)+"\nRECENT:\n"+json.dumps(history[-6:])+"\nQUESTION:\n"+question)
 resp=model_config.invoke_checked(model,[HumanMessage(content=prompt)],"probing",item="hands-free interviewee")
 return question_id(question),str(resp.content).strip()
