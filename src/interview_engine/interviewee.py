"""Hands-free interviewees seeded by one ground-truth persona file."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

def load_persona(path: str|Path) -> dict:
 from .errors import UserFacingError
 source = Path(path)
 try:
  d=json.loads(source.read_text())
 except FileNotFoundError as e:
  raise UserFacingError(
   f"hands-free persona file not found: {source}.",
   "the path was resolved from a different working directory, the file was moved, or it was not pulled.",
   "pass an existing JSON persona with --interviewee; the hands-free driver resolves it to an absolute path.") from e
 except (OSError, json.JSONDecodeError) as e:
  raise UserFacingError(
   f"hands-free persona could not be read: {source}.",
   f"{type(e).__name__}: {str(e)[:240]}",
   "fix the file permissions/JSON and retry the same command.") from e
 if not isinstance(d, dict) or not str(d.get("idea") or "").strip() or not isinstance(d.get("answers"),list):
  raise UserFacingError(
   f"hands-free persona is invalid: {source}.",
   "the JSON must be an object with a non-empty idea and answers array.",
   "add {idea: string, answers: [{keywords: [string], answer: string}]} and retry.")
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
         "using the persona as ground truth. For a decision gate, select the rendered option matching persona.decisions and include a rationale in the requested format. Never invent facts; say 'you decide' for unspecified technical choices. "
         "Return only the answer.\nPERSONA:\n"+json.dumps(persona)+"\nRECENT:\n"+json.dumps(history[-6:])+"\nQUESTION:\n"+question)
 resp=model_config.invoke_checked(model,[HumanMessage(content=prompt)],"probing",item="hands-free interviewee")
 return question_id(question),str(resp.content).strip()

def decision_answer(persona: dict, question: str) -> tuple[str, str] | None:
 """Resolve a named persona decision against the options code rendered."""
 if "concept decision" not in question.casefold(): return None
 decision=(persona.get("decisions") or {}).get("concept") or {}
 prefer=[str(x).casefold() for x in decision.get("prefer_keywords") or [] if str(x).strip()]
 rationale=str(decision.get("rationale") or "This best matches the stated first-release constraints.").strip()
 options=re.findall(r"(?m)^(\d+)\.\s+([^:\n]+):",question)
 matches=[(number,name) for number,name in options if any(k in name.casefold() for k in prefer)]
 if len(matches)!=1: return None
 number,name=matches[0]
 return question_id(question),f"{number} - {rationale}"
