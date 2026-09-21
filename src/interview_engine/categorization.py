"""Provider-neutral categorization ports, routing, fallback, and decision audit.

Business stages call ``Categorizer.classify`` with a slot and typed question
specification. Provider choice lives in one registry assembled from config; no
stage branches on Jev, LLM, or rules.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .model_config import _env


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: Mapping[str, str]
    multiple: bool = False


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: Sequence[str]


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


@dataclass(frozen=True)
class ClassificationRequest:
    slot: str
    state: str
    questions: Mapping[str, Question]
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassificationAnswer:
    value: str | float | list[str]
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResult:
    answers: Mapping[str, ClassificationAnswer]
    provider: str
    model: str


class ClassificationProvider(Protocol):
    name: str
    def classify(self, request: ClassificationRequest) -> ProviderResult: ...


@dataclass(frozen=True)
class DecisionAudit:
    slot: str
    verdict: Mapping[str, str | float | list[str]]
    confidence: Mapping[str, float | None]
    decider: str
    provider: str
    model: str
    fallback_path: tuple[str, ...]
    timestamp: str


AuditSink = Callable[[DecisionAudit], None]


def jsonl_audit_sink(path: str | Path) -> AuditSink:
    path = Path(path)
    def write(record: DecisionAudit) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
    return write


class ProviderChain:
    """One slot's ordered providers. Failure advances; success is audited once."""
    def __init__(self, slot: str, providers: Sequence[ClassificationProvider], audit: AuditSink):
        if not providers:
            raise ValueError(f"categorization slot {slot!r} has no providers")
        self.slot, self.providers, self.audit = slot, tuple(providers), audit

    def classify(self, request: ClassificationRequest) -> ProviderResult:
        if request.slot != self.slot:
            raise ValueError(f"request slot {request.slot!r} does not match chain {self.slot!r}")
        attempted: list[str] = []
        errors: list[str] = []
        for provider in self.providers:
            attempted.append(provider.name)
            try:
                result = provider.classify(request)
                self.audit(DecisionAudit(
                    slot=self.slot,
                    verdict={key: answer.value for key, answer in result.answers.items()},
                    confidence={key: answer.confidence for key, answer in result.answers.items()},
                    decider="jev" if result.provider == "typesafe" else result.provider,
                    provider=result.provider,
                    model=result.model,
                    fallback_path=tuple(attempted),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))
                return result
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"all providers failed for {self.slot}: {'; '.join(errors)}")


class Categorizer:
    def __init__(self, chains: Mapping[str, ProviderChain]): self._chains = dict(chains)
    def classify(self, request: ClassificationRequest) -> ProviderResult:
        try: chain = self._chains[request.slot]
        except KeyError as exc: raise KeyError(f"unknown categorization slot {request.slot!r}") from exc
        return chain.classify(request)


class FunctionProvider:
    """Adapter for the existing LLM/rules implementation of one slot."""
    def __init__(self, name: str, model: str, fn: Callable[[ClassificationRequest], Mapping[str, ClassificationAnswer]]):
        self.name, self.model, self._fn = name, model, fn
    def classify(self, request: ClassificationRequest) -> ProviderResult:
        return ProviderResult(self._fn(request), self.name, self.model)


class TypeSafeProvider:
    name = "typesafe"
    def __init__(self, client=None, *, model: str = "jev-latest"):
        if client is None:
            from typesafe_sdk import TypeSafeClient
            client = TypeSafeClient()
        self.client, self.model = client, model

    def classify(self, request: ClassificationRequest) -> ProviderResult:
        from typesafe_sdk import Choice, Noul, Score
        questions = {}
        for key, question in request.questions.items():
            if isinstance(question, ChoiceQuestion):
                if question.multiple:
                    # System One Choice is singular. Multi-label uses one Noul
                    # question per code-owned label, assembled by the caller.
                    raise ValueError("multi-label Choice must be expanded to Noul questions")
                questions[key] = Choice(instructions=question.instructions, criteria=dict(question.criteria))
            elif isinstance(question, ScoreQuestion):
                questions[key] = Score(instructions=question.instructions, criteria=list(question.criteria))
            else:
                questions[key] = Noul(instructions=question.instructions)
        response = self.client.system_one(state=request.state, questions=questions, model=self.model)
        answers = {}
        for key, answer in response.answers.items():
            if hasattr(answer, "choice"):
                value = answer.choice
            elif hasattr(answer, "score"):
                value = float(answer.score)
            else:
                value = float(answer.noul)
            answers[key] = ClassificationAnswer(value=value,
                confidence=getattr(answer, "confidence", None),
                probabilities=dict(getattr(answer, "probabilities", {}) or {}))
        return ProviderResult(answers, self.name, getattr(response, "model", self.model))


def configured_provider_names(slot: str) -> tuple[str, ...]:
    """Single on/off registry: per-slot override, then global, then safe default."""
    raw = (_env(f"INTERVIEW_ENGINE_CATEGORIZER_{slot.upper()}")
           or _env("INTERVIEW_ENGINE_CATEGORIZERS") or "jev,llm,rules")
    names = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    allowed = {"jev", "llm", "rules"}
    unknown = [x for x in names if x not in allowed]
    if unknown: raise ValueError(f"unknown categorization providers for {slot}: {', '.join(unknown)}")
    if not names: raise ValueError(f"categorization provider chain for {slot} is empty")
    return names


def build_categorizer(adapters: Mapping[str, Mapping[str, ClassificationProvider]], audit: AuditSink) -> Categorizer:
    chains = {}
    for slot, available in adapters.items():
        providers = []
        for name in configured_provider_names(slot):
            if name == "jev":
                if not _env("TYPESAFE_API_KEY"):
                    continue  # disabled/unavailable: deterministic chain fallback
                provider = available.get("jev") or TypeSafeProvider()
            else:
                provider = available.get(name)
            if provider is not None: providers.append(provider)
        chains[slot] = ProviderChain(slot, providers, audit)
    return Categorizer(chains)
