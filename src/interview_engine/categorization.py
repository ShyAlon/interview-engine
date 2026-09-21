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
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float | None = None
    cost_usd: float | None = None


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
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float | None = None
    cost_usd: float | None = None
    pricing_status: str = "unpriced"


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
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_s=result.latency_s,
                    cost_usd=result.cost_usd,
                    pricing_status="priced" if result.cost_usd is not None else "unpriced",
                ))
                return result
            except Exception as exc:
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"all providers failed for {self.slot}: {'; '.join(errors)}")


class Categorizer:
    def __init__(self, chains: Mapping[str, ProviderChain], active_slots: Sequence[str] | None = None):
        self._chains = dict(chains); self._active_slots = frozenset(active_slots or chains)
    def active(self, slot: str) -> bool: return slot in self._active_slots
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
    @staticmethod
    def available() -> bool:
        import importlib.util
        return importlib.util.find_spec("typesafe_sdk") is not None
    def __init__(self, client=None, *, api_key: str | None = None,
                 model: str = "jev-latest", usage_log=None):
        if client is None:
            if not self.available():
                raise RuntimeError("typesafe-sdk is not installed in this Python runtime")
            if not api_key:
                raise RuntimeError("TYPESAFE_API_KEY is not configured")
            from typesafe_sdk import TypeSafeClient
            # Never depend on whether the SDK happens to load dotenv. Runtime
            # configuration has one resolver and the adapter passes it through.
            client = TypeSafeClient(api_key=api_key)
        self.client, self.model, self.usage_log = client, model, usage_log

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
        import time
        started = time.monotonic()
        response = self.client.system_one(state=request.state, questions=questions, model=self.model)
        latency_s = time.monotonic() - started
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
        usage = getattr(response, "usage", None)
        def usage_value(name):
            if usage is None: return 0
            return int((usage.get(name, 0) if isinstance(usage, dict) else getattr(usage, name, 0)) or 0)
        # TypeSafe's public response reports token counts but does not publish a
        # price in the SDK contract. Never invent $0: audit tokens/call/latency
        # and mark cost unpriced until an operator price table is configured.
        returned_model = getattr(response, "model", self.model)
        input_tokens, output_tokens = usage_value("input_tokens"), usage_value("output_tokens")
        if self.usage_log is not None:
            self.usage_log.record_unpriced("categorization_" + request.slot, self.name, returned_model,
                                           {"input_tokens": input_tokens, "output_tokens": output_tokens},
                                           duration_s=latency_s)
        return ProviderResult(answers, self.name, returned_model, input_tokens=input_tokens,
                              output_tokens=output_tokens, latency_s=latency_s, cost_usd=None)


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
    chains = {}; active = []
    for slot, available in adapters.items():
        providers = []
        for name in configured_provider_names(slot):
            if name == "jev":
                if not _env("TYPESAFE_API_KEY"):
                    continue  # disabled/unavailable: deterministic chain fallback
                # Registry assembly owns provider construction. Never create
                # a network client here: conformance tests and business call
                # sites stay hermetic regardless of ambient dotenv state.
                provider = available.get("jev")
                if provider is None:
                    continue
            else:
                provider = available.get(name)
            if provider is not None: providers.append(provider)
        chains[slot] = ProviderChain(slot, providers, audit)
        explicit = _env(f"INTERVIEW_ENGINE_CATEGORIZER_{slot.upper()}") or _env("INTERVIEW_ENGINE_CATEGORIZERS")
        if slot == "scope" or _env("TYPESAFE_API_KEY") or explicit:
            active.append(slot)
    return Categorizer(chains, active)


COVERAGE_AREAS = ("goal", "users", "flows", "data", "integrations", "nfrs", "failures", "constraints", "risks", "done", "glossary")
INTENT_LABELS = {
    "answer": "ordinary product answer", "delegate": "founder delegates the current detail",
    "dismiss": "founder closes the current topic/area", "skip": "founder explicitly skips the current area",
    "enough": "founder explicitly ends the whole interview",
}

def coverage_request(question: str, answer: str, target: str | None = None) -> ClassificationRequest:
    questions: dict[str, Question] = {
        "relevance": ChoiceQuestion("Did the answer address the question?", {
            "addressed": "directly and substantively answers it", "partial": "answers only part or remains vague",
            "dodged": "does not answer, changes subject, or refuses without delegating"}),
    }
    for area in COVERAGE_AREAS:
        questions[f"covers_{area}"] = NoulQuestion(f"The answer supplies concrete product facts for coverage area: {area}")
    return ClassificationRequest("coverage", f"Question: {question}\nAnswer: {answer}", questions, {"target": target or ""})

def intent_request(question: str, answer: str) -> ClassificationRequest:
    return ClassificationRequest("intent", f"Question: {question}\nFounder answer: {answer}", {
        "intent": ChoiceQuestion("Classify the founder's control intent", INTENT_LABELS),
    })

TERMINAL_INTENTS = frozenset({"dismiss", "skip", "enough"})

def apply_coverage_answers(coverage: Mapping[str, str], answers: Mapping[str, ClassificationAnswer]) -> dict[str, str]:
    """Code-owned accumulation. Classifiers judge one answer; they never own state."""
    from .coverage import COVERED, DELEGATED, SKIPPED, THIN
    out = dict(coverage)
    relevance = str(answers["relevance"].value)
    for area in COVERAGE_AREAS:
        signal = answers.get(f"covers_{area}")
        if signal is None or float(signal.value) < .5 or out.get(area) in (COVERED, SKIPPED, DELEGATED):
            continue
        out[area] = COVERED if relevance == "addressed" else THIN
    return out


def classified_intent(answers: Mapping[str, ClassificationAnswer], terminal_threshold: float = .9) -> str:
    answer = answers["intent"]
    intent = str(answer.value)
    if intent in TERMINAL_INTENTS and (answer.confidence is None or answer.confidence < terminal_threshold):
        return "answer"
    return intent
