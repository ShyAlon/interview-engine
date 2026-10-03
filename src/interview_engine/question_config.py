"""Question-quality configuration (JEV-D12, E2).

One global config, named constants with code defaults, a validated schema,
read per session so an edit applies without a restart. Nothing here changes
interview behavior by itself: later changes read these values.

Unknown keys and out-of-range values are rejected, never silently dropped.
The store is a JSON file; with no path set the defaults apply and edits are
kept in memory for the process.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

CONFIG_VERSION = 1

DEFAULTS: dict = {
    "retrieval": {
        "max_words": 90,        # cap on the session-facts block
        "max_chars": 600,       # secondary safeguard, lowest-ranked spans dropped first
        "top_k": 4,             # spans per target area
    },
    "verification": {
        "retry_enabled": False,       # one retry with the failed check named; OFF until calibrated
        "grounding_overlap_min": 1,   # shared content words between question and a span
    },
    "question_repeat": {
        # A candidate whose normalized wording overlaps a PRIOR QUESTION at or above
        # this percentage is a repeat. Founder answers are never compared.
        "similarity_threshold_pct": 45,
    },
    "integrations": {
        "none_stated_enabled": True,
        "external_system_verbs": ["connect", "integrate", "sync", "import", "export"],
        "self_contained_cues": ["offline", "no external", "self contained", "self-contained"],
    },
    # Words whose presence ties a founder sentence to an area (prefix match).
    "area_cues": {
        "goal": ["problem", "want", "need", "help", "goal", "so that", "because", "hurt", "struggl"],
        "users": ["user", "customer", "parent", "kid", "child", "student", "teacher", "admin", "owner", "manager", "staff", "team"],
        "flows": ["when", "then", "after", "before", "step", "click", "tap", "sign", "book", "pay", "play", "submit", "upload"],
        "data": ["data", "from", "own", "store", "bundled", "enter", "import", "song", "tune", "music", "image", "video", "record", "content", "file"],
        "integrations": ["api", "connect", "integrat", "sync", "import", "export", "login", "payment", "calendar", "email",
                         "exchange", "existing system", "existing service", "existing tool", "external", "third-party",
                         "third party", "standalone", "stand-alone", "self-contained", "self contained"],
        "nfrs": ["fast", "slow", "wait", "quick", "load", "secure", "private", "reliab", "scale", "offline", "available"],
        "failures": ["fail", "break", "crash", "error", "lose", "lost", "wrong", "never", "down", "miss"],
        "constraints": ["budget", "deadline", "weeks", "months", "team", "cost", "cheap", "language", "cloud", "platform"],
        "risks": ["risk", "assum", "depend", "unsure", "if wrong", "worry", "concern"],
        "done": ["done", "ready", "launch", "first version", "success", "demo", "mvp"],
        "glossary": ["means", "call", "term", "we say", "known as", "stands for"],
    },
}

# (min, max) for ints; bool and list[str] are checked by default type.
_INT_RANGES = {
    ("retrieval", "max_words"): (20, 300),
    ("retrieval", "max_chars"): (100, 2000),
    ("retrieval", "top_k"): (1, 10),
    ("verification", "grounding_overlap_min"): (1, 5),
    ("question_repeat", "similarity_threshold_pct"): (10, 100),
}


class ConfigError(ValueError):
    """The submitted configuration is invalid; the message names the key."""


def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)


def validate(cfg: dict) -> dict:
    """Return a complete, normalized config. Missing keys take defaults."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be an object")
    out = defaults()
    for section, body in cfg.items():
        if section not in DEFAULTS:
            raise ConfigError(f"unknown section: {section}")
        if not isinstance(body, dict):
            raise ConfigError(f"{section} must be an object")
        for key, val in body.items():
            if section == "area_cues":
                from .coverage_taxonomy import AREA_KEYS
                if key not in AREA_KEYS:
                    raise ConfigError(f"area_cues.{key} is not a coverage area")
                if not _str_list(val):
                    raise ConfigError(f"area_cues.{key} must be a list of non-empty strings")
                out[section][key] = [v.strip() for v in val]
                continue
            if key not in DEFAULTS[section]:
                raise ConfigError(f"unknown key: {section}.{key}")
            default = DEFAULTS[section][key]
            if isinstance(default, bool):
                if not isinstance(val, bool):
                    raise ConfigError(f"{section}.{key} must be true or false")
            elif isinstance(default, int):
                lo, hi = _INT_RANGES[(section, key)]
                if isinstance(val, bool) or not isinstance(val, int) or not lo <= val <= hi:
                    raise ConfigError(f"{section}.{key} must be an integer from {lo} to {hi}")
            elif not _str_list(val):
                raise ConfigError(f"{section}.{key} must be a list of non-empty strings")
            out[section][key] = [v.strip() for v in val] if isinstance(default, list) else val
    return out


def _str_list(val) -> bool:
    return isinstance(val, list) and all(isinstance(v, str) and v.strip() for v in val)


class ConfigStore:
    """Global store. get() re-reads the file, so each session sees the latest edit."""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._mem: dict = defaults()

    def get(self) -> dict:
        with self._lock:
            if self._path is None:
                return copy.deepcopy(self._mem)
            try:
                return validate(json.loads(self._path.read_text()))
            except (FileNotFoundError, ValueError):
                # Missing or corrupt file: fall back to code defaults, never fail a session.
                return defaults()

    def put(self, cfg: dict) -> dict:
        clean = validate(cfg)
        with self._lock:
            if self._path is None:
                self._mem = clean
            else:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".tmp")
                tmp.write_text(json.dumps(clean, indent=2, sort_keys=True))
                os.replace(tmp, self._path)
        return copy.deepcopy(clean)


_store = ConfigStore(os.environ.get("INTERVIEW_ENGINE_QUESTION_QUALITY_CONFIG") or None)


def store() -> ConfigStore:
    return _store


def set_store(new: ConfigStore) -> None:
    global _store
    _store = new


def current() -> dict:
    """The config a session reads at start."""
    return _store.get()
