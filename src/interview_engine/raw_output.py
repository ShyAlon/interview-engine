"""Durable raw model-output evidence for structured stages."""
from __future__ import annotations
import re
from pathlib import Path

class RawOutputLog:
    def __init__(self, root: str | Path | None):
        self.root = Path(root).resolve() if root else None
        self.counts: dict[str,int] = {}

    def write(self, stage: str, label: str, raw: str) -> Path | None:
        from .document_diagnostics import configured
        if configured():return self._write(stage,label,raw)
        from .stop_fence import guarded_publish
        return guarded_publish(lambda:self._write(stage,label,raw))

    def _write(self, stage: str, label: str, raw: str) -> Path | None:
        if self.root is None:
            return None
        slug = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-") or "output"
        key=f"{stage}-{slug}"; self.counts[key]=self.counts.get(key,0)+1
        path=self.root/f"{key}-attempt-{self.counts[key]}.raw.txt"
        from .document_diagnostics import configured,write
        if configured():write(path,raw)
        else:
            self.root.mkdir(parents=True,exist_ok=True)
            path.write_text(raw)
        return path
