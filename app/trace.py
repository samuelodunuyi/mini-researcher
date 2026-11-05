"""JSONL execution trace: one JSON object per line, enough to reconstruct a run without the agents' chats.

Every record has the same top-level fields. Secrets are scrubbed before anything is written.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_PATTERN = re.compile(r"(sk-ant-[A-Za-z0-9_\-]{8,}|tvly-[A-Za-z0-9_\-]{8,})")
REDACTED = "[REDACTED]"


class Trace:
    def __init__(
        self,
        path: Path | None,
        run_id: str = "",
        secrets: tuple[str, ...] = (),
        echo: Callable[[dict], None] | None = None,
    ):
        self.path = path
        self.run_id = run_id
        self._secrets = [s for s in secrets if s and len(s) >= 8]
        self._echo = echo
        self._handoffs = 0
        self.records: list[dict] = []  # kept in memory for the run summary

    def emit(
        self,
        event: str,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        handoff_id: str | None = None,
        status: str = "ok",
        duration_ms: float | None = None,
        error: BaseException | str | None = None,
        **metadata,
    ) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "agent": agent,
            "session_id": session_id,
            "event": event,
            "task_id": task_id,
            "handoff_id": handoff_id,
            "status": status,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
            "metadata": metadata,
            "error": describe_error(error),
        }
        record = self.redact(record)
        self.records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self._echo:
            self._echo(record)
        return record

    def new_handoff(self, kind: str) -> str:
        self._handoffs += 1
        return f"H{self._handoffs}-{kind}"

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, REDACTED)
            return SECRET_PATTERN.sub(REDACTED, value)
        if isinstance(value, dict):
            return {k: self.redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(v) for v in value]
        return value


def describe_error(error: BaseException | str | None) -> dict | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": "Error", "message": str(error)}


class Timer:
    """with Timer() as t: ...; t.ms"""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000


def null_trace() -> Trace:
    return Trace(None)
