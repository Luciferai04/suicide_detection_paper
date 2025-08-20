from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class AuditLogger:
    """Minimal append-only JSONL audit logger for ethics/PII handling events.

    Usage:
        log = AuditLogger(Path("logs/privacy_audit.log"))
        log.log_event("anonymization_start", {"file": "data.csv"})
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: str, payload: Dict[str, Any] | None = None) -> None:
        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "payload": payload or {},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
