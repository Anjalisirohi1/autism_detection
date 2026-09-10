"""Shared logging setup: console + a timestamped JSONL run log."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class _JsonlHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.format(record)
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:  # pragma: no cover - trivial
        try:
            self._fh.close()
        finally:
            super().close()


def setup_logging(name: str, repo_root: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging for a script run and return a named logger."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = repo_root / "logs" / f"{name}-{stamp}.jsonl"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    root.addHandler(_JsonlHandler(log_path))

    logger = logging.getLogger(name)
    logger.info("run log: %s", log_path)
    return logger
