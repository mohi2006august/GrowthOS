"""Structured logging: one JSON object per line, with explicit fields."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class StructuredLogger:
    """Thin wrapper so call sites read `log.info("event", key=value)`."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, event: str, **fields: Any) -> None:
        self._logger.debug(event, extra={"fields": fields})

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(event, extra={"fields": fields})

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(event, extra={"fields": fields})

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(event, extra={"fields": fields})

    def exception(self, event: str, **fields: Any) -> None:
        self._logger.error(event, extra={"fields": fields}, exc_info=True)


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    # stderr keeps stdout clean for MCP stdio servers, which speak JSON-RPC on stdout.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("growthos")
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name))
