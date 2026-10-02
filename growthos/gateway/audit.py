"""Append-only audit trail of proposals, approvals and spend (PRD section 8).

Each entry stores the hash of the previous one, so editing or deleting a past entry breaks
the chain and `verify()` reports where.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel

from growthos.clock import utc_now_iso
from growthos.db import Database
from growthos.log import get_logger

Outcome = Literal["ok", "denied", "error", "rolled_back"]
_GENESIS = "0" * 64
log = get_logger(__name__)


class AuditEntry(BaseModel):
    seq: int
    ts: str
    actor: str
    action: str
    tool: str | None
    approval_id: str | None
    experiment_id: str | None
    amount_usd: float | None
    outcome: Outcome
    detail: dict[str, Any]
    prev_hash: str
    hash: str


class AuditLog:
    def __init__(self, db: Database) -> None:
        self._db = db

    def record(
        self,
        *,
        actor: str,
        action: str,
        outcome: Outcome = "ok",
        tool: str | None = None,
        approval_id: str | None = None,
        experiment_id: str | None = None,
        amount_usd: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> AuditEntry:
        # Hash exactly what reading the row back will produce (REAL amounts, JSON-decoded
        # detail); otherwise an int amount or a tuple in detail would fail verification.
        body = {
            "ts": utc_now_iso(),
            "actor": actor,
            "action": action,
            "tool": tool,
            "approval_id": approval_id,
            "experiment_id": experiment_id,
            "amount_usd": None if amount_usd is None else float(amount_usd),
            "outcome": outcome,
            "detail": json.loads(_canonical(detail or {})),
        }
        with self._db.tx() as conn:
            last = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = last["hash"] if last else _GENESIS
            entry_hash = _digest(prev_hash, body)
            cursor = conn.execute(
                "INSERT INTO audit_log (ts, actor, action, tool, approval_id, experiment_id, amount_usd, "
                "outcome, detail, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    body["ts"],
                    actor,
                    action,
                    tool,
                    approval_id,
                    experiment_id,
                    amount_usd,
                    outcome,
                    _canonical(body["detail"]),
                    prev_hash,
                    entry_hash,
                ),
            )
            seq = cursor.lastrowid
        log.info(
            "audit",
            seq=seq,
            actor=actor,
            action=action,
            tool=tool,
            approval_id=approval_id,
            experiment_id=experiment_id,
            amount_usd=amount_usd,
            outcome=outcome,
        )
        return AuditEntry(seq=seq, prev_hash=prev_hash, hash=entry_hash, **body)

    def entries(self, limit: int = 200, experiment_id: str | None = None) -> list[AuditEntry]:
        """Most recent entries first."""
        query = "SELECT * FROM audit_log"
        params: tuple[Any, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            params = (experiment_id,)
        with self._db.tx() as conn:
            rows = conn.execute(query + " ORDER BY seq DESC LIMIT ?", (*params, limit)).fetchall()
        return [_entry(row) for row in rows]

    def verify(self) -> int | None:
        """Return the sequence number of the first tampered entry, or None if the chain is intact."""
        with self._db.tx() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY seq").fetchall()
        prev_hash = _GENESIS
        for row in rows:
            entry = _entry(row)
            body = entry.model_dump(exclude={"seq", "prev_hash", "hash"})
            if entry.prev_hash != prev_hash or entry.hash != _digest(prev_hash, body):
                return entry.seq
            prev_hash = entry.hash
        return None


def _entry(row: Any) -> AuditEntry:
    return AuditEntry(
        seq=row["seq"],
        ts=row["ts"],
        actor=row["actor"],
        action=row["action"],
        tool=row["tool"],
        approval_id=row["approval_id"],
        experiment_id=row["experiment_id"],
        amount_usd=row["amount_usd"],
        outcome=row["outcome"],
        detail=json.loads(row["detail"]),
        prev_hash=row["prev_hash"],
        hash=row["hash"],
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(prev_hash: str, body: dict[str, Any]) -> str:
    return hashlib.sha256(f"{prev_hash}{_canonical(body)}".encode()).hexdigest()
