"""SQLite system of record.

One file holds experiments, approvals, the audit log, metric sources, embeddings and the
simulated ad platforms. See BRAIN.md (Decision Log) for why this is SQLite rather than
Postgres + pgvector + Redis for now.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- FR-01: metric sources (ads, CRM, billing, bank) ---------------------------
CREATE TABLE IF NOT EXISTS channel_spend (
    day       TEXT NOT NULL,
    channel   TEXT NOT NULL,
    spend_usd REAL NOT NULL,
    PRIMARY KEY (day, channel)
);
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    acquired_on TEXT NOT NULL,
    channel     TEXT NOT NULL,
    segment_id  TEXT,
    mrr_usd     REAL NOT NULL,
    churned_on  TEXT,
    campaign_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_customers_acquired ON customers (acquired_on, channel);
CREATE TABLE IF NOT EXISTS finance_monthly (
    month           TEXT PRIMARY KEY,
    revenue_usd     REAL NOT NULL,
    cogs_usd        REAL NOT NULL,
    sm_overhead_usd REAL NOT NULL,
    other_opex_usd  REAL NOT NULL,
    cash_end_usd    REAL NOT NULL
);

-- FR-02: experiment memory ---------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id         TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    channel    TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    body       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status);
CREATE TABLE IF NOT EXISTS memory_docs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    experiment_id TEXT,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    source        TEXT,
    created_at    TEXT NOT NULL,
    embedding     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_docs_experiment ON memory_docs (experiment_id);

-- FR-04: approvals, audit, guardrail policy ----------------------------------
CREATE TABLE IF NOT EXISTS approvals (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    tool          TEXT NOT NULL,
    args          TEXT NOT NULL,
    action_hash   TEXT NOT NULL,
    amount_usd    REAL NOT NULL,
    preview       TEXT NOT NULL,
    required_role TEXT NOT NULL,
    status        TEXT NOT NULL,
    experiment_id TEXT,
    requested_by  TEXT NOT NULL,
    requested_at  TEXT NOT NULL,
    decided_by    TEXT,
    decided_role  TEXT,
    decided_at    TEXT,
    reason        TEXT,
    executed_at   TEXT,
    result        TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals (status);
CREATE INDEX IF NOT EXISTS idx_approvals_hash ON approvals (action_hash);
CREATE TABLE IF NOT EXISTS audit_log (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    actor         TEXT NOT NULL,
    action        TEXT NOT NULL,
    tool          TEXT,
    approval_id   TEXT,
    experiment_id TEXT,
    amount_usd    REAL,
    outcome       TEXT NOT NULL,
    detail        TEXT NOT NULL,
    prev_hash     TEXT NOT NULL,
    hash          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS guardrail_policy (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    set_by  TEXT NOT NULL,
    body    TEXT NOT NULL
);

-- FR-05: simulated ad / email / landing platforms ---------------------------
CREATE TABLE IF NOT EXISTS sim_campaigns (
    id               TEXT PRIMARY KEY,
    platform         TEXT NOT NULL,
    name             TEXT NOT NULL,
    kind             TEXT NOT NULL,
    experiment_id    TEXT,
    segment_id       TEXT,
    status           TEXT NOT NULL,
    daily_budget_usd REAL NOT NULL,
    total_budget_usd REAL,
    spent_usd        REAL NOT NULL DEFAULT 0,
    created_on       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sim_ads (
    id          TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES sim_campaigns (id) ON DELETE CASCADE,
    variant     TEXT NOT NULL,
    headline    TEXT NOT NULL,
    body        TEXT NOT NULL,
    cta         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sim_stats (
    campaign_id TEXT NOT NULL,
    variant     TEXT NOT NULL,
    day         TEXT NOT NULL,
    impressions INTEGER NOT NULL,
    clicks      INTEGER NOT NULL,
    spend_usd   REAL NOT NULL,
    PRIMARY KEY (campaign_id, variant, day)
);
CREATE TABLE IF NOT EXISTS sim_conversions (
    campaign_id TEXT NOT NULL,
    variant     TEXT NOT NULL,
    click_day   TEXT NOT NULL,
    reported_on TEXT NOT NULL,
    conversions INTEGER NOT NULL,
    PRIMARY KEY (campaign_id, variant, click_day, reported_on)
);
CREATE TABLE IF NOT EXISTS sim_pages (
    id       TEXT PRIMARY KEY,
    slug     TEXT NOT NULL,
    headline TEXT NOT NULL,
    body     TEXT NOT NULL,
    status   TEXT NOT NULL
);

-- FR-07: segmentation ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS leads (
    id         TEXT PRIMARY KEY,
    company    TEXT NOT NULL,
    industry   TEXT NOT NULL,
    employees  INTEGER NOT NULL,
    role       TEXT NOT NULL,
    behaviour  TEXT NOT NULL,
    segment_id TEXT,
    embedding  BLOB
);
CREATE TABLE IF NOT EXISTS segments (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    size       INTEGER NOT NULL,
    persona    TEXT NOT NULL,
    centroid   BLOB,
    created_at TEXT NOT NULL
);

-- FR-09: budget ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS budget_recommendations (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL,
    approval_id TEXT,
    body        TEXT NOT NULL
);

-- FR-10: caches ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    query      TEXT NOT NULL,
    value      TEXT NOT NULL,
    embedding  BLOB,
    version    TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    hits       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (namespace, key)
);

-- Failure modes: alerts -------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    campaign_id   TEXT,
    experiment_id TEXT,
    status        TEXT NOT NULL,
    detail        TEXT NOT NULL
);
"""


class Database:
    """A single shared connection guarded by a re-entrant lock.

    `tx()` nests: only the outermost block begins and commits, so services can call each
    other inside one atomic unit.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 10000")
        if self.path != ":memory:":
            # WAL lets the MCP sidecar processes read while the app writes.
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._depth = 0
        self.migrate()

    def migrate(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self._conn
            except BaseException:
                self._depth -= 1
                if outermost:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outermost:
                    self._conn.execute("COMMIT")

    def next_id(self, prefix: str, width: int = 3, start: int = 1) -> str:
        """Human-readable sequential ids such as EXP-113 or APR-0007."""
        with self.tx() as conn:
            row = conn.execute("SELECT value FROM counters WHERE name = ?", (prefix,)).fetchone()
            value = start if row is None else row["value"] + 1
            conn.execute(
                "INSERT INTO counters (name, value) VALUES (?, ?) "
                "ON CONFLICT (name) DO UPDATE SET value = excluded.value",
                (prefix, value),
            )
        return f"{prefix}-{value:0{width}d}"

    def bump_counter(self, prefix: str, at_least: int) -> None:
        """Make sure the next id issued for `prefix` is greater than `at_least`."""
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO counters (name, value) VALUES (?, ?) "
                "ON CONFLICT (name) DO UPDATE SET value = MAX(value, excluded.value)",
                (prefix, at_least),
            )

    def get_state(self, key: str) -> str | None:
        with self.tx() as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_state(self, key: str, value: str) -> None:
        with self.tx() as conn:
            conn.execute(
                "INSERT INTO app_state (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
