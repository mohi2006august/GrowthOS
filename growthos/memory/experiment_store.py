"""Experiment memory (FR-02): structured experiment records plus narrative learnings.

Experiments are the system of record. Finished experiments and free-form notes (win/loss
notes, customer interviews, competitor pages) are also indexed as memory documents so new
proposals can cite them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
from pydantic import BaseModel

from growthos.clock import Clock, utc_now_iso
from growthos.db import Database, blob_to_vector, vector_to_blob
from growthos.embeddings import Embedder
from growthos.models import FINISHED_STATUSES, Experiment, ExperimentStatus

DocKind = Literal["experiment", "win_loss", "interview", "competitor"]
_EXPERIMENT_PREFIX = "EXP"
_NOTE_PREFIX = "NOTE"


class ExperimentNotFound(KeyError):
    pass


class MemoryDoc(BaseModel):
    id: str
    kind: DocKind
    experiment_id: str | None = None
    title: str
    text: str
    source: str | None = None
    created_at: str


class ExperimentStore:
    def __init__(self, db: Database, embedder: Embedder, clock: Clock) -> None:
        self._db = db
        self._embedder = embedder
        self._clock = clock

    # -- experiments ---------------------------------------------------------

    def new_id(self) -> str:
        return self._db.next_id(_EXPERIMENT_PREFIX)

    def save(self, experiment: Experiment) -> Experiment:
        """Insert or update; finished experiments are (re)indexed into memory."""
        now = utc_now_iso()
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO experiments (id, status, channel, segment_id, created_at, updated_at, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET status = excluded.status, channel = excluded.channel, "
                "segment_id = excluded.segment_id, updated_at = excluded.updated_at, body = excluded.body",
                (
                    experiment.id,
                    experiment.status,
                    experiment.channel,
                    experiment.segment_id,
                    now,
                    now,
                    experiment.model_dump_json(),
                ),
            )
            if experiment.status in FINISHED_STATUSES:
                self._index_experiment(experiment)
        return experiment

    def get(self, experiment_id: str) -> Experiment:
        with self._db.tx() as conn:
            row = conn.execute("SELECT body FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise ExperimentNotFound(experiment_id)
        return Experiment.model_validate_json(row["body"])

    def list_experiments(self, statuses: Iterable[ExperimentStatus] | None = None) -> list[Experiment]:
        query = "SELECT body FROM experiments"
        params: tuple[str, ...] = ()
        if statuses is not None:
            params = tuple(statuses)
            query += f" WHERE status IN ({', '.join('?' for _ in params)})"
        with self._db.tx() as conn:
            rows = conn.execute(query + " ORDER BY id", params).fetchall()
        return [Experiment.model_validate_json(row["body"]) for row in rows]

    # -- memory documents ----------------------------------------------------

    def add_note(
        self,
        kind: DocKind,
        title: str,
        text: str,
        *,
        experiment_id: str | None = None,
        source: str | None = None,
    ) -> MemoryDoc:
        doc = MemoryDoc(
            id=self._db.next_id(_NOTE_PREFIX),
            kind=kind,
            experiment_id=experiment_id,
            title=title,
            text=text,
            source=source,
            created_at=self._clock.today().isoformat(),
        )
        self._upsert_doc(doc)
        return doc

    def docs(self) -> tuple[list[MemoryDoc], np.ndarray]:
        """All memory documents and their embedding matrix (one row per document)."""
        with self._db.tx() as conn:
            rows = conn.execute("SELECT * FROM memory_docs ORDER BY id").fetchall()
        docs = [
            MemoryDoc(
                id=row["id"],
                kind=row["kind"],
                experiment_id=row["experiment_id"],
                title=row["title"],
                text=row["text"],
                source=row["source"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        if not rows:
            return docs, np.zeros((0, self._embedder.dim), dtype=np.float32)
        return docs, np.vstack([blob_to_vector(row["embedding"]) for row in rows])

    def index_version(self) -> str:
        """Changes whenever memory changes; caches key on it to avoid stale answers."""
        with self._db.tx() as conn:
            row = conn.execute("SELECT COUNT(*) AS n, MAX(rowid) AS latest FROM memory_docs").fetchone()
            updated = conn.execute("SELECT MAX(updated_at) AS ts FROM experiments").fetchone()["ts"]
        return f"{row['n']}:{row['latest']}:{updated}"

    def _index_experiment(self, experiment: Experiment) -> None:
        self._upsert_doc(
            MemoryDoc(
                id=f"exp:{experiment.id}",
                kind="experiment",
                experiment_id=experiment.id,
                title=f"{experiment.id}: {experiment.hypothesis}",
                text=experiment_text(experiment),
                source=None,
                created_at=(experiment.ended_on or experiment.created_on).isoformat(),
            )
        )

    def _upsert_doc(self, doc: MemoryDoc) -> None:
        embedding = self._embedder.embed([f"{doc.title}\n{doc.text}"])[0]
        with self._db.tx() as conn:
            conn.execute(
                "INSERT INTO memory_docs (id, kind, experiment_id, title, text, source, created_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET title = excluded.title, text = excluded.text, "
                "source = excluded.source, created_at = excluded.created_at, embedding = excluded.embedding",
                (
                    doc.id,
                    doc.kind,
                    doc.experiment_id,
                    doc.title,
                    doc.text,
                    doc.source,
                    doc.created_at,
                    vector_to_blob(embedding),
                ),
            )


def experiment_text(experiment: Experiment) -> str:
    """The narrative that gets embedded and retrieved for a finished experiment."""
    parts = [
        f"Channel: {experiment.channel.replace('_', ' ')}. Segment: {experiment.segment_id.replace('-', ' ')}.",
        f"Tactic: {(experiment.tactic or 'unspecified').replace('_', ' ')}.",
        f"Variants: {'; '.join(experiment.variants)}.",
    ]
    if experiment.outcome:
        parts.append(f"Outcome: {experiment.outcome}.")
    if experiment.result and experiment.result.summary:
        parts.append(experiment.result.summary)
    if experiment.learnings:
        parts.append(f"Learnings: {experiment.learnings}")
    return " ".join(parts)


def outcome_note(experiment: Experiment) -> str:
    """Short citation note, e.g. '+19% CVR, pain point copy'."""
    metric = {"cvr": "CVR", "cac": "CAC", "payback": "payback"}[experiment.primary_metric]
    tactic = (experiment.tactic or "untagged").replace("_", " ")
    if experiment.observed_lift is None:
        return f"{experiment.outcome or experiment.status}, {tactic}"
    return f"{experiment.observed_lift:+.0%} {metric}, {tactic} ({experiment.outcome})"
