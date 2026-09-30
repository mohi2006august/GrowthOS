from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from growthos.clock import Clock
from growthos.db import Database
from growthos.embeddings import HashingEmbedder
from growthos.memory import ExperimentStore
from growthos.sim.history import seed_history
from growthos.sim.past_experiments import seed_memory


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "growthos.db")
    yield database
    database.close()


@pytest.fixture
def clock(db: Database) -> Clock:
    return Clock(db)


@pytest.fixture
def history(db: Database, clock: Clock) -> Database:
    """A database holding the synthetic year of spend, customers and finance."""
    seed_history(db, clock)
    return db


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store(db: Database, clock: Clock, embedder: HashingEmbedder) -> ExperimentStore:
    return ExperimentStore(db, embedder, clock)


@pytest.fixture
def memory(history: Database, store: ExperimentStore) -> ExperimentStore:
    """History plus the seeded past experiments and research notes."""
    seed_memory(history, store)
    return store
