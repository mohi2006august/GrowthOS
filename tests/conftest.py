from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from growthos.clock import Clock
from growthos.db import Database
from growthos.sim.history import seed_history


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
