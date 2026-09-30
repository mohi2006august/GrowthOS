"""Business date. The simulator moves it forward; production follows the wall clock."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from growthos.db import Database

_SIM_TODAY = "sim_today"


class Clock:
    def __init__(self, db: Database) -> None:
        self._db = db

    def today(self) -> date:
        simulated = self._db.get_state(_SIM_TODAY)
        return date.fromisoformat(simulated) if simulated else datetime.now(UTC).date()

    def set_today(self, day: date) -> None:
        self._db.set_state(_SIM_TODAY, day.isoformat())

    def advance(self, days: int = 1) -> date:
        day = self.today() + timedelta(days=days)
        self.set_today(day)
        return day


def utc_now_iso() -> str:
    """Wall-clock timestamp for the audit trail, which never follows simulated time."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")
