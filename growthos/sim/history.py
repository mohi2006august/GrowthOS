"""Synthetic company history: a year of ad spend, customers and finance.

Stands in for the real sources (ads APIs, CRM, Stripe, bank) so the metrics layer, the
optimiser and the agent have something realistic to work on.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np

from growthos.clock import Clock
from growthos.db import Database
from growthos.sim import world

SIM_START = date(2026, 10, 1)
HISTORY_MONTHS = 12
_BASE_CUSTOMERS = 1500
_BASE_LOOKBACK_DAYS = 730
_BASE_CHANNEL_WEIGHTS = {
    "google_ads": 0.34,
    "meta_ads": 0.16,
    "linkedin_ads": 0.20,
    "email": 0.08,
    "organic": 0.22,
}
_DAILY_SPEND_NOISE = 0.08
_MRR_NOISE = 0.15
# Spend levels vary month to month so response curves can be fitted; the last quarter sits
# near the always-on level so "current" metrics are stable.
_EARLY_MULTIPLIER = (0.55, 1.45)
_RECENT_MULTIPLIER = (0.92, 1.08)
_RECENT_MONTHS = 3


def seed_history(db: Database, clock: Clock, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    clock.set_today(SIM_START)
    months = _months_before(SIM_START, HISTORY_MONTHS)
    window_start = date(months[0][0], months[0][1], 1)

    customers: list[dict] = []
    for _ in range(_BASE_CUSTOMERS):
        acquired = window_start - timedelta(days=int(rng.integers(1, _BASE_LOOKBACK_DAYS)))
        channel = _choice(rng, _BASE_CHANNEL_WEIGHTS)
        customers.append(_customer(rng, len(customers), acquired, channel))

    spend_rows: list[tuple[str, str, float]] = []
    for index, (year, month) in enumerate(months):
        recent = index >= len(months) - _RECENT_MONTHS
        low, high = _RECENT_MULTIPLIER if recent else _EARLY_MULTIPLIER
        multipliers = {name: rng.uniform(low, high) for name in world.CHANNELS}
        for day_number in range(1, calendar.monthrange(year, month)[1] + 1):
            day = date(year, month, day_number)
            for name, truth in world.CHANNELS.items():
                noise = max(1 + rng.normal(0, _DAILY_SPEND_NOISE), 0.0)
                spend = truth.always_on_daily_usd * multipliers[name] * noise
                spend_rows.append((day.isoformat(), name, round(spend, 2)))
                for _ in range(int(rng.poisson(truth.customers_per_day(spend)))):
                    customers.append(_customer(rng, len(customers), day, name))
            for _ in range(int(rng.poisson(world.ORGANIC_CUSTOMERS_PER_DAY))):
                customers.append(_customer(rng, len(customers), day, "organic"))

    for customer in customers:
        customer["churned_on"] = _churn_date(rng, customer["acquired_on"], window_start)

    finance = _finance_rows(months, customers, spend_rows)
    with db.tx() as conn:
        conn.executemany("INSERT INTO channel_spend (day, channel, spend_usd) VALUES (?, ?, ?)", spend_rows)
        conn.executemany(
            "INSERT INTO customers (id, acquired_on, channel, segment_id, mrr_usd, churned_on) "
            "VALUES (:id, :acquired_on, :channel, :segment_id, :mrr_usd, :churned_on)",
            [
                {
                    **c,
                    "acquired_on": c["acquired_on"].isoformat(),
                    "churned_on": c["churned_on"].isoformat() if c["churned_on"] else None,
                }
                for c in customers
            ],
        )
        conn.executemany(
            "INSERT INTO finance_monthly "
            "(month, revenue_usd, cogs_usd, sm_overhead_usd, other_opex_usd, cash_end_usd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            finance,
        )


def _months_before(day: date, count: int) -> list[tuple[int, int]]:
    year, month = day.year, day.month
    months: list[tuple[int, int]] = []
    for _ in range(count):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        months.append((year, month))
    return list(reversed(months))


def _choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights)
    probs = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=probs / probs.sum()))]


def _customer(rng: np.random.Generator, index: int, acquired: date, channel: str) -> dict:
    segment_id = _choice(rng, world.CHANNEL_SEGMENT_MIX[channel])
    mrr = world.SEGMENTS[segment_id].mrr_usd * float(rng.lognormal(0, _MRR_NOISE))
    return {
        "id": f"cus-{index + 1:05d}",
        "acquired_on": acquired,
        "channel": channel,
        "segment_id": segment_id,
        "mrr_usd": round(mrr, 2),
        "churned_on": None,
    }


def _churn_date(rng: np.random.Generator, acquired: date, window_start: date) -> date | None:
    """Churn is only simulated inside the history window; the base cohort starts it alive."""
    at_risk_from = max(acquired, window_start)
    months_alive = int(rng.geometric(world.MONTHLY_CHURN))
    churned = at_risk_from + timedelta(days=30 * (months_alive - 1) + int(rng.integers(1, 31)))
    return churned if churned < SIM_START else None


def _finance_rows(
    months: list[tuple[int, int]], customers: list[dict], spend_rows: list[tuple[str, str, float]]
) -> list[tuple[str, float, float, float, float, float]]:
    media_by_month: dict[str, float] = {}
    for day, _, spend in spend_rows:
        media_by_month[day[:7]] = media_by_month.get(day[:7], 0.0) + spend

    rows: list[dict] = []
    for year, month in months:
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        revenue = sum(
            c["mrr_usd"]
            for c in customers
            if c["acquired_on"] <= month_end and (c["churned_on"] is None or c["churned_on"] > month_end)
        )
        key = f"{year}-{month:02d}"
        cogs = revenue * (1 - world.GROSS_MARGIN)
        net = (
            cogs
            + world.SM_OVERHEAD_MONTHLY_USD
            + world.OTHER_OPEX_MONTHLY_USD
            + media_by_month[key]
            - revenue
        )
        rows.append({"month": key, "revenue": revenue, "cogs": cogs, "net_burn": net})

    # Walk cash backwards from today's balance so the latest month ends on CASH_NOW_USD.
    cash = world.CASH_NOW_USD
    for row in reversed(rows):
        row["cash_end"] = cash
        cash += row["net_burn"]
    return [
        (
            r["month"],
            round(r["revenue"], 2),
            round(r["cogs"], 2),
            world.SM_OVERHEAD_MONTHLY_USD,
            world.OTHER_OPEX_MONTHLY_USD,
            round(r["cash_end"], 2),
        )
        for r in rows
    ]
