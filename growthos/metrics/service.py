"""Metrics layer (FR-01): channel CAC, payback, LTV and burn/runway from the source tables."""

from __future__ import annotations

import calendar
import math
import re
from datetime import date, timedelta

from pydantic import BaseModel

from growthos.clock import Clock
from growthos.db import Database
from growthos.metrics import burn, cac, payback

ORGANIC = "organic"
_PERIOD = re.compile(r"^\s*(\d+)\s*([dwm])\s*$", re.IGNORECASE)
_DAYS_PER_UNIT = {"d": 1, "w": 7, "m": 30}
# Burn is averaged over the last closed months so one odd month does not swing runway.
_BURN_WINDOW_MONTHS = 3


class MetricsUnavailable(RuntimeError):
    """The source tables do not hold enough data to compute the requested metric."""


class ChannelMetrics(BaseModel):
    channel: str
    period_days: int
    start: date
    end: date
    media_spend_usd: float
    overhead_usd: float
    new_customers: int
    cac_usd: float | None
    fully_loaded_cac_usd: float | None
    arpa_usd: float | None
    payback_months: float | None
    ltv_usd: float | None
    ltv_to_cac: float | None


class BlendedMetrics(BaseModel):
    period_days: int
    media_spend_usd: float
    overhead_usd: float
    paid_customers: int
    organic_customers: int
    paid_cac_usd: float | None
    blended_cac_usd: float | None
    load_factor: float


class FinanceSnapshot(BaseModel):
    as_of_month: str
    cash_usd: float
    revenue_usd: float
    gross_margin: float
    gross_burn_usd: float
    net_burn_usd: float
    # None means the company is not burning cash (infinite runway).
    runway_months: float | None
    arpa_usd: float
    monthly_churn: float
    net_new_arr_usd: float
    burn_multiple: float | None


class MonthlyPoint(BaseModel):
    month: str
    channel: str
    spend_usd: float
    new_customers: int


def parse_period(period: str) -> int:
    """'30d', '12w' or '3m' to a number of days."""
    match = _PERIOD.match(period)
    if not match:
        raise ValueError(f"period must look like 30d, 12w or 3m, got {period!r}")
    days = int(match.group(1)) * _DAYS_PER_UNIT[match.group(2).lower()]
    if days <= 0:
        raise ValueError("period must be positive")
    return days


class MetricsService:
    def __init__(self, db: Database, clock: Clock) -> None:
        self._db = db
        self._clock = clock

    # -- channels ------------------------------------------------------------

    def channels(self) -> list[str]:
        with self._db.tx() as conn:
            rows = conn.execute("SELECT DISTINCT channel FROM channel_spend ORDER BY channel").fetchall()
        return [row["channel"] for row in rows]

    def channel_metrics(self, channel: str, period_days: int = 90) -> ChannelMetrics:
        start, end = self._window(period_days)
        spend = self._spend_by_channel(start, end)
        if channel not in spend:
            raise MetricsUnavailable(f"no spend recorded for channel {channel!r} in the last {period_days}d")
        overhead = cac.allocate_overhead(spend, self._overhead(start, end))[channel]
        customers, arpa = self._acquired(channel, start, end)
        finance = self.finance_snapshot()
        arpa_for_payback = arpa if arpa is not None else finance.arpa_usd
        loaded = cac.fully_loaded_cac(spend[channel], overhead, customers)
        lifetime = payback.ltv(arpa_for_payback, finance.gross_margin, finance.monthly_churn)
        return ChannelMetrics(
            channel=channel,
            period_days=period_days,
            start=start,
            end=end,
            media_spend_usd=spend[channel],
            overhead_usd=overhead,
            new_customers=customers,
            cac_usd=cac.cac(spend[channel], customers),
            fully_loaded_cac_usd=loaded,
            arpa_usd=arpa,
            payback_months=payback.payback_months(loaded, arpa_for_payback, finance.gross_margin),
            ltv_usd=lifetime,
            ltv_to_cac=payback.ltv_to_cac(lifetime, loaded),
        )

    def all_channel_metrics(self, period_days: int = 90) -> list[ChannelMetrics]:
        return [self.channel_metrics(channel, period_days) for channel in self.channels()]

    def blended(self, period_days: int = 90) -> BlendedMetrics:
        start, end = self._window(period_days)
        spend = self._spend_by_channel(start, end)
        media = sum(spend.values())
        overhead = self._overhead(start, end)
        paid = sum(self._acquired(channel, start, end)[0] for channel in spend)
        organic, _ = self._acquired(ORGANIC, start, end)
        return BlendedMetrics(
            period_days=period_days,
            media_spend_usd=media,
            overhead_usd=overhead,
            paid_customers=paid,
            organic_customers=organic,
            paid_cac_usd=cac.fully_loaded_cac(media, overhead, paid),
            blended_cac_usd=cac.blended_cac(media, overhead, paid, organic),
            load_factor=cac.load_factor(media, overhead),
        )

    def load_factor(self, period_days: int = 90) -> float:
        """Fully loaded cost per dollar of media spend."""
        start, end = self._window(period_days)
        return cac.load_factor(sum(self._spend_by_channel(start, end).values()), self._overhead(start, end))

    def monthly_history(self, months: int = 12) -> list[MonthlyPoint]:
        """Spend and new customers per channel and closed month, for response-curve fitting."""
        current = self._clock.today().strftime("%Y-%m")
        with self._db.tx() as conn:
            spend_rows = conn.execute(
                "SELECT substr(day, 1, 7) AS month, channel, SUM(spend_usd) AS spend "
                "FROM channel_spend WHERE substr(day, 1, 7) < ? GROUP BY month, channel",
                (current,),
            ).fetchall()
            customer_rows = conn.execute(
                "SELECT substr(acquired_on, 1, 7) AS month, channel, COUNT(*) AS n "
                "FROM customers WHERE substr(acquired_on, 1, 7) < ? GROUP BY month, channel",
                (current,),
            ).fetchall()
        acquired = {(row["month"], row["channel"]): row["n"] for row in customer_rows}
        keep = sorted({row["month"] for row in spend_rows})[-months:]
        return [
            MonthlyPoint(
                month=row["month"],
                channel=row["channel"],
                spend_usd=row["spend"],
                new_customers=acquired.get((row["month"], row["channel"]), 0),
            )
            for row in sorted(spend_rows, key=lambda r: (r["month"], r["channel"]))
            if row["month"] in keep
        ]

    # -- finance -------------------------------------------------------------

    def finance_snapshot(self) -> FinanceSnapshot:
        with self._db.tx() as conn:
            months = conn.execute(
                "SELECT * FROM finance_monthly ORDER BY month DESC LIMIT ?", (_BURN_WINDOW_MONTHS,)
            ).fetchall()
            if not months:
                raise MetricsUnavailable("finance_monthly is empty; load bank and billing data first")
            media_by_month = {
                row["month"]: row["spend"]
                for row in conn.execute(
                    "SELECT substr(day, 1, 7) AS month, SUM(spend_usd) AS spend "
                    "FROM channel_spend GROUP BY month"
                ).fetchall()
            }
        n = len(months)
        revenue = sum(m["revenue_usd"] for m in months) / n
        cogs = sum(m["cogs_usd"] for m in months) / n
        opex = sum(m["sm_overhead_usd"] + m["other_opex_usd"] for m in months) / n
        media = sum(media_by_month.get(m["month"], 0.0) for m in months) / n
        gross = burn.gross_burn(cogs, opex, media)
        net = burn.net_burn(gross, revenue)
        latest = months[0]
        runway = burn.runway_months(latest["cash_end_usd"], net)
        month_end = _month_end(latest["month"])
        arpa, churn, net_new_mrr = self._recurring_revenue(month_end)
        return FinanceSnapshot(
            as_of_month=latest["month"],
            cash_usd=latest["cash_end_usd"],
            revenue_usd=revenue,
            gross_margin=(revenue - cogs) / revenue if revenue > 0 else 0.0,
            gross_burn_usd=gross,
            net_burn_usd=net,
            runway_months=None if math.isinf(runway) else runway,
            arpa_usd=arpa,
            monthly_churn=churn,
            net_new_arr_usd=net_new_mrr * 12,
            burn_multiple=burn.burn_multiple(net, net_new_mrr * 12),
        )

    # -- internals -----------------------------------------------------------

    def _window(self, period_days: int) -> tuple[date, date]:
        if period_days <= 0:
            raise ValueError("period_days must be positive")
        # Half-open [start, end): the last `period_days` complete days, excluding today.
        end = self._clock.today()
        return end - timedelta(days=period_days), end

    def _spend_by_channel(self, start: date, end: date) -> dict[str, float]:
        with self._db.tx() as conn:
            rows = conn.execute(
                "SELECT channel, SUM(spend_usd) AS spend FROM channel_spend "
                "WHERE day >= ? AND day < ? GROUP BY channel",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return {row["channel"]: row["spend"] for row in rows}

    def _acquired(self, channel: str, start: date, end: date) -> tuple[int, float | None]:
        with self._db.tx() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, AVG(mrr_usd) AS arpa FROM customers "
                "WHERE channel = ? AND acquired_on >= ? AND acquired_on < ?",
                (channel, start.isoformat(), end.isoformat()),
            ).fetchone()
        return row["n"], row["arpa"]

    def _overhead(self, start: date, end: date) -> float:
        """Sales-and-marketing overhead for the window, prorated by day.

        Days in months that are not closed yet use the latest closed month's rate.
        """
        with self._db.tx() as conn:
            rows = conn.execute("SELECT month, sm_overhead_usd FROM finance_monthly ORDER BY month").fetchall()
        if not rows:
            return 0.0
        by_month = {row["month"]: row["sm_overhead_usd"] for row in rows}
        latest = rows[-1]["sm_overhead_usd"]
        total = 0.0
        day = start
        while day < end:
            month = day.strftime("%Y-%m")
            total += by_month.get(month, latest) / calendar.monthrange(day.year, day.month)[1]
            day += timedelta(days=1)
        return total

    def _recurring_revenue(self, as_of: date) -> tuple[float, float, float]:
        """ARPA, monthly churn and net new MRR over the 90 days before `as_of`."""
        start = (as_of - timedelta(days=90)).isoformat()
        end = as_of.isoformat()
        with self._db.tx() as conn:
            active = conn.execute(
                "SELECT COUNT(*) AS n, AVG(mrr_usd) AS arpa FROM customers "
                "WHERE acquired_on <= ? AND (churned_on IS NULL OR churned_on > ?)",
                (end, end),
            ).fetchone()
            active_at_start = conn.execute(
                "SELECT COUNT(*) AS n FROM customers "
                "WHERE acquired_on <= ? AND (churned_on IS NULL OR churned_on > ?)",
                (start, start),
            ).fetchone()["n"]
            churned = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(mrr_usd), 0) AS mrr FROM customers "
                "WHERE churned_on > ? AND churned_on <= ?",
                (start, end),
            ).fetchone()
            new_mrr = conn.execute(
                "SELECT COALESCE(SUM(mrr_usd), 0) AS mrr FROM customers "
                "WHERE acquired_on > ? AND acquired_on <= ?",
                (start, end),
            ).fetchone()["mrr"]
        if active["n"] == 0:
            raise MetricsUnavailable("no active customers; load CRM data first")
        churn = churned["n"] / active_at_start / 3 if active_at_start else 0.0
        return active["arpa"], churn, (new_mrr - churned["mrr"]) / 3


def _month_end(month: str) -> date:
    year, mon = (int(part) for part in month.split("-"))
    return date(year, mon, calendar.monthrange(year, mon)[1])
