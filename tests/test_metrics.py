"""FR-01: CAC, payback, LTV, burn and runway."""

from __future__ import annotations

import math
from datetime import date

import pytest

from growthos.clock import Clock
from growthos.db import Database
from growthos.metrics import MetricsService, MetricsUnavailable, burn, cac, parse_period, payback


class TestCac:
    def test_cac_is_spend_per_customer(self) -> None:
        assert cac.cac(10_000, 8) == 1250

    def test_cac_is_undefined_without_customers(self) -> None:
        assert cac.cac(10_000, 0) is None

    def test_negative_spend_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            cac.cac(-1, 3)

    def test_overhead_is_allocated_pro_rata_and_sums_to_total(self) -> None:
        allocated = cac.allocate_overhead({"google_ads": 3000, "linkedin_ads": 1000}, 2000)
        assert allocated == {"google_ads": 1500, "linkedin_ads": 500}

    def test_fully_loaded_cac_includes_overhead(self) -> None:
        assert cac.fully_loaded_cac(8000, 2000, 10) == 1000
        assert cac.load_factor(8000, 2000) == 1.25

    def test_blended_cac_is_lower_than_paid_because_organic_is_free(self) -> None:
        paid = cac.fully_loaded_cac(8000, 2000, 10)
        blended = cac.blended_cac(8000, 2000, 10, 10)
        assert paid is not None and blended is not None and blended < paid


class TestPayback:
    def test_payback_uses_gross_margin_revenue(self) -> None:
        assert payback.payback_months(1220, 250, 0.8) == pytest.approx(6.1)

    def test_ltv_and_ratio(self) -> None:
        lifetime = payback.ltv(250, 0.8, 0.025)
        assert lifetime == pytest.approx(8000)
        assert payback.ltv_to_cac(lifetime, 2000) == pytest.approx(4)

    def test_undefined_inputs_propagate_as_none(self) -> None:
        assert payback.payback_months(None, 250, 0.8) is None
        assert payback.ltv(250, 0.8, 0) is None

    def test_margin_outside_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            payback.payback_months(1000, 250, 1.4)


class TestBurn:
    def test_runway_is_cash_over_net_burn(self) -> None:
        net = burn.net_burn(burn.gross_burn(100_000, 500_000, 60_000), 450_000)
        assert net == 210_000
        assert burn.runway_months(4_200_000, net) == 20

    def test_profitable_company_has_infinite_runway(self) -> None:
        assert math.isinf(burn.runway_months(1_000_000, -5_000))

    def test_one_time_cost_comes_out_of_cash(self) -> None:
        assert burn.runway_after_one_time(4_200_000, 210_000, 21_000) == pytest.approx(19.9)

    def test_recurring_cost_raises_burn(self) -> None:
        assert burn.runway_after_recurring(4_200_000, 210_000, 23_333.33) == pytest.approx(18.0, abs=1e-3)

    def test_headroom_matches_policy_boundary(self) -> None:
        headroom = burn.max_recurring_increase(4_200_000, 210_000, 18)
        assert burn.runway_after_recurring(4_200_000, 210_000, headroom) == pytest.approx(18)
        one_time = burn.max_one_time_spend(4_200_000, 210_000, 18)
        assert burn.runway_after_one_time(4_200_000, 210_000, one_time) == pytest.approx(18)

    def test_headroom_is_negative_when_already_below_policy(self) -> None:
        assert burn.max_recurring_increase(3_000_000, 210_000, 18) < 0
        assert burn.max_one_time_spend(3_000_000, 210_000, 18) == 0

    def test_burn_multiple(self) -> None:
        assert burn.burn_multiple(210_000, 175_000) == pytest.approx(1.2)
        assert burn.burn_multiple(210_000, 0) is None


def test_parse_period() -> None:
    assert parse_period("30d") == 30
    assert parse_period("2w") == 14
    assert parse_period("3m") == 90
    with pytest.raises(ValueError):
        parse_period("last quarter")


@pytest.fixture
def small(db: Database, clock: Clock) -> MetricsService:
    """A hand-built dataset small enough to check every number by hand."""
    clock.set_today(date(2026, 7, 1))
    with db.tx() as conn:
        conn.executemany(
            "INSERT INTO channel_spend (day, channel, spend_usd) VALUES (?, ?, ?)",
            [(f"2026-06-{d:02d}", "google_ads", 300.0) for d in range(1, 31)]
            + [(f"2026-06-{d:02d}", "linkedin_ads", 100.0) for d in range(1, 31)],
        )
        conn.execute(
            "INSERT INTO finance_monthly VALUES ('2026-06', 5000, 1000, 3000, 20000, 240000)"
        )
        customers = (
            [(f"old-{i}", "2025-01-10", "organic", 100.0, None) for i in range(40)]
            + [(f"g-{i}", "2026-06-15", "google_ads", 100.0, None) for i in range(9)]
            + [(f"l-{i}", "2026-06-20", "linkedin_ads", 200.0, None) for i in range(2)]
            + [(f"o-{i}", "2026-06-05", "organic", 100.0, None) for i in range(1)]
            + [(f"x-{i}", "2025-02-01", "organic", 100.0, "2026-05-10") for i in range(3)]
        )
        conn.executemany(
            "INSERT INTO customers (id, acquired_on, channel, mrr_usd, churned_on) VALUES (?, ?, ?, ?, ?)",
            customers,
        )
    return MetricsService(db, clock)


class TestMetricsService:
    def test_channel_cac_is_fully_loaded(self, small: MetricsService) -> None:
        google = small.channel_metrics("google_ads", 30)
        # 30 days * $300 media; overhead $3000 split 3:1 with LinkedIn.
        assert google.media_spend_usd == 9000
        assert google.overhead_usd == pytest.approx(2250)
        assert google.new_customers == 9
        assert google.cac_usd == 1000
        assert google.fully_loaded_cac_usd == pytest.approx(1250)

    def test_payback_uses_channel_arpa_and_company_margin(self, small: MetricsService) -> None:
        linkedin = small.channel_metrics("linkedin_ads", 30)
        assert linkedin.fully_loaded_cac_usd == pytest.approx((3000 + 750) / 2)
        # Gross margin is (5000 - 1000) / 5000 = 0.8; LinkedIn ARPA is $200.
        assert linkedin.payback_months == pytest.approx(1875 / (200 * 0.8))

    def test_blended_hides_channel_truth(self, small: MetricsService) -> None:
        blended = small.blended(30)
        assert blended.paid_customers == 11
        assert blended.organic_customers == 1
        assert blended.paid_cac_usd == pytest.approx(15000 / 11)
        assert blended.blended_cac_usd == pytest.approx(15000 / 12)
        assert blended.load_factor == pytest.approx(1.25)

    def test_finance_snapshot(self, small: MetricsService) -> None:
        snapshot = small.finance_snapshot()
        # Gross burn: 1000 COGS + 23000 opex + 12000 media; net of 5000 revenue.
        assert snapshot.gross_burn_usd == pytest.approx(36000)
        assert snapshot.net_burn_usd == pytest.approx(31000)
        assert snapshot.runway_months == pytest.approx(240000 / 31000)
        assert snapshot.gross_margin == pytest.approx(0.8)
        # 3 of 43 customers churned over the 90 days before month end.
        assert snapshot.monthly_churn == pytest.approx(3 / 43 / 3)

    def test_unknown_channel_is_an_explicit_error(self, small: MetricsService) -> None:
        with pytest.raises(MetricsUnavailable):
            small.channel_metrics("tiktok_ads", 30)

    def test_monthly_history_feeds_curve_fitting(self, small: MetricsService) -> None:
        points = {(p.month, p.channel): p for p in small.monthly_history()}
        assert points[("2026-06", "google_ads")].spend_usd == 9000
        assert points[("2026-06", "google_ads")].new_customers == 9


class TestSeededHistory:
    def test_company_is_burning_with_runway_near_policy(self, history: Database, clock: Clock) -> None:
        snapshot = MetricsService(history, clock).finance_snapshot()
        assert snapshot.cash_usd == 4_200_000
        assert snapshot.runway_months is not None
        assert 18.5 < snapshot.runway_months < 22
        assert snapshot.gross_margin == pytest.approx(0.8)

    def test_every_paid_channel_has_a_plausible_cac(self, history: Database, clock: Clock) -> None:
        metrics = MetricsService(history, clock).all_channel_metrics(90)
        assert {m.channel for m in metrics} == {"google_ads", "meta_ads", "linkedin_ads", "email"}
        for channel in metrics:
            assert channel.fully_loaded_cac_usd is not None
            assert channel.cac_usd is not None
            assert channel.fully_loaded_cac_usd > channel.cac_usd
            assert channel.payback_months is not None and channel.payback_months > 0
