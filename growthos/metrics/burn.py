"""Burn rate and runway (FR-01). These functions back every spend guardrail."""

from __future__ import annotations

import math


def gross_burn(cogs_usd: float, opex_usd: float, marketing_usd: float) -> float:
    """Total monthly cash out."""
    return cogs_usd + opex_usd + marketing_usd


def net_burn(gross_burn_usd: float, revenue_usd: float) -> float:
    """Monthly cash out minus cash in. Negative means the company is profitable."""
    return gross_burn_usd - revenue_usd


def runway_months(cash_usd: float, net_burn_usd: float) -> float:
    """Months until cash hits zero at the given net burn; infinite when not burning."""
    if cash_usd <= 0:
        return 0.0
    if net_burn_usd <= 0:
        return math.inf
    return cash_usd / net_burn_usd


def runway_after_one_time(cash_usd: float, net_burn_usd: float, one_time_cost_usd: float) -> float:
    """Runway if a one-off cost (an experiment budget) comes straight out of cash."""
    return runway_months(cash_usd - one_time_cost_usd, net_burn_usd)


def runway_after_recurring(cash_usd: float, net_burn_usd: float, monthly_delta_usd: float) -> float:
    """Runway if monthly spend changes by `monthly_delta_usd` (a budget reallocation)."""
    return runway_months(cash_usd, net_burn_usd + monthly_delta_usd)


def max_one_time_spend(cash_usd: float, net_burn_usd: float, min_runway_months: float) -> float:
    """Largest one-off cost that still leaves `min_runway_months` of runway."""
    if net_burn_usd <= 0:
        return max(cash_usd, 0.0)
    return max(cash_usd - min_runway_months * net_burn_usd, 0.0)


def max_recurring_increase(cash_usd: float, net_burn_usd: float, min_runway_months: float) -> float:
    """Largest monthly burn increase that keeps runway at or above policy.

    Negative when the company is already below policy and must cut.
    """
    if min_runway_months <= 0:
        raise ValueError("min_runway_months must be positive")
    return cash_usd / min_runway_months - net_burn_usd


def burn_multiple(net_burn_usd: float, net_new_arr_usd: float) -> float | None:
    """Net burn per dollar of net new ARR; under 1.5 is efficient growth."""
    if net_new_arr_usd <= 0:
        return None
    return net_burn_usd / net_new_arr_usd
