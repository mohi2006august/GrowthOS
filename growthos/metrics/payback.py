"""CAC payback and lifetime value (FR-01)."""

from __future__ import annotations


def payback_months(cac_usd: float | None, arpa_monthly_usd: float, gross_margin: float) -> float | None:
    """Months of gross-margin-adjusted revenue needed to recover CAC."""
    _check_margin(gross_margin)
    if cac_usd is None:
        return None
    monthly_margin = arpa_monthly_usd * gross_margin
    if monthly_margin <= 0:
        return None
    return cac_usd / monthly_margin


def ltv(arpa_monthly_usd: float, gross_margin: float, monthly_churn: float) -> float | None:
    """Gross-margin lifetime value: margin per month over expected lifetime (1 / churn)."""
    _check_margin(gross_margin)
    if monthly_churn <= 0:
        return None
    return arpa_monthly_usd * gross_margin / monthly_churn


def ltv_to_cac(ltv_usd: float | None, cac_usd: float | None) -> float | None:
    if ltv_usd is None or cac_usd is None or cac_usd <= 0:
        return None
    return ltv_usd / cac_usd


def _check_margin(gross_margin: float) -> None:
    if not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1")
