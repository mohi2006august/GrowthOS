"""Customer acquisition cost (FR-01).

Budget decisions use *fully loaded* CAC: media spend plus the sales-and-marketing overhead
(salaries, tools) allocated to each channel in proportion to its media spend.
"""

from __future__ import annotations

from collections.abc import Mapping


def cac(spend_usd: float, new_customers: float) -> float | None:
    """Spend per new customer, or None when nothing was acquired (CAC is undefined, not zero)."""
    if spend_usd < 0:
        raise ValueError("spend_usd cannot be negative")
    if new_customers <= 0:
        return None
    return spend_usd / new_customers


def allocate_overhead(media_spend_by_channel: Mapping[str, float], overhead_usd: float) -> dict[str, float]:
    """Split overhead across channels pro rata to media spend."""
    if overhead_usd < 0:
        raise ValueError("overhead_usd cannot be negative")
    total = sum(media_spend_by_channel.values())
    if total <= 0:
        return {channel: 0.0 for channel in media_spend_by_channel}
    return {
        channel: overhead_usd * spend / total for channel, spend in media_spend_by_channel.items()
    }


def load_factor(media_spend_usd: float, overhead_usd: float) -> float:
    """Multiplier that turns media-only cost into fully loaded cost."""
    if media_spend_usd <= 0:
        return 1.0
    return (media_spend_usd + overhead_usd) / media_spend_usd


def fully_loaded_cac(media_spend_usd: float, overhead_usd: float, new_customers: float) -> float | None:
    return cac(media_spend_usd + overhead_usd, new_customers)


def blended_cac(
    media_spend_usd: float, overhead_usd: float, paid_customers: float, organic_customers: float
) -> float | None:
    """All sales-and-marketing cost over all new customers. Hides channel truth; always slice."""
    return cac(media_spend_usd + overhead_usd, paid_customers + organic_customers)
