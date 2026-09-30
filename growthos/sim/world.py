"""Ground truth of the simulated market.

The agent never reads this module: it only sees what the platforms report (spend, clicks,
delayed conversions). Tests and the simulator use it to generate data and to check that the
agent's conclusions match reality.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelTruth:
    name: str
    base_cpc: float
    # Cost per click rises as (1 + daily_spend / saturation): diminishing returns.
    saturation_usd_per_day: float
    base_cvr: float  # click -> lead (the experiment conversion)
    signup_to_paid: float  # lead -> paying customer
    always_on_daily_usd: float

    def cpc(self, daily_spend_usd: float) -> float:
        return self.base_cpc * (1 + max(daily_spend_usd, 0.0) / self.saturation_usd_per_day)

    def clicks_per_day(self, daily_spend_usd: float) -> float:
        if daily_spend_usd <= 0:
            return 0.0
        return daily_spend_usd / self.cpc(daily_spend_usd)

    def customers_per_day(self, daily_spend_usd: float) -> float:
        return self.clicks_per_day(daily_spend_usd) * self.base_cvr * self.signup_to_paid


@dataclass(frozen=True)
class SegmentTruth:
    id: str
    name: str
    mrr_usd: float
    industries: tuple[str, ...]
    employees: tuple[int, int]
    roles: tuple[str, ...]
    behaviours: tuple[str, ...]


CHANNELS: dict[str, ChannelTruth] = {
    c.name: c
    for c in (
        ChannelTruth("google_ads", 1.50, 3600, 0.30, 0.00917, 900),
        ChannelTruth("meta_ads", 0.62, 1800, 0.25, 0.00392, 500),
        ChannelTruth("linkedin_ads", 2.30, 2800, 0.30, 0.01027, 700),
        ChannelTruth("email", 0.35, 400, 0.22, 0.00818, 60),
    )
}

SEGMENTS: dict[str, SegmentTruth] = {
    s.id: s
    for s in (
        SegmentTruth(
            "seg-smb-finance",
            "SMB finance teams",
            240,
            ("retail", "ecommerce", "professional services", "manufacturing"),
            (15, 120),
            ("finance manager", "head of finance", "accountant"),
            (
                "viewed month-end close checklist",
                "downloaded spreadsheet reconciliation template",
                "searched how to close the books faster",
                "attended webinar on manual journal entries",
                "viewed pricing page twice",
            ),
        ),
        SegmentTruth(
            "seg-midmarket-controllers",
            "Mid-market controllers",
            420,
            ("software", "healthcare", "logistics", "fintech"),
            (200, 1500),
            ("controller", "vp finance", "cfo"),
            (
                "requested soc 2 report",
                "viewed multi-entity consolidation docs",
                "asked about erp integration with netsuite",
                "downloaded audit readiness guide",
                "booked demo with sales engineer",
            ),
        ),
        SegmentTruth(
            "seg-startup-founders",
            "Startup founders",
            150,
            ("software", "marketplace", "consumer apps", "ai tooling"),
            (2, 25),
            ("founder", "ceo", "chief of staff"),
            (
                "signed up from product hunt launch",
                "viewed runway and burn dashboard",
                "connected stripe and bank account",
                "asked about investor update templates",
                "started free trial on mobile",
            ),
        ),
        SegmentTruth(
            "seg-bookkeeping-agencies",
            "Bookkeeping agencies",
            310,
            ("accounting services", "bookkeeping", "fractional cfo"),
            (3, 40),
            ("partner", "bookkeeper", "practice manager"),
            (
                "asked about multi-client workspace",
                "viewed partner program page",
                "requested white label reports",
                "imported clients from quickbooks",
                "joined partner onboarding webinar",
            ),
        ),
    )
}

# Share of each channel's customers that falls into each segment.
CHANNEL_SEGMENT_MIX: dict[str, dict[str, float]] = {
    "google_ads": {
        "seg-smb-finance": 0.45,
        "seg-midmarket-controllers": 0.15,
        "seg-startup-founders": 0.25,
        "seg-bookkeeping-agencies": 0.15,
    },
    "meta_ads": {
        "seg-smb-finance": 0.35,
        "seg-midmarket-controllers": 0.05,
        "seg-startup-founders": 0.45,
        "seg-bookkeeping-agencies": 0.15,
    },
    "linkedin_ads": {
        "seg-smb-finance": 0.40,
        "seg-midmarket-controllers": 0.40,
        "seg-startup-founders": 0.05,
        "seg-bookkeeping-agencies": 0.15,
    },
    "email": {
        "seg-smb-finance": 0.35,
        "seg-midmarket-controllers": 0.20,
        "seg-startup-founders": 0.15,
        "seg-bookkeeping-agencies": 0.30,
    },
    "organic": {
        "seg-smb-finance": 0.35,
        "seg-midmarket-controllers": 0.15,
        "seg-startup-founders": 0.30,
        "seg-bookkeeping-agencies": 0.20,
    },
}

ORGANIC_CUSTOMERS_PER_DAY = 0.72
MONTHLY_CHURN = 0.025
GROSS_MARGIN = 0.80
SM_OVERHEAD_MONTHLY_USD = 22_000.0
OTHER_OPEX_MONTHLY_USD = 600_000.0
CASH_NOW_USD = 4_200_000.0

# Reporting delay for conversions, in days after the click: platforms attribute late.
CONVERSION_DELAY_DAYS: tuple[int, ...] = (1, 2, 3)
CONVERSION_DELAY_PROBS: tuple[float, ...] = (0.5, 0.3, 0.2)


@dataclass(frozen=True)
class _Tactic:
    name: str
    keywords: tuple[str, ...]
    lift: float
    segment_lift: dict[str, float]


# How copy really moves signup rate. Hidden from the agent; it has to learn this by testing.
_TACTICS: tuple[_Tactic, ...] = (
    _Tactic(
        "pain_point",
        ("close books", "close the books", "month-end", "late night", "spreadsheet", "manual", "chaos", "tired of"),
        0.22,
        {"seg-smb-finance": 0.10, "seg-startup-founders": -0.08, "seg-bookkeeping-agencies": 0.05},
    ),
    _Tactic(
        "social_proof",
        ("trusted by", "join ", "finance teams", "customers", "rated"),
        0.08,
        {"seg-midmarket-controllers": 0.10},
    ),
    _Tactic(
        "quantified_outcome",
        ("hours", " days", "x faster", "save ", "%"),
        0.10,
        {"seg-midmarket-controllers": 0.06},
    ),
    _Tactic(
        "urgency_discount",
        ("limited", "today only", "% off", "discount", "free month"),
        -0.06,
        {"seg-startup-founders": 0.18, "seg-midmarket-controllers": -0.06},
    ),
    _Tactic(
        "generic_feature",
        ("platform", "solution", "all-in-one", "powerful", "next-generation"),
        -0.05,
        {},
    ),
)

_MIN_MULTIPLIER = 0.4
_MAX_MULTIPLIER = 2.0
_NOISE_SD = 0.04


def copy_multiplier(text: str, segment_id: str, noise_key: str) -> float:
    """True signup-rate multiplier of a piece of copy for a segment."""
    lowered = text.lower()
    lift = 0.0
    for tactic in _TACTICS:
        if any(keyword in lowered for keyword in tactic.keywords):
            lift += tactic.lift + tactic.segment_lift.get(segment_id, 0.0)
    digest = hashlib.sha256(noise_key.encode()).digest()
    noise = np.random.default_rng(int.from_bytes(digest[:8], "big")).normal(0.0, _NOISE_SD)
    return float(np.clip(1 + lift + noise, _MIN_MULTIPLIER, _MAX_MULTIPLIER))


def true_cvr(channel: str, segment_id: str, text: str, noise_key: str) -> float:
    return CHANNELS[channel].base_cvr * copy_multiplier(text, segment_id, noise_key)
