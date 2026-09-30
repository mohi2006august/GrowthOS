"""Bayesian A/B analysis (FR-08).

Conversions per variant follow Beta(1 + s, 1 + n - s). CAC per variant is cost divided by
expected customers, sampled from that posterior, so a variant with few conversions carries
wide CAC uncertainty instead of a falsely precise point estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from growthos.models import PrimaryMetric


@dataclass(frozen=True)
class VariantData:
    name: str
    visitors: int
    conversions: int
    spend_usd: float

    def __post_init__(self) -> None:
        if self.visitors < 0 or self.conversions < 0 or self.spend_usd < 0:
            raise ValueError(f"variant {self.name!r} has negative counts or spend")
        if self.conversions > self.visitors:
            raise ValueError(f"variant {self.name!r} has more conversions than visitors")


@dataclass(frozen=True)
class AnalysisConfig:
    primary_metric: PrimaryMetric = "cvr"
    samples: int = 40_000
    seed: int = 0
    # Share of experiment conversions (trial signups) that become paying customers.
    downstream_rate: float = 1.0
    # Fully loaded cost per dollar of media spend (see metrics.cac.load_factor).
    load_factor: float = 1.0
    arpa_usd: float | None = None
    gross_margin: float | None = None
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    interval: tuple[float, float] = (0.05, 0.95)


@dataclass(frozen=True)
class Posterior:
    control: str
    primary_metric: PrimaryMetric
    prob_best: dict[str, float]
    prob_beats_control: dict[str, float]
    expected_loss: dict[str, float]
    lift_vs_control: dict[str, float]
    cvr_mean: dict[str, float]
    cvr_interval: dict[str, tuple[float, float]]
    cac_mean: dict[str, float | None]
    cac_interval: dict[str, tuple[float, float] | None]
    payback_months: dict[str, float | None]

    @property
    def leader(self) -> str:
        return max(self.prob_best, key=lambda name: self.prob_best[name])


def analyse(variants: Sequence[VariantData], control: str, config: AnalysisConfig | None = None) -> Posterior:
    cfg = config or AnalysisConfig()
    names = [v.name for v in variants]
    if len(set(names)) != len(names) or len(names) < 2:
        raise ValueError("analysis needs at least two uniquely named variants")
    if control not in names:
        raise ValueError(f"control {control!r} is not one of {names}")
    if not 0 < cfg.downstream_rate <= 1 or cfg.load_factor < 1:
        raise ValueError("downstream_rate must be in (0, 1] and load_factor at least 1")

    rng = np.random.default_rng(cfg.seed)
    alpha = np.array([cfg.prior_alpha + v.conversions for v in variants])
    beta = np.array([cfg.prior_beta + v.visitors - v.conversions for v in variants])
    rates = rng.beta(alpha, beta, size=(cfg.samples, len(variants)))

    visitors = np.array([v.visitors for v in variants], dtype=float)
    cost = np.array([v.spend_usd for v in variants]) * cfg.load_factor
    # Customers per dollar: comparable across variants and finite even when CAC is not.
    customers = rates * visitors * cfg.downstream_rate
    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(cost > 0, customers / cost, 0.0)
        cac = np.where(customers > 0, cost / customers, np.inf)

    # "Goodness" is always higher-is-better, so one code path serves every metric.
    goodness = rates if cfg.primary_metric == "cvr" else efficiency
    best = goodness.max(axis=1, keepdims=True)
    winners = goodness.argmax(axis=1)
    control_index = names.index(control)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_loss = np.where(best > 0, (best - goodness) / best, 0.0)
    metric = rates if cfg.primary_metric == "cvr" else cac
    low, high = cfg.interval

    cac_mean: dict[str, float | None] = {}
    cac_interval: dict[str, tuple[float, float] | None] = {}
    payback: dict[str, float | None] = {}
    lift: dict[str, float] = {}
    for i, v in enumerate(variants):
        cac_mean[v.name] = _cac_mean(v, cfg)
        usable = v.visitors > 0 and v.spend_usd > 0
        cac_interval[v.name] = _interval(cac[:, i], low, high) if usable else None
        payback[v.name] = _payback(cac_mean[v.name], cfg)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = metric[:, i] / metric[:, control_index]
        finite = ratio[np.isfinite(ratio)]
        lift[v.name] = float(np.median(finite) - 1) if finite.size else 0.0

    return Posterior(
        control=control,
        primary_metric=cfg.primary_metric,
        prob_best={n: float(np.mean(winners == i)) for i, n in enumerate(names)},
        prob_beats_control={
            n: float(np.mean(goodness[:, i] > goodness[:, control_index]))
            for i, n in enumerate(names)
            if n != control
        },
        expected_loss={n: float(relative_loss[:, i].mean()) for i, n in enumerate(names)},
        lift_vs_control=lift,
        cvr_mean={n: float(alpha[i] / (alpha[i] + beta[i])) for i, n in enumerate(names)},
        cvr_interval={n: _interval(rates[:, i], low, high) for i, n in enumerate(names)},
        cac_mean=cac_mean,
        cac_interval=cac_interval,
        payback_months=payback,
    )


def _cac_mean(v: VariantData, cfg: AnalysisConfig) -> float | None:
    """Posterior mean of cost / (visitors * rate * downstream).

    E[1 / rate] under Beta(a, b) is (a + b - 1) / (a - 1), finite only for a > 1, so the mean
    is undefined until the posterior has moved off the prior. The closed form avoids the
    heavy-tailed Monte Carlo estimate.
    """
    a = cfg.prior_alpha + v.conversions
    b = cfg.prior_beta + v.visitors - v.conversions
    if v.visitors == 0 or v.spend_usd == 0 or a <= 1:
        return None
    inverse_rate = (a + b - 1) / (a - 1)
    return v.spend_usd * cfg.load_factor * inverse_rate / (v.visitors * cfg.downstream_rate)


def _payback(cac_usd: float | None, cfg: AnalysisConfig) -> float | None:
    if cac_usd is None or not cfg.arpa_usd or not cfg.gross_margin:
        return None
    return cac_usd / (cfg.arpa_usd * cfg.gross_margin)


def _interval(draws: np.ndarray, low: float, high: float) -> tuple[float, float]:
    lo, hi = np.quantile(draws, [low, high])
    return float(lo), float(hi)


def describe(posterior: Posterior, variant: str) -> str:
    """One line a growth lead can read, e.g.
    'Variant B: CAC -22% (94% probability better), payback 6.1 -> 4.8 months'."""
    label = {"cvr": "CVR", "cac": "CAC", "payback": "Payback"}[posterior.primary_metric]
    change = posterior.lift_vs_control[variant]
    probability = posterior.prob_beats_control.get(variant, 0.0)
    line = f"{variant}: {label} {change:+.0%} ({probability:.0%} probability better)"
    before = posterior.payback_months.get(posterior.control)
    after = posterior.payback_months.get(variant)
    if before is not None and after is not None:
        line += f", payback {before:.1f} -> {after:.1f} months"
    return line
