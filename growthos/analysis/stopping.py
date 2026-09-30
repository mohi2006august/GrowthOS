"""Stopping rules (FR-08): ship, kill or continue.

Ship when P(best) clears the threshold and expected loss is below epsilon; kill early for
futility. Two things keep the false-positive rate at or under `alpha`:

* Decisions happen only at pre-registered looks (sample checkpoints), never on every refresh.
* The ship threshold tightens per look by spending alpha quadratically over the looks
  (a union bound), so peeking cannot accumulate error. With a single look the threshold is
  exactly `ship_probability`.

Callers must pass *matured* samples only: ad platforms report conversions 1-3 days late, so
recent clicks look like non-converters until the lag has passed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from growthos.analysis.bayes import Posterior
from growthos.models import Decision


@dataclass(frozen=True)
class StoppingRule:
    min_sample: int
    max_sample: int
    ship_probability: float = 0.95
    expected_loss_epsilon: float = 0.01
    futility_probability: float = 0.05
    looks: int = 4
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.min_sample <= 0 or self.max_sample < self.min_sample:
            raise ValueError("need 0 < min_sample <= max_sample")
        if self.looks < 1:
            raise ValueError("looks must be at least 1")
        if not 0.5 < self.ship_probability < 1 or not 0 < self.alpha < 0.5:
            raise ValueError("ship_probability must be in (0.5, 1) and alpha in (0, 0.5)")

    def checkpoints(self) -> list[int]:
        """Matured per-variant sample at which each look happens."""
        if self.looks == 1 or self.max_sample == self.min_sample:
            return [self.max_sample]
        step = (self.max_sample - self.min_sample) / (self.looks - 1)
        return [round(self.min_sample + step * i) for i in range(self.looks)]

    def look_for(self, matured_sample: int) -> int:
        """Number of checkpoints passed (0 means still below the minimum sample)."""
        return sum(1 for checkpoint in self.checkpoints() if matured_sample >= checkpoint)

    def ship_threshold(self, look: int, challengers: int = 1) -> float:
        """P(best) a challenger must reach at `look` (1-based)."""
        total = len(self.checkpoints())
        if not 1 <= look <= total:
            raise ValueError(f"look must be between 1 and {total}")
        if total == 1:
            return self.ship_probability
        spent = self.alpha * ((look / total) ** 2 - ((look - 1) / total) ** 2)
        return max(self.ship_probability, 1 - spent / max(challengers, 1))


@dataclass(frozen=True)
class StoppingDecision:
    decision: Decision
    winner: str | None
    reason: str
    look: int
    threshold: float | None


def decide(
    posterior: Posterior,
    rule: StoppingRule,
    matured_samples: Mapping[str, int],
    *,
    looks_done: int = 0,
    final: bool = False,
) -> StoppingDecision:
    """Evaluate the rule once per look.

    `looks_done` is how many looks were already evaluated for this experiment; re-running
    between checkpoints returns "continue" so hourly refreshes do not become extra peeks.
    `final` forces the last look when no more data will arrive (budget spent, lag elapsed).
    """
    smallest = min(matured_samples.values())
    total_looks = len(rule.checkpoints())
    look = total_looks if final else rule.look_for(smallest)

    if look == 0:
        return StoppingDecision(
            "continue", None, f"collecting minimum sample ({smallest}/{rule.min_sample} per variant)", 0, None
        )
    if look <= looks_done:
        upcoming = rule.checkpoints()[look] if look < total_looks else rule.max_sample
        return StoppingDecision(
            "continue", None, f"between looks; next decision at {upcoming} matured samples per variant", look, None
        )

    control = posterior.control
    challengers = [name for name in posterior.prob_best if name != control]
    threshold = rule.ship_threshold(look, len(challengers))
    leader = posterior.leader

    if (
        leader != control
        and posterior.prob_best[leader] >= threshold
        and posterior.expected_loss[leader] < rule.expected_loss_epsilon
    ):
        return StoppingDecision(
            "ship",
            leader,
            f"P(best) {posterior.prob_best[leader]:.3f} >= {threshold:.3f} and expected loss "
            f"{posterior.expected_loss[leader]:.4f} < {rule.expected_loss_epsilon}",
            look,
            threshold,
        )

    best_challenger_chance = max(posterior.prob_beats_control[name] for name in challengers)
    if best_challenger_chance < rule.futility_probability:
        return StoppingDecision(
            "kill",
            control,
            f"futility: no challenger has more than a {best_challenger_chance:.1%} chance of beating control",
            look,
            threshold,
        )

    if look >= total_looks:
        return StoppingDecision(
            "kill",
            None,
            f"inconclusive at the planned sample: best P(best) {posterior.prob_best[leader]:.3f} "
            f"< {threshold:.3f}; keep control",
            look,
            threshold,
        )

    return StoppingDecision(
        "continue",
        None,
        f"look {look}/{total_looks}: P(best) {posterior.prob_best[leader]:.3f} below {threshold:.3f}",
        look,
        threshold,
    )
