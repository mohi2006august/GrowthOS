"""FR-08: stopping rules, including the false-positive-rate validation from TECHNICAL.md."""

from __future__ import annotations

import numpy as np
import pytest

from growthos.analysis.bayes import AnalysisConfig, VariantData, analyse
from growthos.analysis.stopping import StoppingRule, decide

RULE = StoppingRule(min_sample=500, max_sample=2000)


def _posterior(conv_a: int, conv_b: int, n: int = 2000):
    return analyse(
        [VariantData("A", n, conv_a, 1000.0), VariantData("B", n, conv_b, 1000.0)],
        control="A",
        config=AnalysisConfig(samples=20_000),
    )


class TestSchedule:
    def test_looks_are_evenly_spaced_between_min_and_max(self) -> None:
        assert RULE.checkpoints() == [500, 1000, 1500, 2000]
        assert [RULE.look_for(n) for n in (0, 499, 500, 1700, 2000, 9999)] == [0, 0, 1, 3, 4, 4]

    def test_thresholds_never_drop_below_the_spec_and_spend_at_most_alpha(self) -> None:
        thresholds = [RULE.ship_threshold(look) for look in (1, 2, 3, 4)]
        assert all(t >= 0.95 for t in thresholds)
        assert thresholds == sorted(thresholds, reverse=True)
        assert sum(1 - t for t in thresholds) == pytest.approx(RULE.alpha)

    def test_more_challengers_means_a_stricter_threshold(self) -> None:
        assert RULE.ship_threshold(4, challengers=2) > RULE.ship_threshold(4, challengers=1)

    def test_single_look_uses_the_plain_spec_threshold(self) -> None:
        fixed = StoppingRule(min_sample=2000, max_sample=2000, looks=1)
        assert fixed.checkpoints() == [2000]
        assert fixed.ship_threshold(1) == 0.95

    def test_invalid_rules_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            StoppingRule(min_sample=2000, max_sample=500)
        with pytest.raises(ValueError):
            RULE.ship_threshold(5)


class TestDecide:
    def test_waits_for_the_minimum_sample(self) -> None:
        verdict = decide(_posterior(30, 60, n=300), RULE, {"A": 300, "B": 300})
        assert verdict.decision == "continue"
        assert verdict.look == 0
        assert "minimum sample" in verdict.reason

    def test_ships_a_clear_winner(self) -> None:
        verdict = decide(_posterior(200, 290), RULE, {"A": 2000, "B": 2000})
        assert verdict.decision == "ship"
        assert verdict.winner == "B"

    def test_does_not_peek_again_between_looks(self) -> None:
        samples = {"A": 1200, "B": 1200}
        posterior = _posterior(120, 190, n=1200)
        assert decide(posterior, RULE, samples, looks_done=1).decision == "ship"
        again = decide(posterior, RULE, samples, looks_done=2)
        assert again.decision == "continue"
        assert "between looks" in again.reason

    def test_kills_for_futility_when_the_challenger_is_clearly_worse(self) -> None:
        verdict = decide(_posterior(130, 80, n=1000), RULE, {"A": 1000, "B": 1000})
        assert verdict.decision == "kill"
        assert verdict.winner == "A"
        assert "futility" in verdict.reason

    def test_keeps_control_when_inconclusive_at_the_planned_sample(self) -> None:
        verdict = decide(_posterior(200, 212), RULE, {"A": 2000, "B": 2000})
        assert verdict.decision == "kill"
        assert verdict.winner is None
        assert "inconclusive" in verdict.reason

    def test_continues_at_an_interim_look_without_a_winner(self) -> None:
        verdict = decide(_posterior(100, 112, n=1000), RULE, {"A": 1000, "B": 1000})
        assert verdict.decision == "continue"
        assert verdict.look == 2

    def test_final_flag_forces_the_last_look_when_no_more_data_is_coming(self) -> None:
        samples = {"A": 900, "B": 900}
        posterior = _posterior(90, 100, n=900)
        assert decide(posterior, RULE, samples, looks_done=1).decision == "continue"
        assert decide(posterior, RULE, samples, looks_done=1, final=True).decision == "kill"

    def test_a_large_expected_loss_blocks_shipping(self) -> None:
        samples = {"A": 2000, "B": 2000}
        posterior = _posterior(200, 248)
        assert decide(posterior, RULE, samples).decision == "ship"
        strict = StoppingRule(min_sample=500, max_sample=2000, expected_loss_epsilon=1e-9)
        assert decide(posterior, strict, samples).decision != "ship"


def _simulate(true_a: float, true_b: float, runs: int, seed: int) -> tuple[float, float]:
    """Run sequential experiments; return (ship rate with the rule, ship rate if peeking at 0.95)."""
    rng = np.random.default_rng(seed)
    checkpoints = RULE.checkpoints()
    increments = np.diff([0, *checkpoints])
    shipped = naive = 0
    for run in range(runs):
        conv_a = np.cumsum(rng.binomial(increments, true_a))
        conv_b = np.cumsum(rng.binomial(increments, true_b))
        looks_done, stopped, rule_shipped, naive_shipped = 0, False, False, False
        for i, n in enumerate(checkpoints):
            posterior = analyse(
                [VariantData("A", n, int(conv_a[i]), 1.0), VariantData("B", n, int(conv_b[i]), 1.0)],
                control="A",
                config=AnalysisConfig(samples=4000, seed=run * 10 + i),
            )
            naive_shipped = naive_shipped or posterior.prob_best["B"] >= 0.95
            if stopped:
                continue
            verdict = decide(posterior, RULE, {"A": n, "B": n}, looks_done=looks_done)
            looks_done = verdict.look
            stopped = verdict.decision != "continue"
            rule_shipped = verdict.decision == "ship"
        shipped += rule_shipped
        naive += naive_shipped
    return shipped / runs, naive / runs


@pytest.mark.slow
def test_false_positive_rate_on_aa_tests_is_at_most_five_percent() -> None:
    rule_rate, naive_rate = _simulate(0.10, 0.10, runs=1000, seed=1)
    assert rule_rate <= 0.05
    # Peeking at a flat 0.95 on every look is what the look schedule protects against.
    assert naive_rate > 0.07


@pytest.mark.slow
def test_real_effects_are_still_detected() -> None:
    rule_rate, _ = _simulate(0.10, 0.13, runs=400, seed=2)
    assert rule_rate > 0.6
