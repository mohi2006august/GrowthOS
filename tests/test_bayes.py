"""FR-08: Beta-binomial posteriors, CAC distributions and the growth-lead summary line."""

from __future__ import annotations

import pytest

from growthos.analysis.bayes import AnalysisConfig, VariantData, analyse, describe

A = VariantData("A", visitors=2000, conversions=200, spend_usd=4000)
B = VariantData("B", visitors=2000, conversions=300, spend_usd=4000)


def test_clear_winner_has_high_probability_and_low_loss() -> None:
    posterior = analyse([A, B], control="A")
    assert posterior.prob_best["B"] > 0.99
    assert posterior.prob_beats_control["B"] > 0.99
    assert posterior.expected_loss["B"] < 0.001
    assert posterior.expected_loss["A"] > 0.2
    assert posterior.lift_vs_control["B"] == pytest.approx(0.5, abs=0.05)
    assert posterior.leader == "B"


def test_probabilities_sum_to_one_across_three_variants() -> None:
    c = VariantData("C", visitors=2000, conversions=210, spend_usd=4000)
    posterior = analyse([A, B, c], control="A")
    assert sum(posterior.prob_best.values()) == pytest.approx(1.0)
    assert set(posterior.prob_beats_control) == {"B", "C"}


def test_identical_data_is_a_coin_flip() -> None:
    twin = VariantData("B", visitors=2000, conversions=200, spend_usd=4000)
    posterior = analyse([A, twin], control="A")
    assert posterior.prob_best["B"] == pytest.approx(0.5, abs=0.02)
    assert posterior.lift_vs_control["B"] == pytest.approx(0.0, abs=0.02)


def test_cac_metric_penalises_expensive_clicks_even_at_equal_conversion_rate() -> None:
    pricey = VariantData("B", visitors=2000, conversions=200, spend_usd=8000)
    by_cvr = analyse([A, pricey], control="A", config=AnalysisConfig(primary_metric="cvr"))
    by_cac = analyse([A, pricey], control="A", config=AnalysisConfig(primary_metric="cac"))
    assert by_cvr.prob_best["B"] == pytest.approx(0.5, abs=0.02)
    assert by_cac.prob_best["A"] > 0.99
    assert by_cac.lift_vs_control["B"] == pytest.approx(1.0, abs=0.15)


def test_cac_mean_is_fully_loaded_and_counts_only_paying_customers() -> None:
    v = VariantData("A", visitors=1000, conversions=50, spend_usd=1000)
    other = VariantData("B", visitors=1000, conversions=50, spend_usd=1000)
    config = AnalysisConfig(primary_metric="cac", downstream_rate=0.5, load_factor=1.25)
    posterior = analyse([v, other], control="A", config=config)
    # E[1/rate] = (n + 1) / s under a flat prior.
    assert posterior.cac_mean["A"] == pytest.approx(1000 * 1.25 * (1001 / 50) / (1000 * 0.5))
    low, high = posterior.cac_interval["A"]  # type: ignore[misc]
    assert low < posterior.cac_mean["A"] < high


def test_cac_is_undefined_without_conversions() -> None:
    empty = VariantData("B", visitors=500, conversions=0, spend_usd=900)
    posterior = analyse([A, empty], control="A", config=AnalysisConfig(primary_metric="cac"))
    assert posterior.cac_mean["B"] is None
    assert posterior.payback_months["B"] is None
    assert posterior.prob_best["A"] > 0.99


def test_variant_with_no_traffic_cannot_win_on_cac() -> None:
    unlaunched = VariantData("B", visitors=0, conversions=0, spend_usd=0)
    posterior = analyse([A, unlaunched], control="A", config=AnalysisConfig(primary_metric="cac"))
    assert posterior.prob_best["A"] == 1.0
    assert posterior.cac_interval["B"] is None


def test_summary_line_matches_the_growth_lead_story() -> None:
    config = AnalysisConfig(primary_metric="cac", arpa_usd=250, gross_margin=0.8, downstream_rate=0.03)
    control = VariantData("Variant A", visitors=3000, conversions=300, spend_usd=11_000)
    challenger = VariantData("Variant B", visitors=3000, conversions=385, spend_usd=11_000)
    posterior = analyse([control, challenger], control="Variant A", config=config)
    line = describe(posterior, "Variant B")
    assert line.startswith("Variant B: CAC -2")
    assert "probability better" in line
    assert "payback" in line and "->" in line
    assert posterior.payback_months["Variant B"] < posterior.payback_months["Variant A"]  # type: ignore[operator]


def test_same_seed_gives_same_answer() -> None:
    first = analyse([A, B], control="A", config=AnalysisConfig(seed=11))
    second = analyse([A, B], control="A", config=AnalysisConfig(seed=11))
    assert first == second


@pytest.mark.parametrize(
    "variants, control",
    [
        ([A], "A"),
        ([A, VariantData("A", 10, 1, 1)], "A"),
        ([A, B], "C"),
    ],
)
def test_invalid_experiments_are_rejected(variants: list[VariantData], control: str) -> None:
    with pytest.raises(ValueError):
        analyse(variants, control=control)


def test_impossible_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        VariantData("A", visitors=10, conversions=11, spend_usd=5)
