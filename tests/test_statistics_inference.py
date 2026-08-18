"""Validation and known-answer tests for publication-facing statistics."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy import stats

from fitness_landscape.analysis.statistics import (
    _adjust_pvalues,
    _json_compatible,
    analyze_fitness_distribution,
    hypothesis_testing,
    permutation_test,
)


class _ValueLandscape:
    """Small duck-typed landscape for distribution-policy edge cases."""

    def __init__(self, values):
        self._values = list(values)
        self.sequences = list(range(len(self._values)))

    def get_fitness(self, sequence):
        return self._values[sequence]


def test_distribution_shapiro_matches_scipy_reference():
    values = np.array([-1.2, -0.3, 0.1, 0.4, 1.7, 2.2])
    result = analyze_fitness_distribution(_ValueLandscape(values), alpha=0.1)
    reference = stats.shapiro(values)

    assert result["sample_size"] == values.size
    assert result["omitted_count"] == 0
    assert result["normality_test"]["status"] == "performed"
    assert result["normality_test"]["shapiro_statistic"] == pytest.approx(
        reference.statistic
    )
    assert result["normality_test"]["shapiro_p_value"] == pytest.approx(
        reference.pvalue
    )
    assert result["normality_test"]["is_normal"] is bool(reference.pvalue >= 0.1)


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([1.0, 2.0], "at least 3"),
        ([1.0, 1.0, 1.0], "constant"),
    ],
)
def test_distribution_small_and_constant_samples_skip_shapiro(values, reason):
    result = analyze_fitness_distribution(_ValueLandscape(values))

    normality = result["normality_test"]
    assert normality["status"] == "not_run"
    assert reason in normality["reason"]
    assert normality["is_normal"] is None
    assert np.isnan(normality["shapiro_p_value"])


def test_distribution_large_sample_warns_and_skips_shapiro():
    landscape = _ValueLandscape(np.linspace(-1.0, 1.0, 5001))

    with pytest.warns(UserWarning, match="larger than 5000"):
        result = analyze_fitness_distribution(landscape)

    assert result["normality_test"]["status"] == "not_run"
    assert "maximum of 5000" in result["normality_test"]["reason"]


def test_distribution_nan_policy_is_explicit_and_infinities_are_rejected():
    landscape = _ValueLandscape([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="nan_policy='omit'"):
        analyze_fitness_distribution(landscape)

    result = analyze_fitness_distribution(landscape, nan_policy="omit")
    assert result["input_sample_size"] == 3
    assert result["sample_size"] == 2
    assert result["omitted_count"] == 1

    with pytest.raises(ValueError, match="infinity"):
        analyze_fitness_distribution(
            _ValueLandscape([1.0, np.inf]), nan_policy="omit"
        )

    with pytest.raises(TypeError, match="numeric"):
        analyze_fitness_distribution(_ValueLandscape(["not-numeric"]))
    with pytest.raises(ValueError, match="scalar"):
        analyze_fitness_distribution(_ValueLandscape([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="nan_policy"):
        analyze_fitness_distribution(
            _ValueLandscape([1.0, 2.0, 3.0]), nan_policy="bad"
        )
    with pytest.raises(TypeError, match="nan_policy"):
        analyze_fitness_distribution(
            _ValueLandscape([1.0, 2.0, 3.0]), nan_policy=[]
        )


@pytest.mark.parametrize("values", [[], [np.nan]])
def test_distribution_rejects_empty_post_policy_samples(values):
    with pytest.raises(ValueError, match="at least one finite value"):
        analyze_fitness_distribution(
            _ValueLandscape(values), nan_policy="omit"
        )


@pytest.mark.parametrize(
    ("alpha", "error"),
    [
        (0.0, ValueError),
        (1.0, ValueError),
        (-0.1, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        ("invalid", TypeError),
    ],
)
def test_distribution_rejects_invalid_alpha(alpha, error):
    with pytest.raises(error, match="alpha"):
        analyze_fitness_distribution(_ValueLandscape([1.0, 2.0, 3.0]), alpha=alpha)


def test_hypothesis_tests_match_scipy_references_without_correction():
    first = np.array([0.5, 1.5, 2.0, 4.5, 7.0])
    second = np.array([1.0, 2.5, 3.0, 3.5, 8.5])
    result = hypothesis_testing(
        groups={"first": first, "second": second},
        correction_method=None,
    )
    comparison = result["pairwise_tests"]["first"]["second"]

    references = {
        "t_test": stats.ttest_ind(first, second, equal_var=False),
        "mann_whitney": stats.mannwhitneyu(
            first, second, alternative="two-sided"
        ),
        "ks_test": stats.ks_2samp(
            first, second, alternative="two-sided", mode="auto"
        ),
    }
    for test_name, reference in references.items():
        observed = comparison[test_name]
        assert observed["statistic"] == pytest.approx(reference.statistic)
        assert observed["p_value"] == pytest.approx(reference.pvalue)
        assert observed["adjusted_p_value"] == pytest.approx(reference.pvalue)
        assert observed["correction_method"] == "none"

    assert result["correction_family_size"] == 3


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", [0.01, 0.04, 0.03]),
        ("bonferroni", [0.03, 0.12, 0.09]),
        ("holm", [0.03, 0.06, 0.06]),
        ("fdr_bh", [0.03, 0.04, 0.04]),
    ],
)
def test_multiple_testing_corrections_have_known_answers(method, expected):
    adjusted = _adjust_pvalues(np.array([0.01, 0.04, 0.03]), method)
    np.testing.assert_allclose(adjusted, expected)


def test_multiple_testing_helper_rejects_invalid_p_value_arrays():
    with pytest.raises(ValueError, match="one-dimensional"):
        _adjust_pvalues(np.ones((2, 2)), "holm")
    with pytest.raises(ValueError, match="finite"):
        _adjust_pvalues(np.array([0.1, np.nan]), "holm")
    assert _adjust_pvalues(np.array([]), "holm").size == 0


def test_hypothesis_testing_applies_default_holm_family():
    result = hypothesis_testing(
        groups={
            "a": np.array([0.0, 1.0, 2.0, 3.0]),
            "b": np.array([1.0, 2.0, 3.0, 4.0]),
            "c": np.array([8.0, 9.0, 10.0, 11.0]),
        },
        run_tests=("ttest",),
    )

    assert result["correction_method"] == "holm"
    assert result["correction_family_size"] == 3
    for first, second in (("a", "b"), ("a", "c"), ("b", "c")):
        test = result["pairwise_tests"][first][second]["t_test"]
        assert test["adjusted_p_value"] >= test["p_value"]
        assert test["significant"] is (
            test["adjusted_p_value"] < result["alpha"]
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"alpha": 0.0}, ValueError, "alpha"),
        ({"alpha": True}, TypeError, "alpha"),
        ({"run_tests": ()}, ValueError, "at least one"),
        ({"run_tests": "ttest"}, TypeError, "sequence"),
        ({"run_tests": ("unknown",)}, ValueError, "Unsupported"),
        ({"run_tests": ("ks", "ks")}, ValueError, "duplicate"),
        ({"run_tests": (1,)}, TypeError, "string"),
        ({"run_tests": 1}, TypeError, "sequence"),
        ({"correction_method": "bad"}, ValueError, "correction_method"),
        ({"correction_method": []}, TypeError, "correction_method"),
        ({"equal_var": 1}, TypeError, "equal_var"),
    ],
)
def test_hypothesis_testing_rejects_invalid_configuration(kwargs, error, message):
    with pytest.raises(error, match=message):
        hypothesis_testing(
            groups={"a": [1.0, 2.0], "b": [2.0, 3.0]},
            **kwargs,
        )


def test_hypothesis_testing_validates_groups_and_finite_values():
    with pytest.raises(ValueError, match="At least two"):
        hypothesis_testing(groups={"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least one finite"):
        hypothesis_testing(groups={"a": [], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="infinity"):
        hypothesis_testing(groups={"a": [1.0, np.inf], "b": [1.0, 2.0]})
    with pytest.raises(TypeError, match="numeric"):
        hypothesis_testing(groups={"a": ["bad"], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="nan_policy='omit'"):
        hypothesis_testing(groups={"a": [1.0, np.nan], "b": [1.0, 2.0]})
    with pytest.raises(ValueError, match="at least two observations"):
        hypothesis_testing(groups={"a": [1.0], "b": [1.0, 2.0]})

    result = hypothesis_testing(
        groups={"a": [1.0, np.nan], "b": [1.0, 2.0]},
        run_tests=("mannwhitney",),
        nan_policy="omit",
    )
    assert result["group_stats"]["a"]["n"] == 1


def test_hypothesis_testing_supports_both_landscape_extraction_modes():
    first = _ValueLandscape([1.0, 2.0])
    second = _ValueLandscape([3.0, 4.0])
    first.values = np.array([1.0, 2.0])
    second.values = np.array([3.0, 4.0])

    landscape_result = hypothesis_testing(
        landscapes={"first": first, "second": second},
        value_fn=lambda item: item.values,
        run_tests=("ks",),
    )
    assert landscape_result["group_stats"]["first"]["n"] == 2

    layer_values = {"first": [1.0, 2.0], "second": [3.0, 4.0]}
    layer_result = hypothesis_testing(
        landscape=object(),
        layer_names=("first", "second"),
        value_fn_layers=lambda _landscape, layer: layer_values[layer],
        run_tests=("ks",),
    )
    assert layer_result["group_stats"]["second"]["n"] == 2

    with pytest.raises(ValueError, match="duplicate"):
        hypothesis_testing(
            landscape=object(),
            layer_names=("first", "first"),
            value_fn_layers=lambda _landscape, layer: layer_values[layer],
            run_tests=("ks",),
        )


def test_hypothesis_testing_rejects_incomplete_or_mixed_input_modes():
    with pytest.raises(ValueError, match="exactly ONE"):
        hypothesis_testing()
    with pytest.raises(ValueError, match="provided together"):
        hypothesis_testing(landscapes={"a": object()})
    with pytest.raises(ValueError, match="provided together"):
        hypothesis_testing(landscape=object(), layer_names=("a", "b"))
    with pytest.raises(ValueError, match="exactly ONE"):
        hypothesis_testing(
            groups={"a": [1.0], "b": [2.0]},
            landscapes={"a": object()},
        )


def test_undefined_scipy_result_is_not_declared_significant():
    with pytest.warns(RuntimeWarning):
        result = hypothesis_testing(
            groups={"a": [1.0, 1.0], "b": [1.0, 1.0]},
            run_tests=("ttest",),
        )

    test = result["pairwise_tests"]["a"]["b"]["t_test"]
    assert np.isnan(test["p_value"])
    assert np.isnan(test["adjusted_p_value"])
    assert test["significant"] is False
    assert result["correction_family_size"] == 0


def test_seeded_permutation_test_is_reproducible_and_reports_monte_carlo_error():
    kwargs = {
        "groups": {"low": np.arange(5.0), "high": np.arange(20.0, 25.0)},
        "n_permutations": 99,
        "alternative": "two-sided",
        "random_state": 1729,
    }

    first = permutation_test(**kwargs)
    second = permutation_test(**kwargs)
    assert first == second

    comparison = first[("low", "high")]
    assert comparison["observed"] == -20.0
    assert comparison["extreme_count"] == 2
    assert comparison["p_value"] == 0.03
    assert comparison["p_value"] > 0.0
    assert comparison["p_value_resolution"] == 0.01
    assert comparison["monte_carlo_standard_error"] == pytest.approx(
        np.sqrt(99 * comparison["p_value"] * (1 - comparison["p_value"])) / 100
    )
    assert comparison["random_state"]["kind"] == "seed"
    assert comparison["random_state"]["seed"] == 1729
    assert comparison["random_state"]["bit_generator"] == "PCG64"


def test_generator_state_is_recorded_and_replays_a_permutation_comparison():
    groups = {"a": np.arange(4.0), "b": np.arange(5.0, 9.0)}
    generator = np.random.default_rng(31415)
    initial_state = copy.deepcopy(generator.bit_generator.state)

    result = permutation_test(
        groups=groups,
        n_permutations=31,
        random_state=generator,
    )
    comparison = result[("a", "b")]

    assert comparison["random_state"]["kind"] == "generator"
    assert comparison["random_state"]["state"] == initial_state
    assert generator.bit_generator.state != initial_state

    replay_bit_generator = np.random.PCG64()
    replay_bit_generator.state = comparison["random_state"]["state"]
    replay = permutation_test(
        groups=groups,
        n_permutations=31,
        random_state=np.random.Generator(replay_bit_generator),
    )
    assert replay[("a", "b")]["extreme_count"] == comparison["extreme_count"]
    assert replay[("a", "b")]["p_value"] == comparison["p_value"]


def test_random_state_metadata_is_json_compatible():
    value = {
        "array": np.array([1, 2]),
        "items": (np.int64(3), np.float64(4.5)),
    }
    assert _json_compatible(value) == {"array": [1, 2], "items": [3, 4.5]}

    result = permutation_test(
        groups={"a": [1.0], "b": [2.0]},
        n_permutations=1,
    )
    assert result[("a", "b")]["random_state"]["kind"] == "entropy"


def test_permutation_multiple_testing_returns_adjusted_p_values():
    result = permutation_test(
        groups={
            "a": np.arange(4.0),
            "b": np.arange(5.0, 9.0),
            "c": np.arange(10.0, 14.0),
        },
        n_permutations=49,
        random_state=7,
        correction_method="bonferroni",
    )

    assert len(result) == 3
    for comparison in result.values():
        assert comparison["correction_method"] == "bonferroni"
        assert comparison["correction_family_size"] == 3
        assert comparison["adjusted_p_value"] == pytest.approx(
            min(1.0, comparison["p_value"] * 3)
        )


@pytest.mark.parametrize(
    ("alternative", "expected_extreme_count"),
    [("greater", 0), ("less", 19)],
)
def test_seeded_one_sided_permutation_tests_have_known_answers(
    alternative, expected_extreme_count
):
    result = permutation_test(
        groups={"high": np.arange(5.0, 9.0), "low": np.arange(4.0)},
        n_permutations=19,
        alternative=alternative,
        random_state=11,
        correction_method=None,
    )

    comparison = result[("high", "low")]
    assert comparison["observed"] == 5.0
    assert comparison["extreme_count"] == expected_extreme_count
    assert comparison["p_value"] == (expected_extreme_count + 1) / 20


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"alternative": "invalid"}, ValueError, "alternative"),
        ({"alternative": []}, TypeError, "alternative"),
        ({"n_permutations": 0}, ValueError, "positive integer"),
        ({"n_permutations": -1}, ValueError, "positive integer"),
        ({"n_permutations": 2.5}, TypeError, "positive integer"),
        ({"n_permutations": True}, TypeError, "positive integer"),
        ({"alpha": 1.0}, ValueError, "alpha"),
        ({"random_state": "seed"}, TypeError, "random_state"),
        ({"random_state": -1}, ValueError, "non-negative"),
        ({"correction_method": "bad"}, ValueError, "correction_method"),
        ({"correction_method": []}, TypeError, "correction_method"),
        ({"statistic_func": 1}, TypeError, "callable"),
    ],
)
def test_permutation_test_rejects_invalid_configuration(kwargs, error, message):
    call_kwargs = {
        "groups": {"a": [1.0, 2.0], "b": [2.0, 3.0]},
        "n_permutations": 5,
    }
    call_kwargs.update(kwargs)
    with pytest.raises(error, match=message):
        permutation_test(**call_kwargs)


def test_permutation_test_validates_groups_values_and_statistic():
    with pytest.raises(ValueError, match="At least two"):
        permutation_test(groups={"a": [1.0]}, n_permutations=5)
    with pytest.raises(ValueError, match="at least one finite"):
        permutation_test(groups={"a": [], "b": [1.0]}, n_permutations=5)
    with pytest.raises(ValueError, match="infinity"):
        permutation_test(
            groups={"a": [1.0, np.inf], "b": [1.0]}, n_permutations=5
        )
    with pytest.raises(ValueError, match="nan_policy='omit'"):
        permutation_test(
            groups={"a": [1.0, np.nan], "b": [1.0]}, n_permutations=5
        )
    with pytest.raises(ValueError, match="finite scalar"):
        permutation_test(
            groups={"a": [1.0], "b": [2.0]},
            statistic_func=lambda _a, _b: np.nan,
            n_permutations=5,
        )
    with pytest.raises(ValueError, match="finite scalar"):
        permutation_test(
            groups={"a": [1.0], "b": [2.0]},
            statistic_func=lambda _a, _b: np.array([1.0, 2.0]),
            n_permutations=5,
        )

    result = permutation_test(
        groups={"a": [1.0, np.nan], "b": [2.0]},
        nan_policy="omit",
        n_permutations=5,
        random_state=1,
    )
    assert result[("a", "b")]["n_permutations"] == 5
