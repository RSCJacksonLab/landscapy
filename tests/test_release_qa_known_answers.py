"""Known-answer tests for publication-facing analysis routines."""

from __future__ import annotations

import numpy as np
import pytest

from fitness_landscape.analysis.adaptive_walk import (
    adaptive_walk_stochastic,
    analyze_path_accessibility,
    calculate_basin_of_attraction_greedy,
    find_greedy_accessible_paths,
    neutral_network_analysis,
)
from fitness_landscape.analysis.dirichlet_energy import (
    calculate_ruggedness_dirichlet_energy,
    local_dirichlet_energy_contribution,
)
from fitness_landscape.analysis.epistasis import (
    calculate_epistasis_ensemble,
    calculate_epistasis_reference_free,
    calculate_epistasis_regression,
    calculate_epistasis_walsh,
)
from fitness_landscape.analysis.statistics import (
    analyze_fitness_distribution,
    hypothesis_testing,
    permutation_test,
)
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BinarySequence


@pytest.fixture
def additive_square_landscape() -> FitnessLandscape:
    sequences = [
        BinarySequence.from_bits(bits)
        for bits in ([0, 0], [0, 1], [1, 0], [1, 1])
    ]
    layer = NumericFitness("default", [[0.0], [1.0], [1.0], [2.0]])
    return FitnessLandscape.build(
        sequences,
        fitness_layers={"default": layer},
        graph="hamming",
    )


def test_accessible_paths_on_additive_square(additive_square_landscape):
    landscape = additive_square_landscape
    start, end = landscape.sequences[0], landscape.sequences[3]

    result = find_greedy_accessible_paths(landscape, start, end)

    assert result["path_count"] == 2
    assert {tuple(path["indices"]) for path in result["paths"]} == {
        (0, 1, 3),
        (0, 2, 3),
    }
    assert result["mean_path_length"] == 2
    assert all(np.all(np.diff(path["fitness"]) > 0) for path in result["paths"])


def test_accessibility_and_greedy_basin_cover_additive_square(
    additive_square_landscape,
):
    landscape = additive_square_landscape

    accessibility = analyze_path_accessibility(landscape)
    basin = calculate_basin_of_attraction_greedy(
        landscape, landscape.sequences[3]
    )

    assert accessibility["local_minima"] == [0]
    assert accessibility["local_maxima"] == [3]
    assert accessibility["paths_to_maxima"] == {0: {3: 2}}
    assert accessibility["accessibility"] == 1.0
    assert set(basin["basin"]) == {0, 1, 2, 3}
    assert basin["basin_fraction"] == 1.0


def test_adaptive_walk_is_monotone_and_stops_at_optimum(
    additive_square_landscape,
):
    result = adaptive_walk_stochastic(
        additive_square_landscape,
        start_sequence=additive_square_landscape.sequences[0],
        strategy="greedy",
    )

    assert result["walk_indices"][0] == 0
    assert result["walk_indices"][-1] == 3
    assert np.all(np.diff(result["walk_fitness"]) > 0)
    assert result["steps_taken"] == 2
    assert result["fitness_gain"] == 2.0
    assert result["reached_optimum"] is True


def test_adaptive_walk_input_failures_are_actionable(additive_square_landscape):
    missing = BinarySequence.from_bits([1, 1, 1])

    with pytest.raises(ValueError, match="Start sequence not found"):
        find_greedy_accessible_paths(
            additive_square_landscape,
            missing,
            additive_square_landscape.sequences[3],
        )
    with pytest.raises(ValueError, match="End sequence not found"):
        find_greedy_accessible_paths(
            additive_square_landscape,
            additive_square_landscape.sequences[0],
            missing,
        )
    with pytest.raises(ValueError, match="not a local optimum"):
        calculate_basin_of_attraction_greedy(
            additive_square_landscape,
            additive_square_landscape.sequences[0],
        )
    with pytest.raises(ValueError, match="Unsupported walk strategy"):
        adaptive_walk_stochastic(
            additive_square_landscape,
            start_sequence=additive_square_landscape.sequences[0],
            strategy="unsupported",
        )


def test_neutral_networks_at_zero_and_unit_threshold(additive_square_landscape):
    separate = neutral_network_analysis(additive_square_landscape, threshold=0.0)
    connected = neutral_network_analysis(additive_square_landscape, threshold=1.0)

    assert separate["network_count"] == 4
    assert separate["largest_network_size"] == 1
    assert connected["network_count"] == 1
    assert connected["largest_network_size"] == 4


def test_dirichlet_energy_matches_square_graph_quadratic_form(
    additive_square_landscape,
):
    result = calculate_ruggedness_dirichlet_energy(additive_square_landscape)
    local = local_dirichlet_energy_contribution(additive_square_landscape)

    # Four unit-difference edges give f.T @ L @ f == 4; the public metric is
    # normalized by the four nodes.
    assert result["total_dirichlet_energy"] == pytest.approx(1.0)
    assert sum(local.values()) == pytest.approx(4.0)
    assert all(value == pytest.approx(1.0) for value in local.values())


@pytest.mark.parametrize(
    "analyzer",
    [
        calculate_epistasis_walsh,
        calculate_epistasis_regression,
        calculate_epistasis_ensemble,
        calculate_epistasis_reference_free,
    ],
)
def test_additive_landscape_has_zero_second_order_epistasis(
    additive_square_landscape,
    analyzer,
):
    result = analyzer(additive_square_landscape, order=2)

    assert result["by_order"][2]
    assert max(abs(value) for value in result["by_order"][2].values()) < 1e-12


def test_distribution_summary_has_known_moments(additive_square_landscape):
    result = analyze_fitness_distribution(additive_square_landscape)

    assert result["sample_size"] == 4
    assert result["mean"] == 1.0
    assert result["median"] == 1.0
    assert result["std"] == pytest.approx(np.sqrt(0.5))
    assert result["range"] == 2.0
    assert result["skewness"] == pytest.approx(0.0)


def test_hypothesis_testing_removes_nan_and_returns_all_requested_tests():
    result = hypothesis_testing(
        groups={
            "low": np.array([0.0, 1.0, np.nan, 2.0]),
            "high": np.array([3.0, 4.0, 5.0]),
        }
    )

    assert result["group_stats"]["low"]["n"] == 3
    assert set(result["pairwise_tests"]["low"]["high"]) == {
        "t_test",
        "mann_whitney",
        "ks_test",
    }


def test_permutation_test_detects_a_large_location_shift():
    np.random.seed(7)
    result = permutation_test(
        groups={"low": np.arange(5.0), "high": np.arange(20.0, 25.0)},
        n_permutations=499,
        alternative="two-sided",
    )

    comparison = result[("low", "high")]
    assert comparison["observed"] == -20.0
    assert comparison["p_value"] < 0.05
    assert comparison["significant"] is True
