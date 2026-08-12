from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.linalg import expm

from fitness_landscape.analysis import time_continuous_autocorrelation
from fitness_landscape.analysis.random_walk import (
    calculate_ruggedness_autocorrelation_analytical,
    calculate_ruggedness_autocorrelation_stochastic,
)
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition


def _landscape(graph: nx.Graph, values) -> FitnessLandscape:
    sequences = [
        BaseNumpySequence([index], sequence_id=f"s{index}")
        for index in range(graph.number_of_nodes())
    ]
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["sequence"] = sequences[index]
    fitness = NumericFitness(name="fitness", values=values)
    return FitnessLandscape(
        sequences=sequences,
        graph=graph,
        fitness_layers={"fitness": fitness},
    )


def _stationary_quadratic_form(graph, values, lags, weight=None):
    adjacency = nx.to_numpy_array(graph, weight=weight, dtype=float)
    degree = adjacency.sum(axis=1)
    transition = adjacency / degree[:, None]
    stationary = degree / degree.sum()
    centered = np.asarray(values, dtype=float) - stationary @ values
    variance = stationary @ centered**2
    correlations = []
    propagated = centered.copy()
    for _ in lags:
        correlations.append(stationary @ (centered * propagated) / variance)
        propagated = transition @ propagated
    return transition, stationary, centered, np.asarray(correlations)


def test_four_node_path_matches_issue_178_exact_regression():
    landscape = _landscape(nx.path_graph(4), [0.0, 1.0, 4.0, 9.0])

    result = calculate_ruggedness_autocorrelation_analytical(
        landscape,
        lag_max=3,
    )

    np.testing.assert_array_equal(result["lags"], [0, 1, 2, 3])
    np.testing.assert_allclose(
        result["autocorrelation"],
        [1.0, 0.3617021276595745, 0.270516717325228, 0.06990881458966565],
        atol=1e-14,
    )
    assert np.max(np.abs(result["autocorrelation"])) <= 1.0


def test_regular_graph_matches_matrix_power_and_stadler_spectral_forms():
    graph = nx.cycle_graph(6)
    values = np.array([0.0, 1.0, 4.0, 2.0, -1.0, 3.0])
    landscape = _landscape(graph, values)
    lags = np.arange(8)

    public = calculate_ruggedness_autocorrelation_analytical(
        landscape,
        lag_max=int(lags[-1]),
    )["autocorrelation"]
    _, _, centered, direct = _stationary_quadratic_form(
        graph,
        values,
        lags,
    )
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    powers = np.square(eigenvectors.T @ centered)
    powers /= powers.sum()
    stadler = np.array(
        [np.sum(powers * np.power(1.0 - eigenvalues / 2.0, lag)) for lag in lags]
    )

    np.testing.assert_allclose(public, direct, atol=1e-14)
    np.testing.assert_allclose(public, stadler, atol=1e-14)


def test_weighted_irregular_graph_matches_direct_powers_and_matrix_exponential():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 2.0), (1, 3, 0.5)],
        weight="conductance",
    )
    values = np.array([0.0, 2.0, 5.0, -1.0])
    landscape = _landscape(graph, values)
    lags = np.arange(6)
    transition, stationary, centered, direct = _stationary_quadratic_form(
        graph,
        values,
        lags,
        weight="conductance",
    )

    discrete = calculate_ruggedness_autocorrelation_analytical(
        landscape,
        lag_max=5,
        weight_key="conductance",
    )
    np.testing.assert_allclose(discrete["autocorrelation"], direct, atol=1e-14)

    times = np.array([0.0, 0.25, 1.7, 4.0])
    variance = stationary @ centered**2
    expected_continuous = np.array(
        [
            stationary @ (centered * (expm(-time * (np.eye(4) - transition)) @ centered))
            / variance
            for time in times
        ]
    )
    continuous = time_continuous_autocorrelation(
        landscape,
        times=times,
        weight_key="conductance",
    )
    np.testing.assert_array_equal(continuous["times"], times)
    np.testing.assert_allclose(
        continuous["autocorrelation"],
        expected_continuous,
        atol=1e-13,
    )
    assert continuous["correlation_time"] is None


def test_continuous_scalar_time_does_not_get_relabelled_as_zero():
    landscape = _landscape(nx.path_graph(3), [0.0, 1.0, 4.0])

    at_one = time_continuous_autocorrelation(landscape, times=1.0)

    np.testing.assert_array_equal(at_one["times"], [1.0])
    assert 0.0 < at_one["autocorrelation"][0] < 1.0


def test_bipartite_elementary_mode_oscillates_discretely_and_decays_continuously():
    landscape = _landscape(nx.path_graph(2), [1.0, -1.0])

    discrete = calculate_ruggedness_autocorrelation_analytical(
        landscape,
        lag_max=4,
    )
    times = np.array([0.0, 0.25, 1.0, 2.0])
    continuous = time_continuous_autocorrelation(landscape, times=times)

    np.testing.assert_array_equal(discrete["autocorrelation"], [1, -1, 1, -1, 1])
    np.testing.assert_allclose(
        continuous["autocorrelation"],
        np.exp(-2.0 * times),
        atol=1e-14,
    )
    assert discrete["equivalent_single_exponential_length"] == np.inf
    assert discrete["elementary"] is True
    assert continuous["elementary"] is True
    assert continuous["elementary_correlation_time"] == pytest.approx(0.5)


def test_lag_one_descriptor_handles_zero_negative_and_multimode_cases():
    zero_mode = calculate_ruggedness_autocorrelation_analytical(
        _landscape(nx.cycle_graph(4), [1.0, 0.0, -1.0, 0.0]),
        lag_max=2,
    )
    negative_mode = calculate_ruggedness_autocorrelation_analytical(
        _landscape(nx.complete_graph(3), [1.0, -1.0, 0.0]),
        lag_max=2,
    )
    multimode = calculate_ruggedness_autocorrelation_analytical(
        _landscape(nx.path_graph(4), [0.0, 1.0, 4.0, 9.0]),
        lag_max=2,
    )

    np.testing.assert_allclose(zero_mode["autocorrelation"], [1.0, 0.0, 0.0])
    assert zero_mode["equivalent_single_exponential_length"] == 0.0
    assert zero_mode["elementary"] is True
    np.testing.assert_allclose(negative_mode["autocorrelation"], [1.0, -0.5, 0.25])
    assert negative_mode["equivalent_single_exponential_length"] == pytest.approx(
        1.0 / np.log(2.0)
    )
    assert negative_mode["correlation_length"] is None
    assert multimode["elementary"] is False
    assert multimode["correlation_length"] is None


def test_stochastic_periodic_positive_control_is_exact_and_lag_is_inclusive():
    landscape = _landscape(nx.path_graph(2), [1.0, -1.0])

    result = calculate_ruggedness_autocorrelation_stochastic(
        landscape,
        n_walks=7,
        steps=9,
        lag_max=3,
        seed=17,
    )

    np.testing.assert_array_equal(result["lags"], [0, 1, 2, 3])
    np.testing.assert_array_equal(result["autocorrelation"], [1, -1, 1, -1])
    np.testing.assert_array_equal(result["pair_counts"], [63, 56, 49, 42])


def test_long_run_weighted_monte_carlo_converges_to_exact_markov_result():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 1.0), (1, 2, 3.0), (2, 3, 2.0), (1, 3, 0.5)],
        weight="conductance",
    )
    landscape = _landscape(graph, [0.0, 2.0, 5.0, -1.0])
    exact = calculate_ruggedness_autocorrelation_analytical(
        landscape,
        lag_max=4,
        weight_key="conductance",
    )

    estimated = calculate_ruggedness_autocorrelation_stochastic(
        landscape,
        n_walks=2500,
        steps=80,
        lag_max=4,
        seed=20260812,
        weight_key="conductance",
    )

    np.testing.assert_allclose(
        estimated["autocorrelation"],
        exact["autocorrelation"],
        atol=0.025,
    )
    np.testing.assert_array_equal(
        estimated["pair_counts"],
        2500 * (80 - np.arange(5)),
    )


def test_precomputed_combinatorial_eigenpairs_are_ignored_compatibly():
    landscape = _landscape(nx.path_graph(4), [0.0, 1.0, 0.5, 1.5])
    eigenvalues, eigenvectors = eigenmode_decomposition(
        landscape.graph,
        matrix="laplacian",
        k=None,
    )

    base = calculate_ruggedness_autocorrelation_analytical(landscape)
    precomputed = calculate_ruggedness_autocorrelation_analytical(
        landscape,
        _eigenvalues=eigenvalues,
        _eigenvectors=eigenvectors,
    )

    np.testing.assert_allclose(base["autocorrelation"], precomputed["autocorrelation"])
    assert base["correlation_length"] is precomputed["correlation_length"] is None


def test_empty_disconnected_isolated_and_constant_inputs_are_rejected():
    empty = _landscape(nx.path_graph(2), [0.0, 1.0])
    empty.graph.clear()
    with pytest.raises(ValueError, match="no nodes"):
        calculate_ruggedness_autocorrelation_analytical(empty)

    disconnected = _landscape(
        nx.Graph([(0, 1), (2, 3)]),
        [0.0, 1.0, 2.0, 3.0],
    )
    with pytest.raises(ValueError, match="get_components"):
        calculate_ruggedness_autocorrelation_analytical(disconnected)

    isolated = _landscape(nx.empty_graph(1), [1.0])
    with pytest.raises(ValueError, match="non-trivial"):
        time_continuous_autocorrelation(isolated, times=[0.0])

    constant = _landscape(nx.path_graph(3), [2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="zero-stationary-variance"):
        calculate_ruggedness_autocorrelation_stochastic(constant)


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
def test_invalid_conductances_are_rejected(invalid):
    graph = nx.path_graph(3)
    nx.set_edge_attributes(graph, 1.0, "conductance")
    graph.edges[1, 2]["conductance"] = invalid
    landscape = _landscape(graph, [0.0, 1.0, 3.0])

    with pytest.raises(ValueError, match="invalid conductance"):
        calculate_ruggedness_autocorrelation_analytical(
            landscape,
            weight_key="conductance",
        )


def test_zero_conductance_components_are_rejected():
    graph = nx.cycle_graph(4)
    nx.set_edge_attributes(graph, 0.0, "conductance")
    graph.edges[0, 1]["conductance"] = 1.0
    graph.edges[2, 3]["conductance"] = 1.0
    landscape = _landscape(graph, [0.0, 1.0, 2.0, 4.0])

    with pytest.raises(ValueError, match="Positive-conductance"):
        time_continuous_autocorrelation(
            landscape,
            times=[0.0],
            weight_key="conductance",
        )


def test_directed_input_is_rejected_explicitly():
    landscape = _landscape(nx.path_graph(3), [0.0, 1.0, 3.0])
    landscape.graph = nx.DiGraph(landscape.graph)

    with pytest.raises(TypeError, match="undirected"):
        calculate_ruggedness_autocorrelation_analytical(landscape)


@pytest.mark.parametrize("times", [[], [-0.1], [np.nan], [np.inf]])
def test_continuous_time_domain_is_validated(times):
    landscape = _landscape(nx.path_graph(3), [0.0, 1.0, 3.0])

    with pytest.raises(ValueError):
        time_continuous_autocorrelation(landscape, times=times)


@pytest.mark.parametrize("times", [[True], [1 + 2j], "not-a-time"])
def test_continuous_time_type_is_validated(times):
    landscape = _landscape(nx.path_graph(3), [0.0, 1.0, 3.0])

    with pytest.raises(TypeError):
        time_continuous_autocorrelation(landscape, times=times)


def test_discrete_lag_and_stochastic_sampling_domains_are_validated():
    landscape = _landscape(nx.path_graph(3), [0.0, 1.0, 3.0])

    with pytest.raises(ValueError, match="non-negative"):
        calculate_ruggedness_autocorrelation_analytical(landscape, lag_max=-1)
    with pytest.raises(TypeError, match="non-negative"):
        calculate_ruggedness_autocorrelation_analytical(landscape, lag_max=1.5)
    with pytest.raises(ValueError, match="smaller than steps"):
        calculate_ruggedness_autocorrelation_stochastic(
            landscape,
            steps=4,
            lag_max=4,
        )
    with pytest.raises(ValueError, match="positive integer"):
        calculate_ruggedness_autocorrelation_stochastic(landscape, n_walks=0)
