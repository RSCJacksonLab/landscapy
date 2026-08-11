"""Regression tests for analysis independence from graph-node labels."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.analysis.adaptive_walk import (
    adaptive_walk_stochastic,
    analyze_path_accessibility,
    calculate_basin_of_attraction_greedy,
    calculate_basin_of_attraction_stochastic,
    find_greedy_accessible_paths,
    neutral_network_analysis,
)
from fitness_landscape.analysis.diffusion_scale import (
    compute_ruggedness_variance_energy,
)
from fitness_landscape.analysis.dirichlet_energy import (
    calculate_ruggedness_dirichlet_energy,
    local_dirichlet_energy_contribution,
)
from fitness_landscape.analysis.graph import (
    calculate_ruggedness_local_optima,
    graph_properties,
    graph_spectral_analysis,
    resistance_distance_matrix,
)
from fitness_landscape.analysis.random_walk import (
    calculate_ruggedness_autocorrelation_analytical,
    calculate_ruggedness_autocorrelation_stochastic,
    category_boundary_crossing_times,
)
from fitness_landscape.core.fitness import CategoricalFitness, NumericFitness
from fitness_landscape.core.edge_schema import declare_edge_semantics
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.transforms.graph_fourier import graph_fourier_transform


NODE_CASES = [
    pytest.param([0, 1, 2, 3], [0, 1, 2, 3], id="integer-contiguous"),
    pytest.param([10, 30, 20, 99], [0, 1, 2, 3], id="integer-noncontiguous"),
    pytest.param(["start", "middle-a", "middle-b", "peak"], [0, 1, 2, 3], id="string"),
    pytest.param(
        [("node", 0), ("node", 1), ("node", 2), ("node", 3)],
        [0, 1, 2, 3],
        id="tuple",
    ),
    pytest.param(
        ["start", "middle-a", "middle-b", "peak"],
        [2, 0, 3, 1],
        id="reordered",
    ),
]


def _landscape(node_by_index, insertion_indices):
    sequences = [
        BaseNumpySequence([index], sequence_id=f"s{index}") for index in range(4)
    ]
    graph = nx.Graph()
    for sequence_index in insertion_indices:
        graph.add_node(
            node_by_index[sequence_index],
            sequence=sequences[sequence_index],
        )
    for sequence_index in range(3):
        graph.add_edge(
            node_by_index[sequence_index],
            node_by_index[sequence_index + 1],
            weight=1.0,
        )
    declare_edge_semantics(
        graph,
        constructor="node-label-regression",
        conductance_key="weight",
    )

    numeric = NumericFitness.from_scalars("fitness", [0.0, 1.0, 2.0, 3.0])
    categorical = CategoricalFitness.from_values(
        "category",
        ["low", "low", "high", "high"],
        categories=["low", "high"],
    )
    return FitnessLandscape(
        sequences,
        graph,
        fitness_layers={"fitness": numeric, "category": categorical},
    )


@pytest.mark.parametrize("node_by_index,insertion_indices", NODE_CASES)
def test_landscape_node_accessors_and_signal_order(node_by_index, insertion_indices):
    landscape = _landscape(node_by_index, insertion_indices)

    expected_node_to_index = {
        node_by_index[index]: index for index in insertion_indices
    }
    assert landscape.node_to_sequence_index == expected_node_to_index
    assert landscape.sequence_index_to_node == {
        index: node_by_index[index] for index in insertion_indices
    }
    for index, node in enumerate(node_by_index):
        assert landscape.sequence_index_for_node(node) == index
        assert landscape.node_for_sequence_index(index) == node
        assert landscape.sequence_for_node(node) is landscape.sequences[index]

    graph_order = list(landscape.graph.nodes())
    expected = np.array(
        [float(landscape.sequence_index_for_node(node)) for node in graph_order]
    )
    np.testing.assert_array_equal(landscape.get_node_signal(), expected)

    with pytest.raises(KeyError, match="not part"):
        landscape.sequence_index_for_node("missing")
    with pytest.raises(IndexError, match="outside"):
        landscape.node_for_sequence_index(99)


@pytest.mark.parametrize("node_by_index,insertion_indices", NODE_CASES)
def test_paths_optima_basins_and_neutral_networks_preserve_labels(
    node_by_index,
    insertion_indices,
):
    landscape = _landscape(node_by_index, insertion_indices)
    start = landscape.sequences[0]
    peak = landscape.sequences[3]

    paths = find_greedy_accessible_paths(landscape, start, peak)
    assert paths["path_count"] == 1
    assert paths["paths"][0]["nodes"] == node_by_index
    assert paths["paths"][0]["indices"] == [0, 1, 2, 3]

    accessibility = analyze_path_accessibility(landscape)
    assert accessibility["local_minima"] == [node_by_index[0]]
    assert accessibility["local_maxima"] == [node_by_index[3]]
    assert accessibility["local_minima_indices"] == [0]
    assert accessibility["local_maxima_indices"] == [3]

    optima = calculate_ruggedness_local_optima(landscape)
    assert optima["local_optima"] == [node_by_index[3]]
    assert optima["local_optima_indices"] == [3]

    basin = calculate_basin_of_attraction_greedy(landscape, peak)
    assert set(basin["basin"]) == set(node_by_index)
    assert set(basin["basin_indices"]) == {0, 1, 2, 3}
    assert basin["optimum_node"] == node_by_index[3]
    assert basin["optimum_index"] == 3

    stochastic_basin = calculate_basin_of_attraction_stochastic(
        landscape,
        peak,
        n_simulations=1,
        max_steps=8,
        acceptance_threshold=0.0,
    )
    assert set(stochastic_basin["basin_probabilities"]) == set(node_by_index)
    assert set(stochastic_basin["basin_probabilities_by_index"]) == {0, 1, 2, 3}
    assert stochastic_basin["optimum_node"] == node_by_index[3]
    assert stochastic_basin["optimum_index"] == 3

    walk = adaptive_walk_stochastic(
        landscape,
        start_sequence=start,
        strategy="greedy",
    )
    assert walk["walk_nodes"] == node_by_index
    assert walk["walk_indices"] == [0, 1, 2, 3]

    neutral = neutral_network_analysis(landscape, threshold=0.0)
    returned_nodes = {
        node for network in neutral["networks"] for node in network["nodes"]
    }
    assert returned_nodes == set(node_by_index)
    assert all(len(network["sequence_indices"]) == network["size"] for network in neutral["networks"])


@pytest.mark.parametrize("node_by_index,insertion_indices", NODE_CASES)
def test_energy_spectral_and_walk_analyses_use_graph_node_order(
    node_by_index,
    insertion_indices,
):
    landscape = _landscape(node_by_index, insertion_indices)
    landscape.view("fitness")

    global_energy = calculate_ruggedness_dirichlet_energy(landscape)
    assert global_energy["total_dirichlet_energy"] == pytest.approx(0.75)

    local_energy = local_dirichlet_energy_contribution(landscape)
    assert set(local_energy) == set(node_by_index)
    assert local_energy[node_by_index[0]] == pytest.approx(0.5)
    assert local_energy[node_by_index[1]] == pytest.approx(1.0)
    assert local_energy[node_by_index[2]] == pytest.approx(1.0)
    assert local_energy[node_by_index[3]] == pytest.approx(0.5)

    analytical = calculate_ruggedness_autocorrelation_analytical(landscape)
    assert analytical["autocorrelation"][0] == pytest.approx(1.0)

    stochastic = calculate_ruggedness_autocorrelation_stochastic(
        landscape,
        n_walks=5,
        steps=8,
        lag_max=3,
        seed=12,
    )
    assert stochastic["autocorrelation"][0] == pytest.approx(1.0)

    eigenvectors, _, coefficients = graph_fourier_transform(landscape)
    np.testing.assert_allclose(
        eigenvectors @ coefficients,
        landscape.get_node_signal(),
        atol=1e-10,
    )

    spectral = graph_spectral_analysis(landscape)
    assert spectral["node_order"] == list(landscape.graph.nodes())

    variance = compute_ruggedness_variance_energy(landscape, t=1.0)
    assert set(variance["expected_local_energy_by_node"]) == set(node_by_index)
    assert variance["node_order"] == list(landscape.graph.nodes())


@pytest.mark.parametrize("node_by_index,insertion_indices", NODE_CASES)
def test_graph_unwrapping_and_categorical_walks_accept_all_node_labels(
    node_by_index,
    insertion_indices,
):
    landscape = _landscape(node_by_index, insertion_indices)

    assert graph_properties(landscape) == graph_properties(landscape.graph)

    resistance = resistance_distance_matrix(
        landscape.graph,
        sample_nodes=2,
        sample_seed=3,
    )
    assert len(resistance["sampled_nodes"]) == 2
    assert set(resistance["sampled_nodes"]) <= set(node_by_index)

    crossing = category_boundary_crossing_times(
        landscape,
        layer="category",
        n_walks=5,
        max_steps=8,
        seed=4,
        weight_key="weight",
    )
    assert crossing["mean_crossing_time"].shape == (2, 2)
    assert crossing["hit_counts"].shape == (2, 2)
