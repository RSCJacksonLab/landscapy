import networkx as nx
import numpy as np
import pytest

import fitness_landscape.analysis.graph as graph_analysis
from fitness_landscape.analysis.graph import (
    graph_properties,
    graph_spectral_analysis,
    resistance_distance_matrix,
)
from fitness_landscape.core.fitness import CategoricalFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence


def _landscape(graph, *, categories=None):
    node_order = list(graph.nodes())
    sequences = [
        BaseNumpySequence([index], sequence_id=f"s{index}")
        for index in range(len(node_order))
    ]
    for node, sequence in zip(node_order, sequences, strict=True):
        graph.nodes[node]["sequence"] = sequence
    layers = {}
    if categories is not None:
        ordered_categories = list(dict.fromkeys(categories))
        layers["classes"] = CategoricalFitness(
            "classes",
            categories,
            categories=ordered_categories,
        )
    return FitnessLandscape(sequences, graph, fitness_layers=layers)


def test_path_cycle_and_weighted_tree_have_exact_resistances():
    path = resistance_distance_matrix(
        nx.path_graph(4),
        compute_resistance_matrix=True,
    )["resistance_mat"]
    np.testing.assert_allclose(
        path,
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        atol=1e-12,
    )

    cycle = resistance_distance_matrix(
        nx.cycle_graph(4),
        compute_resistance_matrix=True,
    )["resistance_mat"]
    np.testing.assert_allclose(
        cycle,
        [
            [0.0, 0.75, 1.0, 0.75],
            [0.75, 0.0, 0.75, 1.0],
            [1.0, 0.75, 0.0, 0.75],
            [0.75, 1.0, 0.75, 0.0],
        ],
        atol=1e-12,
    )

    tree = nx.Graph()
    tree.add_edge("root", "middle", conductance=2.0)
    tree.add_edge("middle", "leaf", conductance=0.25)
    weighted_tree = resistance_distance_matrix(
        tree,
        nodes=["root", "middle", "leaf"],
        weight_key="conductance",
        compute_resistance_matrix=True,
    )["resistance_mat"]
    np.testing.assert_allclose(
        weighted_tree,
        [
            [0.0, 0.5, 4.5],
            [0.5, 0.0, 4.0],
            [4.5, 4.0, 0.0],
        ],
        atol=1e-12,
    )


def test_landscape_components_are_used_and_cross_component_cost_is_infinite(
    monkeypatch,
):
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("c", "d")])
    landscape = _landscape(graph)
    original = landscape.get_components
    calls = []

    def tracked_get_components():
        calls.append(True)
        return original()

    monkeypatch.setattr(landscape, "get_components", tracked_get_components)
    result = resistance_distance_matrix(
        landscape,
        compute_resistance_matrix=True,
        jitter=1.0,
    )

    assert calls == [True]
    assert result["component_count"] == 2
    assert result["components"] == [["a", "b"], ["c", "d"]]
    assert result["component_ids"] == [0, 0, 1, 1]
    assert np.isinf(result["cross_component_resistance"])
    assert result["jitter_used"] is False
    expected = np.array(
        [
            [0.0, 1.0, np.inf, np.inf],
            [1.0, 0.0, np.inf, np.inf],
            [np.inf, np.inf, 0.0, 1.0],
            [np.inf, np.inf, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(result["resistance_mat"], expected, atol=1e-12)


def test_zero_conductance_edge_separates_electrical_components():
    graph = nx.path_graph(3)
    graph[0][1]["conductance"] = 1.0
    graph[1][2]["conductance"] = 0.0

    result = resistance_distance_matrix(
        graph,
        weight_key="conductance",
        weight_epsilon=10.0,
        compute_resistance_matrix=True,
    )

    assert result["component_count"] == 2
    np.testing.assert_allclose(result["resistance_mat"][:2, :2], [[0.0, 1 / 11], [1 / 11, 0.0]])
    assert np.all(np.isinf(result["resistance_mat"][:2, 2]))
    assert np.all(np.isinf(result["resistance_mat"][2, :2]))


@pytest.mark.parametrize("aggregation", ["expected_pairwise", "ot"])
def test_categories_in_separate_components_have_infinite_distance(aggregation):
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    landscape = _landscape(graph, categories=["A", "A", "B", "B"])

    result = resistance_distance_matrix(
        landscape,
        layers=["classes"],
        aggregation_function=aggregation,
    )

    distance = result["classes"]["distance_mat"]
    assert distance[0, 0] == 0.0
    assert distance[1, 1] == 0.0
    assert np.isinf(distance[0, 1])
    assert np.isinf(distance[1, 0])


def test_ot_is_finite_when_component_masses_match_but_expected_pairwise_is_not():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    landscape = _landscape(graph, categories=["A", "B", "A", "B"])

    ot = resistance_distance_matrix(
        landscape,
        layers=["classes"],
        aggregation_function="ot",
    )["classes"]["distance_mat"]
    expected = resistance_distance_matrix(
        landscape,
        layers=["classes"],
        aggregation_function="expected_pairwise",
    )["classes"]["distance_mat"]

    assert ot[0, 1] == pytest.approx(1.0)
    assert np.isinf(expected[0, 1])


def test_jitter_is_applied_only_after_component_factorization_failure(monkeypatch):
    real_splu = graph_analysis.splu
    calls = []

    def fail_once(matrix):
        calls.append(matrix)
        if len(calls) == 1:
            raise RuntimeError("forced factorization failure")
        return real_splu(matrix)

    monkeypatch.setattr(graph_analysis, "splu", fail_once)
    result = resistance_distance_matrix(
        nx.path_graph(4),
        sparse_threshold=0,
        jitter=1e-8,
        compute_resistance_matrix=True,
    )

    assert len(calls) == 2
    assert result["jitter_used"] is True
    assert result["jittered_components"] == [0]
    assert np.all(np.isfinite(result["resistance_mat"]))


def test_empty_graph_outputs_have_stable_documented_schemas():
    landscape = FitnessLandscape([], nx.Graph(), fitness_layers={})

    properties = graph_properties(landscape)
    spectral = graph_spectral_analysis(landscape)
    resistance = resistance_distance_matrix(
        landscape,
        compute_resistance_matrix=True,
    )

    assert properties["components"] == {"count": 0, "largest_size": 0, "sizes": []}
    assert np.isnan(properties["degree"]["mean"])
    assert np.isnan(properties["clustering"])
    assert np.isnan(properties["path_length"])
    assert properties["density"] == 0.0

    assert spectral["eigenvalues"].shape == (0,)
    assert spectral["node_centralities"].shape == (0, 0)
    assert spectral["spectral_density"]["histogram"].shape == (0,)
    assert "spectral_gap" not in spectral

    assert resistance["component_count"] == 0
    assert resistance["components"] == []
    assert resistance["component_ids"] == []
    assert resistance["resistance_mat"].shape == (0, 0)
    assert resistance["jitter_used"] is False


def test_singleton_graph_outputs_are_finite_and_zero_where_defined():
    graph = nx.Graph()
    graph.add_node("only")
    landscape = _landscape(graph)

    properties = graph_properties(landscape)
    spectral = graph_spectral_analysis(landscape)
    resistance = resistance_distance_matrix(
        landscape,
        compute_resistance_matrix=True,
    )

    assert properties["degree"] == {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}
    assert properties["clustering"] == 0.0
    assert properties["path_length"] == 0.0
    assert properties["components"] == {"count": 1, "largest_size": 1, "sizes": [1]}
    assert properties["density"] == 0.0

    np.testing.assert_allclose(spectral["eigenvalues"], [0.0], atol=1e-12)
    np.testing.assert_allclose(spectral["participation_ratios"], [1.0], atol=1e-12)
    assert "spectral_gap" not in spectral

    assert resistance["component_count"] == 1
    assert resistance["components"] == [["only"]]
    np.testing.assert_array_equal(resistance["resistance_mat"], [[0.0]])
    assert resistance["jitter_used"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"jitter": -1.0}, "jitter"),
        ({"weight_epsilon": np.nan}, "weight_epsilon"),
        ({"nodes": [0, 0]}, "duplicates"),
        ({"nodes": [99]}, "not in the graph"),
    ],
)
def test_resistance_validation_errors_are_clear(kwargs, message):
    with pytest.raises((ValueError, KeyError), match=message):
        resistance_distance_matrix(nx.path_graph(2), **kwargs)


def test_resistance_rejects_multigraphs_instead_of_collapsing_parallel_edges():
    graph = nx.MultiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)

    with pytest.raises(TypeError, match="simple undirected"):
        resistance_distance_matrix(graph)
