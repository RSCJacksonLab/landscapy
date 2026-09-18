"""Known-answer and input-contract tests for reference-relative occupancy."""

import copy
import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import (
    BoundaryModel,
    calculate_graph_occupancy,
    calculate_sequence_occupancy,
)
from fitness_landscape.core.edge_schema import declare_edge_semantics


def _landscape(size, edges=(), *, labels=None, reverse=False, boundary=None):
    sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(size)]
    labels = list(range(size)) if labels is None else labels
    graph = nx.Graph()
    for i in reversed(range(size)) if reverse else range(size):
        graph.add_node(labels[i], sequence=sequences[i])
    for u, v, weight in edges:
        graph.add_edge(labels[u], labels[v], weight=weight)
    declare_edge_semantics(
        graph, constructor="occupancy_test", conductance_key="weight"
    )
    return FitnessLandscape.build(sequences, graph=graph, boundary_model=boundary)


def _cliques(count, width=3):
    return _landscape(
        count * width,
        [
            (base + i, base + j, 1.0)
            for base in range(0, count * width, width)
            for i in range(width)
            for j in range(i + 1, width)
        ],
    )


def _probabilities(result):
    return result["distribution"][
        ["reference_probability", "observed_probability"]
    ].to_numpy()


def test_count_restriction_to_half_the_reference_has_log_two_kl():
    landscape = _cliques(4)
    result = calculate_sequence_occupancy(landscape, [0, 3, 6, 9], [1, 4])
    table = result["distribution"]
    np.testing.assert_allclose(table.reference_probability, [0.25] * 4)
    np.testing.assert_allclose(table.observed_probability, [0.5, 0.5, 0, 0])
    np.testing.assert_array_equal(table.reference_count, [1, 1, 1, 1])
    np.testing.assert_array_equal(table.observed_count, [1, 1, 0, 0])
    np.testing.assert_array_equal(table.node_count, [3] * 4)
    np.testing.assert_allclose(table.enrichment, [2, 2, 0, 0])
    assert result["divergence"]["kl_observed_reference"] == pytest.approx(math.log(2))
    assert result["divergence"]["kl_reference_observed"] == np.inf
    assert result["divergence"]["total_variation"] == pytest.approx(0.5)
    assert result["family_summary"].empty


def test_different_sample_counts_can_have_identical_community_occupancy():
    result = calculate_sequence_occupancy(_cliques(2), [0, 1, 3, 4], [2, 5])
    np.testing.assert_allclose(_probabilities(result), [[0.5, 0.5], [0.5, 0.5]])
    assert all(value == 0 for value in result["divergence"].values())


def test_matched_families_remove_family_abundance_difference():
    landscape = _cliques(2)
    global_result = calculate_sequence_occupancy(landscape, [0, 1, 3], [2, 4, 5])
    result = calculate_sequence_occupancy(
        landscape,
        [0, 1, 3],
        [2, 4, 5],
        weighting="matched_family",
        family_labels=["x"] * 3 + ["y"] * 3,
    )
    assert global_result["divergence"]["kl_observed_reference"] == pytest.approx(
        math.log(2) / 3
    )
    assert result["divergence"]["kl_observed_reference"] == 0
    np.testing.assert_allclose(_probabilities(result), 0.5)
    assert result["family_summary"].to_dict("records") == [
        {
            "family": "x",
            "reference_count": 2,
            "observed_count": 1,
            "target_weight": 0.5,
        },
        {
            "family": "y",
            "reference_count": 1,
            "observed_count": 2,
            "target_weight": 0.5,
        },
    ]


def test_matched_family_mass_preserves_different_occupancy_within_families():
    result = calculate_sequence_occupancy(
        _cliques(2),
        [0, 2, 3],
        [1, 4, 5],
        weighting="matched_family",
        family_labels=["x", "x", "y", "x", "y", "y"],
    )
    np.testing.assert_allclose(_probabilities(result), [[0.75, 0.5], [0.25, 0.5]])
    assert result["divergence"]["kl_observed_reference"] == pytest.approx(
        0.5 * math.log(4 / 3)
    )


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_custom_family_weights_are_shared_and_normalized(method):
    result = method(
        _cliques(2),
        [0, 1, 3],
        [2, 4, 5],
        weighting="matched_family",
        family_labels=pd.Series(["x"] * 3 + ["y"] * 3, index=range(10, 16)),
        family_weights={"x": 3e307, "y": 1e307},
    )
    weights = result["sequence_weights"]
    np.testing.assert_allclose(weights.reference_weight, [0.375, 0.375, 0, 0.25, 0, 0])
    np.testing.assert_allclose(weights.observed_weight, [0, 0, 0.75, 0, 0.125, 0.125])
    np.testing.assert_allclose(result["family_summary"].target_weight, [0.75, 0.25])
    np.testing.assert_allclose(_probabilities(result).sum(axis=0), 1)


def test_louvain_parameters_reproducibility_and_resolution():
    landscape = _cliques(2)
    first = calculate_sequence_occupancy(
        landscape, [0, 3], [1, 4], seed=17, threshold=1e-6
    )
    second = calculate_sequence_occupancy(
        landscape, [0, 3], [1, 4], seed=17, threshold=1e-6
    )
    pd.testing.assert_frame_equal(first["sequence_weights"], second["sequence_weights"])
    fine = calculate_sequence_occupancy(landscape, [0, 3], [1, 4], resolution=20)
    assert first["metadata"]["community_count"] == 2
    assert fine["metadata"]["community_count"] == 6
    assert first["metadata"]["seed"] == 17
    assert first["metadata"]["threshold"] == 1e-6


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_sequence_indices_are_not_graph_labels_or_insertion_order(method):
    landscape = _landscape(
        4,
        [(0, 1, 1), (2, 3, 1)],
        labels=["a", ("b", 2), 77, "z"],
        reverse=True,
    )
    result = method(landscape, [0, 2], [1, 3])
    table = result["sequence_weights"]
    assert table.node.tolist() == ["a", ("b", 2), 77, "z"]
    np.testing.assert_allclose(table.reference_weight, [0.5, 0, 0.5, 0])
    np.testing.assert_allclose(table.observed_weight, [0, 0.5, 0, 0.5])
    assert result["divergence"]["kl_observed_reference"] == 0


def test_duplicate_sequence_strings_keep_separate_row_mass():
    sequences = [BaseNumpySequence([0], sequence_id=f"s{i}") for i in range(3)]
    graph = nx.Graph()
    for i in [2, 0, 1]:
        graph.add_node(f"n{i}", sequence=sequences[i])
    with pytest.warns(UserWarning, match="Duplicate sequences"):
        landscape = FitnessLandscape.build(sequences, graph=graph)
    result = calculate_sequence_occupancy(landscape, [0, 1], [2])
    np.testing.assert_allclose(
        result["sequence_weights"].reference_weight, [0.5, 0.5, 0]
    )
    assert result["divergence"]["kl_observed_reference"] == np.inf


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_identical_panels_and_unused_regions(method):
    result = method(_landscape(3), [0, 1], [0, 1])
    assert all(value == 0 for value in result["divergence"].values())
    assert np.isnan(result["distribution"].enrichment.iloc[2])


@pytest.mark.parametrize("edges", [[], [(0, 1, 0.0)]])
@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_disconnected_or_zero_weight_support_is_not_pseudocounted(edges, method):
    result = method(_landscape(3, edges), [0], [1])
    assert result["divergence"]["kl_observed_reference"] == np.inf
    assert result["divergence"]["jensen_shannon"] == pytest.approx(math.log(2))
    assert result["divergence"]["total_variation"] == 1
    assert result["divergence"]["observed_mass_without_reference"] == 1
    np.testing.assert_allclose(_probabilities(result), [[1, 0], [0, 1], [0, 0]])


def test_random_walk_two_node_known_answer_and_laziness():
    landscape = _landscape(2, [(0, 1, 1)])
    for steps in [0, 1, 2, 7]:
        result = calculate_graph_occupancy(
            landscape, [0], [1], steps=steps, laziness=0.75
        )
        diagonal = (1 + 0.5**steps) / 2
        np.testing.assert_allclose(
            _probabilities(result), [[diagonal, 1 - diagonal], [1 - diagonal, diagonal]]
        )
    default = calculate_graph_occupancy(landscape, [0], [1])
    np.testing.assert_allclose(_probabilities(default), 0.5)
    assert default["metadata"]["steps"] == 1
    assert default["metadata"]["laziness"] == 0.5
    held = calculate_graph_occupancy(landscape, [0], [1], steps=10, laziness=1)
    np.testing.assert_allclose(_probabilities(held), np.eye(2))


@pytest.mark.parametrize("time", [0, 0.01, 0.5, 2, 30])
def test_heat_two_node_analytic_solution(time):
    result = calculate_graph_occupancy(
        _landscape(2, [(0, 1, 1)]),
        [0],
        [1],
        kernel="heat",
        diffusion_time=time,
    )
    diagonal = (1 + np.exp(-2 * time)) / 2
    np.testing.assert_allclose(
        _probabilities(result),
        [[diagonal, 1 - diagonal], [1 - diagonal, diagonal]],
        atol=1e-14,
    )
    assert result["metadata"]["diffusion_time"] == time
    assert result["metadata"]["steps"] is None


@pytest.mark.parametrize("kernel", ["random_walk", "heat"])
def test_weighted_irregular_graph_matches_dense_markov_solution(kernel):
    landscape = _landscape(4, [(0, 0, 3), (0, 1, 1), (1, 2, 2)])
    # The self-loop is counted once; node 3 holds its mass.
    transition = np.array(
        [[0.75, 0.25, 0, 0], [1 / 3, 0, 2 / 3, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    initial = np.array([[0.5, 0], [0, 1], [0, 0], [0.5, 0]])
    if kernel == "heat":
        kwargs = {"diffusion_time": 1.7}
        expected = expm(1.7 * (transition.T - np.eye(4))) @ initial
    else:
        kwargs = {"steps": 3, "laziness": 0.2}
        expected = (
            np.linalg.matrix_power(0.2 * np.eye(4) + 0.8 * transition.T, 3) @ initial
        )
    result = calculate_graph_occupancy(landscape, [0, 3], [1], kernel=kernel, **kwargs)
    np.testing.assert_allclose(_probabilities(result), expected, atol=1e-14)
    assert result["metadata"]["zero_degree_node_count"] == 1
    np.testing.assert_allclose(_probabilities(result).sum(axis=0), 1)


@pytest.mark.parametrize("kernel", ["random_walk", "heat"])
def test_shared_smoothing_contracts_divergence(kernel):
    landscape = _landscape(5, [(i, i + 1, 1) for i in range(4)])
    results = [
        calculate_graph_occupancy(
            landscape,
            [0, 1],
            [3, 4],
            kernel=kernel,
            **(
                {"steps": scale}
                if kernel == "random_walk"
                else {"diffusion_time": scale}
            ),
        )
        for scale in [1, 3, 10, 40]
    ]
    for metric in ["jensen_shannon", "total_variation", "kl_observed_reference"]:
        values = [result["divergence"][metric] for result in results]
        assert all(
            later <= earlier + 1e-12 for earlier, later in zip(values, values[1:])
        )


def test_heat_does_not_exchange_mass_between_components():
    result = calculate_graph_occupancy(
        _cliques(2), [0], [3], kernel="heat", diffusion_time=50
    )
    np.testing.assert_allclose(
        _probabilities(result), [[1 / 3, 0]] * 3 + [[0, 1 / 3]] * 3, atol=1e-13
    )
    assert result["divergence"]["kl_observed_reference"] == np.inf


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_absorbing_boundary_and_landscape_annotations_are_unchanged(method):
    landscape = _landscape(2, [(0, 1, 1)], boundary=BoundaryModel([0]))
    graph_before = copy.deepcopy(landscape.graph)
    result = method(landscape, [0], [1])
    assert result["divergence"]["kl_observed_reference"] == 0
    assert result["metadata"]["absorbing_boundary_applied"] is False
    assert nx.utils.graphs_equal(landscape.graph, graph_before)
    assert landscape.boundary_sequence_indices == (0,)
    assert not landscape.annotation_layers


def test_sparse_heat_does_not_materialize_dense_kernel(monkeypatch):
    import scipy.sparse as sp

    landscape = _landscape(2500, [(i, i + 1, 1) for i in range(2499)])

    def fail_dense(*args, **kwargs):
        raise AssertionError("dense kernel allocation")

    monkeypatch.setattr(sp.csr_matrix, "toarray", fail_dense)
    monkeypatch.setattr(sp.csc_matrix, "toarray", fail_dense)
    result = calculate_graph_occupancy(
        landscape, [0], [2499], kernel="heat", diffusion_time=1
    )
    assert len(result["distribution"]) == 2500
    np.testing.assert_allclose(_probabilities(result).sum(axis=0), 1)


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
@pytest.mark.parametrize(
    "indices,error",
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([True], TypeError),
        ([0.0], TypeError),
        ([-1], IndexError),
        ([3], IndexError),
    ],
)
def test_invalid_sequence_selections(method, indices, error):
    with pytest.raises(error):
        method(_landscape(3), indices, [1])
    with pytest.raises(error):
        method(_landscape(3), [1], indices)


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"weighting": "unknown"}, "weighting"),
        ({"family_labels": ["x", "x"]}, "Family parameters"),
        ({"family_weights": {"x": 1}}, "Family parameters"),
        ({"weighting": "matched_family"}, "one label"),
        ({"weighting": "matched_family", "family_labels": ["x"]}, "one label"),
        ({"weighting": "matched_family", "family_labels": ["x", "y"]}, "both panels"),
        ({"weighting": "matched_family", "family_labels": [None, "x"]}, "missing"),
        ({"weighting": "matched_family", "family_labels": [np.nan, "x"]}, "missing"),
        ({"weighting": "matched_family", "family_labels": [pd.NA, "x"]}, "missing"),
    ],
)
def test_invalid_weighting_and_family_support(method, kwargs, match):
    with pytest.raises(ValueError, match=match):
        method(_landscape(2), [0], [1], **kwargs)


@pytest.mark.parametrize(
    "weights,error",
    [
        ({"x": 0}, ValueError),
        ({"x": -1}, ValueError),
        ({"x": np.inf}, ValueError),
        ({"x": np.nan}, ValueError),
        ({"y": 1}, ValueError),
        ({"x": True}, TypeError),
        ([1], TypeError),
    ],
)
def test_invalid_family_weights(weights, error):
    with pytest.raises(error):
        calculate_sequence_occupancy(
            _landscape(2),
            [0],
            [1],
            weighting="matched_family",
            family_labels=["x", "x"],
            family_weights=weights,
        )


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
@pytest.mark.parametrize("weight", [-1, np.inf, np.nan])
def test_invalid_conductance(method, weight):
    with pytest.raises(ValueError, match="conductance"):
        method(_landscape(2, [(0, 1, weight)]), [0], [1])


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_edge_schema_auto_and_explicit_unweighted_geometry(method):
    landscape = _landscape(2, [(0, 1, 0)])
    result = method(landscape, [0], [1], weight_key=None)
    assert result["divergence"]["kl_observed_reference"] == 0
    landscape.graph.graph.clear()
    with pytest.raises(ValueError, match="undeclared"):
        method(landscape, [0], [1])
    assert (
        method(landscape, [0], [1], weight_key="weight")["metadata"]["weight_key"]
        == "weight"
    )
    with pytest.raises(ValueError, match="missing"):
        method(landscape, [0], [1], weight_key="not_present")


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"kernel": "unknown"}, ValueError),
        ({"steps": -1}, ValueError),
        ({"steps": True}, TypeError),
        ({"steps": 1.5}, TypeError),
        ({"laziness": -0.1}, ValueError),
        ({"laziness": 1.1}, ValueError),
        ({"laziness": np.nan}, ValueError),
        ({"diffusion_time": 1}, ValueError),
        ({"kernel": "heat", "steps": 1}, ValueError),
        ({"kernel": "heat", "laziness": 0.5}, ValueError),
        ({"kernel": "heat", "diffusion_time": -1}, ValueError),
        ({"kernel": "heat", "diffusion_time": np.inf}, ValueError),
        ({"kernel": "heat", "diffusion_time": True}, TypeError),
    ],
)
def test_invalid_kernel_parameters(kwargs, error):
    with pytest.raises(error):
        calculate_graph_occupancy(_landscape(2), [0], [1], **kwargs)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"resolution": 0}, ValueError),
        ({"resolution": np.inf}, ValueError),
        ({"threshold": -1}, ValueError),
        ({"seed": True}, TypeError),
        ({"seed": 1.2}, TypeError),
    ],
)
def test_invalid_louvain_parameters(kwargs, error):
    with pytest.raises(error):
        calculate_sequence_occupancy(_landscape(2), [0], [1], **kwargs)


@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_invalid_landscape_and_stale_mapping(method):
    with pytest.raises(TypeError, match="FitnessLandscape"):
        method(nx.path_graph(2), [0], [1])
    landscape = _landscape(2)
    landscape.graph.add_node("extra")
    with pytest.raises(ValueError, match="one graph node"):
        method(landscape, [0], [1])
    landscape.graph.remove_node(1)
    with pytest.raises(ValueError, match="no longer match"):
        method(landscape, [0], [1])


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiGraph])
@pytest.mark.parametrize(
    "method", [calculate_sequence_occupancy, calculate_graph_occupancy]
)
def test_directed_and_multigraphs_are_explicitly_rejected(graph_type, method):
    landscape = _landscape(2)
    landscape.graph = graph_type(landscape.graph)
    with pytest.raises(TypeError, match="simple undirected"):
        method(landscape, [0], [1])


@pytest.mark.parametrize("kernel", ["random_walk", "heat"])
def test_smoothing_is_invariant_to_subnormal_conductance_scale(kernel):
    ordinary = calculate_graph_occupancy(
        _landscape(2, [(0, 1, 1)]),
        [0],
        [1],
        kernel=kernel,
    )
    tiny = calculate_graph_occupancy(
        _landscape(2, [(0, 1, 1e-320)]),
        [0],
        [1],
        kernel=kernel,
    )
    np.testing.assert_allclose(_probabilities(tiny), _probabilities(ordinary))


def test_extreme_family_weights_do_not_silently_drop_a_family():
    with pytest.raises(ValueError, match="underflow"):
        calculate_sequence_occupancy(
            _cliques(2),
            [0, 3],
            [1, 4],
            weighting="matched_family",
            family_labels=["x"] * 3 + ["y"] * 3,
            family_weights={"x": 1e308, "y": 1e-308},
        )


def test_family_labels_can_be_tuples_but_not_unhashable_lists():
    result = calculate_sequence_occupancy(
        _landscape(2),
        [0],
        [1],
        weighting="matched_family",
        family_labels=[("x", 1), ("x", 1)],
        seed=None,
    )
    assert result["family_summary"].family.tolist() == [("x", 1)]
    with pytest.raises(TypeError, match="hashable"):
        calculate_sequence_occupancy(
            _landscape(2),
            [0],
            [1],
            weighting="matched_family",
            family_labels=[["x"], ["x"]],
        )


def test_weighted_degree_overflow_is_rejected():
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="overflow"):
            calculate_graph_occupancy(
                _landscape(3, [(0, 1, 1e308), (0, 2, 1e308)]),
                [0],
                [1],
            )


@pytest.mark.parametrize("bad_mass", [np.nan, -0.1, 0.1])
def test_failed_heat_conservation_is_not_silently_normalized(monkeypatch, bad_mass):
    import fitness_landscape.analysis.bottleneck as bottleneck

    monkeypatch.setattr(
        bottleneck.spla,
        "expm_multiply",
        lambda *args, **kwargs: np.full((2, 2), bad_mass),
    )
    with pytest.raises(RuntimeError, match="conservation"):
        calculate_graph_occupancy(_landscape(2, [(0, 1, 1)]), [0], [1], kernel="heat")
