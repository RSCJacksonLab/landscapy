"""Known-answer tests for graph-smoothed polarity and edge tilt fields."""

import math

import networkx as nx
import numpy as np

from fitness_landscape import BaseNumpySequence, FitnessLandscape
from fitness_landscape.analysis import calculate_polarity_field
from fitness_landscape.core.edge_schema import declare_edge_semantics


def _landscape(size, edges=(), *, labels=None, reverse=False):
    sequences = [BaseNumpySequence([i], sequence_id=f"s{i}") for i in range(size)]
    labels = list(range(size)) if labels is None else labels
    graph = nx.Graph()
    for index in reversed(range(size)) if reverse else range(size):
        graph.add_node(labels[index], sequence=sequences[index])
    for left, right, weight in edges:
        graph.add_edge(labels[left], labels[right], weight=weight)
    declare_edge_semantics(
        graph, constructor="polarity_test", conductance_key="weight"
    )
    return FitnessLandscape.build(sequences, graph=graph)


def test_two_node_heat_field_and_tilt_have_analytic_values():
    result = calculate_polarity_field(
        _landscape(2, [(0, 1, 1.0)]),
        [0],
        [1],
        diffusion_time=0.5,
    )
    diagonal = (1.0 + math.exp(-1.0)) / 2.0
    expected = math.log((1.0 - diagonal) / diagonal)
    nodes = result["node_field"]
    np.testing.assert_allclose(nodes.polarity, [expected, -expected])
    assert nodes.support.tolist() == ["shared", "shared"]
    edge = result["edge_field"].iloc[0]
    assert edge.source_sequence_index == 0
    assert edge.target_sequence_index == 1
    np.testing.assert_allclose(edge.tilt, -2.0 * expected)
    assert bool(edge.tilt_defined)


def test_edge_orientation_uses_sequence_rows_not_graph_labels_or_order():
    landscape = _landscape(
        3,
        [(0, 2, 2.0), (1, 2, 1.0)],
        labels=["z", ("middle", 1), 7],
        reverse=True,
    )
    result = calculate_polarity_field(
        landscape, [0], [2], diffusion_time=1.0
    )
    edges = result["edge_field"]
    assert edges[["source_sequence_index", "target_sequence_index"]].to_records(
        index=False
    ).tolist() == [(0, 2), (1, 2)]
    assert edges[["source_node", "target_node"]].to_records(index=False).tolist() == [
        ("z", 7),
        (("middle", 1), 7),
    ]
    np.testing.assert_allclose(
        edges.tilt,
        result["node_field"].polarity.to_numpy()[edges.target_sequence_index]
        - result["node_field"].polarity.to_numpy()[edges.source_sequence_index],
    )


def test_one_sided_and_zero_support_remain_explicit_without_pseudocounts():
    result = calculate_polarity_field(
        _landscape(3), [0], [1], diffusion_time=4.0
    )
    nodes = result["node_field"]
    assert nodes.support.tolist() == [
        "reference_only",
        "observed_only",
        "zero_mass",
    ]
    assert nodes.polarity.iloc[0] == -np.inf
    assert nodes.polarity.iloc[1] == np.inf
    assert np.isnan(nodes.polarity.iloc[2])
    assert result["edge_field"].empty
    assert result["metadata"]["pseudocount"] is None
    assert result["metadata"]["beta"] is None


def test_nonfinite_endpoint_polarity_makes_tilt_explicitly_undefined():
    landscape = _landscape(3, [(0, 1, 0.0), (1, 2, 1.0)])
    result = calculate_polarity_field(
        landscape, [0], [1], diffusion_time=0.0
    )
    edges = result["edge_field"]
    assert not edges.tilt_defined.any()
    assert edges.tilt.isna().all()


def test_polarity_preserves_matched_family_mass_and_occupancy_metadata():
    result = calculate_polarity_field(
        _landscape(2, [(0, 1, 1.0)]),
        [0],
        [1],
        weighting="matched_family",
        family_labels=["same", "same"],
        diffusion_time=2.0,
    )
    assert result["metadata"]["weighting"] == "matched_family"
    assert result["metadata"]["diffusion_time"] == 2.0
    assert result["metadata"]["method"] == "graph_polarity_field"
    assert result["family_summary"].target_weight.tolist() == [1.0]
    np.testing.assert_allclose(
        result["node_field"][
            ["reference_probability", "observed_probability"]
        ].sum(),
        1.0,
    )
