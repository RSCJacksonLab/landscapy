"""Tests for explicit absorbing-boundary bottleneck analysis."""

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.analysis import (
    BoundaryModel,
    build_dirichlet_operator,
    calculate_local_bottleneck,
)
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BinarySequence


def _binary(bits, sequence_id):
    return BinarySequence(bits, sequence_id=sequence_id)


def test_boundary_model_validates_sequence_indices():
    assert BoundaryModel([2, np.int64(0)]).sequence_indices == (2, 0)

    with pytest.raises(ValueError, match="at least one"):
        BoundaryModel([])
    with pytest.raises(ValueError, match="duplicates"):
        BoundaryModel([1, 1])
    with pytest.raises(ValueError, match="non-negative"):
        BoundaryModel([-1])
    with pytest.raises(TypeError, match="integers"):
        BoundaryModel([True])
    with pytest.raises(ValueError, match="absorbing"):
        BoundaryModel([0], absorbing=False)


def test_build_routes_boundary_through_regular_graph_construction():
    sequences = [
        _binary([0, 0], "boundary-a"),
        _binary([0, 1], "boundary-b"),
        _binary([1, 0], "interior-a"),
        _binary([1, 1], "interior-b"),
    ]
    boundary = BoundaryModel([0, 1])

    landscape = FitnessLandscape.build(
        sequences,
        graph="hamming",
        boundary_model=boundary,
    )

    assert landscape.boundary_model is boundary
    assert landscape.boundary_sequence_indices == (0, 1)
    assert landscape.boundary_nodes == (0, 1)
    assert landscape.graph.graph["landscapy_boundary_model"]["absorbing"] is True
    assert all(
        landscape.graph.nodes[node]["is_boundary"] == (node in {0, 1})
        for node in landscape.graph
    )
    for node_a, node_b, data in landscape.graph.edges(data=True):
        assert data["crosses_boundary"] == (
            (node_a in {0, 1}) != (node_b in {0, 1})
        )


def test_boundary_binding_uses_duplicate_safe_row_to_node_mapping():
    first = _binary([0, 0], "first")
    duplicate = _binary([0, 0], "duplicate")
    interior = _binary([0, 1], "interior")
    sequences = [first, duplicate, interior]
    graph = nx.Graph()
    graph.add_node("interior-node", sequence=interior)
    graph.add_node("duplicate-node", sequence=duplicate)
    graph.add_node("first-node", sequence=first)
    graph.add_edges_from(
        [
            ("interior-node", "duplicate-node", {"weight": 1.0}),
            ("interior-node", "first-node", {"weight": 2.0}),
        ]
    )

    with pytest.warns(UserWarning, match="Duplicate sequences"):
        landscape = FitnessLandscape.build(
            sequences,
            graph=graph,
            boundary_model=BoundaryModel([1]),
        )

    assert landscape.sequence_index_to_node[1] == "duplicate-node"
    assert landscape.boundary_nodes == ("duplicate-node",)
    assert graph.nodes["duplicate-node"]["absorbing_boundary"] is True
    assert graph.nodes["first-node"]["absorbing_boundary"] is False


def test_dirichlet_operator_uses_only_interior_to_boundary_leakage():
    sequences = [
        _binary([0, 0], "interior-a"),
        _binary([0, 1], "interior-b"),
        _binary([1, 1], "boundary"),
    ]
    graph = nx.Graph()
    graph.add_node("a", sequence=sequences[0])
    graph.add_node("b", sequence=sequences[1])
    graph.add_node("z", sequence=sequences[2])
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "z", weight=3.0)
    landscape = FitnessLandscape.build(
        sequences,
        graph=graph,
        boundary_model=BoundaryModel([2]),
    )

    operator, nodes = build_dirichlet_operator(landscape, normalized=False)

    assert nodes == ["a", "b"]
    np.testing.assert_allclose(operator.toarray(), [[2.0, -2.0], [-2.0, 5.0]])
    result = calculate_local_bottleneck(landscape)
    assert result["boundary_nodes"] == ("z",)
    assert result["boundary_leakage"] == {"a": 0, "b": 3.0}
    assert result["boundary_crossing_edges"].to_dict("records") == [
        {"boundary_node": "z", "interior_node": "b", "weight": 3.0}
    ]


def test_boundary_rejects_indices_outside_constructed_landscape():
    sequence = _binary([0], "only")
    with pytest.raises(IndexError, match="outside the landscape"):
        FitnessLandscape.build(
            [sequence],
            graph="hamming",
            boundary_model=BoundaryModel([1]),
        )
