from inspect import signature

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.analysis.dirichlet_energy import (
    calculate_ruggedness_dirichlet_energy,
    local_dirichlet_energy_contribution,
)
from fitness_landscape.core.edge_schema import declare_edge_semantics
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence


def _landscape(
    layer_values,
    edges,
    *,
    labels=None,
    edge_data=None,
):
    sequence_count = len(next(iter(layer_values.values())))
    sequences = [
        BaseNumpySequence([index], sequence_id=f"s{index}")
        for index in range(sequence_count)
    ]
    labels = labels or [f"node-{index}" for index in range(sequence_count)]
    graph = nx.Graph()
    for index in reversed(range(sequence_count)):
        graph.add_node(labels[index], sequence=sequences[index])
    for edge_index, (first, second) in enumerate(edges):
        attributes = {} if edge_data is None else edge_data[edge_index]
        graph.add_edge(labels[first], labels[second], **attributes)
    layers = {
        name: NumericFitness.from_scalars(name, values)
        for name, values in layer_values.items()
    }
    return FitnessLandscape(sequences, graph, fitness_layers=layers)


def test_active_layer_drives_energy_independent_of_layer_name_and_node_order():
    landscape = _landscape(
        {
            "unused-assay": [8.0, 8.0, 8.0],
            "experiment-42": [0.0, 1.0, 3.0],
        },
        [(0, 1), (1, 2)],
        labels=["left", ("middle", 1), 99],
    )
    landscape.view("experiment-42")

    result = calculate_ruggedness_dirichlet_energy(landscape)

    assert landscape.active_layer_name == "experiment-42"
    assert result["weight_key"] is None
    assert result["weighted_laplacian"] is False
    assert result["global_dirichlet_energy"] == pytest.approx(5.0)
    assert result["total_dirichlet_energy"] == pytest.approx(5.0 / 3.0)


def test_weight_key_is_public_and_defaults_to_unweighted():
    parameter = signature(calculate_ruggedness_dirichlet_energy).parameters[
        "weight_key"
    ]

    assert parameter.default is None


def test_single_undirected_edge_has_no_extra_half_factor():
    landscape = _landscape(
        {"arbitrary-name": [0.0, 2.0]},
        [(0, 1)],
        edge_data=[{"conductance": 1.0}],
    )

    result = calculate_ruggedness_dirichlet_energy(
        landscape,
        edge_weight_bins=[(0.0, 2.0)],
    )
    local = local_dirichlet_energy_contribution(landscape)

    assert result["global_dirichlet_energy"] == pytest.approx(4.0)
    assert result["total_dirichlet_energy"] == pytest.approx(2.0)
    assert result["edge_weight_bins"]["(0.0, 2.0)_dirichlet_energy"] == pytest.approx(4.0)
    assert result["edge_weight_bins"]["(0.0, 2.0)_contribution"] == pytest.approx(1.0)
    assert list(local.values()) == pytest.approx([2.0, 2.0])
    assert sum(local.values()) == pytest.approx(result["global_dirichlet_energy"])


def test_explicit_weight_key_controls_global_bins_and_local_conservation():
    landscape = _landscape(
        {"assay": [0.0, 1.0, 3.0]},
        [(0, 1), (1, 2)],
        edge_data=[{"conductance": 2.0}, {"conductance": 0.5}],
    )

    unweighted = calculate_ruggedness_dirichlet_energy(landscape)
    weighted = calculate_ruggedness_dirichlet_energy(
        landscape,
        weight_key="conductance",
        edge_weight_bins=[(0.0, 0.75), (0.75, 2.5)],
    )
    local = local_dirichlet_energy_contribution(
        landscape,
        weight_key="conductance",
    )

    assert unweighted["global_dirichlet_energy"] == pytest.approx(5.0)
    assert weighted["global_dirichlet_energy"] == pytest.approx(4.0)
    assert weighted["total_dirichlet_energy"] == pytest.approx(4.0 / 3.0)
    assert weighted["weight_key"] == "conductance"
    assert weighted["weighted_laplacian"] is True

    node_order = list(landscape.graph.nodes())
    signal = landscape.get_node_signal(node_order)
    laplacian = nx.laplacian_matrix(
        landscape.graph,
        nodelist=node_order,
        weight="conductance",
    ).toarray()
    assert weighted["global_dirichlet_energy"] == pytest.approx(
        signal @ laplacian @ signal
    )

    bins = weighted["edge_weight_bins"]
    assert bins["(0.0, 0.75)_dirichlet_energy"] == pytest.approx(2.0)
    assert bins["(0.75, 2.5)_dirichlet_energy"] == pytest.approx(2.0)
    assert bins["(0.0, 0.75)_contribution"] == pytest.approx(0.5)
    assert bins["(0.75, 2.5)_contribution"] == pytest.approx(0.5)
    assert sum(value for key, value in bins.items() if key.endswith("_dirichlet_energy")) == pytest.approx(
        weighted["global_dirichlet_energy"]
    )

    node_for_index = landscape.sequence_index_to_node
    assert local[node_for_index[0]] == pytest.approx(1.0)
    assert local[node_for_index[1]] == pytest.approx(2.0)
    assert local[node_for_index[2]] == pytest.approx(1.0)
    assert sum(local.values()) == pytest.approx(weighted["global_dirichlet_energy"])


def test_legacy_weighted_selector_resolves_declared_conductance():
    landscape = _landscape(
        {"assay": [0.0, 2.0]},
        [(0, 1)],
        edge_data=[{"coupling": 0.25}],
    )
    declare_edge_semantics(
        landscape.graph,
        constructor="known-answer",
        conductance_key="coupling",
    )

    result = calculate_ruggedness_dirichlet_energy(
        landscape,
        weighted_laplacian=True,
    )

    assert result["weight_key"] == "coupling"
    assert result["global_dirichlet_energy"] == pytest.approx(1.0)


def test_requested_weight_key_must_exist_on_every_edge():
    landscape = _landscape(
        {"assay": [0.0, 1.0, 2.0]},
        [(0, 1), (1, 2)],
        edge_data=[{"coupling": 1.0}, {}],
    )

    with pytest.raises(ValueError, match="missing from 1 edge"):
        calculate_ruggedness_dirichlet_energy(
            landscape,
            weight_key="coupling",
        )
    with pytest.raises(ValueError, match="missing from 1 edge"):
        local_dirichlet_energy_contribution(
            landscape,
            weight_key="coupling",
        )


def test_disconnected_and_isolated_nodes_have_explicit_local_contributions():
    landscape = _landscape(
        {"assay": [0.0, 1.0, 10.0, 10.0, -50.0]},
        [(0, 1), (2, 3)],
    )

    result = calculate_ruggedness_dirichlet_energy(landscape)
    local = local_dirichlet_energy_contribution(landscape)
    node_for_index = landscape.sequence_index_to_node

    assert result["global_dirichlet_energy"] == pytest.approx(1.0)
    assert result["total_dirichlet_energy"] == pytest.approx(0.2)
    assert local[node_for_index[4]] == pytest.approx(0.0)
    assert sum(local.values()) == pytest.approx(1.0)


def test_zero_energy_has_zero_bin_contributions():
    landscape = _landscape(
        {"assay": [7.0, 7.0, 7.0]},
        [(0, 1), (1, 2)],
    )

    result = calculate_ruggedness_dirichlet_energy(
        landscape,
        edge_weight_bins=[(0.0, 2.0)],
    )
    local = local_dirichlet_energy_contribution(landscape)

    assert result["global_dirichlet_energy"] == 0.0
    assert result["total_dirichlet_energy"] == 0.0
    assert result["edge_weight_bins"]["(0.0, 2.0)_dirichlet_energy"] == 0.0
    assert result["edge_weight_bins"]["(0.0, 2.0)_contribution"] == 0.0
    assert set(local.values()) == {0.0}


def test_empty_landscape_returns_zero_without_requiring_an_active_layer():
    landscape = FitnessLandscape([], nx.Graph(), fitness_layers={})

    result = calculate_ruggedness_dirichlet_energy(
        landscape,
        edge_weight_bins=[(0.0, 2.0)],
    )

    assert result["global_dirichlet_energy"] == 0.0
    assert result["total_dirichlet_energy"] == 0.0
    assert result["edge_weight_bins"]["(0.0, 2.0)_dirichlet_energy"] == 0.0
    assert local_dirichlet_energy_contribution(landscape) == {}


def test_nonfinite_active_signal_is_rejected():
    landscape = _landscape(
        {"assay": [0.0, np.nan]},
        [(0, 1)],
    )

    with pytest.raises(ValueError, match="finite scalar values"):
        calculate_ruggedness_dirichlet_energy(landscape)


@pytest.mark.parametrize(
    "bins",
    [
        [(1.0, 1.0)],
        [(2.0, 1.0)],
        [(0.0, np.inf)],
        [(0.0, 1.0, 2.0)],
        [("low", 1.0)],
    ],
)
def test_invalid_edge_bins_fail_clearly(bins):
    landscape = _landscape({"assay": [0.0, 1.0]}, [(0, 1)])

    with pytest.raises(ValueError, match="bin|bounds"):
        calculate_ruggedness_dirichlet_energy(
            landscape,
            edge_weight_bins=bins,
        )


def test_weight_selector_conflicts_and_invalid_types_fail_clearly():
    landscape = _landscape(
        {"assay": [0.0, 1.0]},
        [(0, 1)],
        edge_data=[{"weight": 1.0}],
    )

    with pytest.raises(ValueError, match="weighted_laplacian=False"):
        calculate_ruggedness_dirichlet_energy(
            landscape,
            weighted_laplacian=False,
            weight_key="weight",
        )
    with pytest.raises(TypeError, match="boolean or None"):
        calculate_ruggedness_dirichlet_energy(
            landscape,
            weighted_laplacian="yes",
        )


def test_invalid_landscape_and_graph_types_fail_clearly():
    with pytest.raises(TypeError, match="FitnessLandscape"):
        calculate_ruggedness_dirichlet_energy(object())

    directed = _landscape({"assay": [0.0, 1.0]}, [(0, 1)])
    directed.graph = nx.DiGraph(directed.graph)
    with pytest.raises(TypeError, match="undirected"):
        calculate_ruggedness_dirichlet_energy(directed)

    multigraph = _landscape({"assay": [0.0, 1.0]}, [(0, 1)])
    multigraph.graph = nx.MultiGraph(multigraph.graph)
    with pytest.raises(TypeError, match="multigraph"):
        calculate_ruggedness_dirichlet_energy(multigraph)


def test_noniterable_edge_bins_fail_clearly():
    landscape = _landscape({"assay": [0.0, 1.0]}, [(0, 1)])

    with pytest.raises(TypeError, match="iterable"):
        calculate_ruggedness_dirichlet_energy(
            landscape,
            edge_weight_bins=1.0,
        )
