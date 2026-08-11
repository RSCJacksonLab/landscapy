"""Regression tests for canonical edge distance and conductance semantics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.analysis.dirichlet_energy import (
    calculate_ruggedness_dirichlet_energy,
)
from fitness_landscape.analysis.graph import resistance_distance_matrix
from fitness_landscape.core.edge_schema import (
    EDGE_SCHEMA_GRAPH_KEY,
    declare_edge_semantics,
    migrate_legacy_edge_semantics,
    resolve_edge_attribute,
)
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.graph import (
    create_diffusion_emb_graph,
    create_hamming_graph,
    create_knn_graph,
    create_tda_graph,
)
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence
from fitness_landscape.transforms.eigenmode import eigenmode_decomposition


def _binary_sequences() -> list[BinarySequence]:
    return [
        BinarySequence.from_bits(bits)
        for bits in ([0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1])
    ]


def test_hamming_and_knn_constructors_declare_distinct_edge_semantics():
    sequences = _binary_sequences()

    hamming = create_hamming_graph(sequences)
    hamming_schema = hamming.graph[EDGE_SCHEMA_GRAPH_KEY]
    assert hamming_schema["distance"] == {
        "key": "distance",
        "units": "hamming_count",
    }
    assert hamming_schema["normalized_distance"]["key"] == "normalized_distance"
    assert hamming_schema["conductance"]["key"] == "weight"
    for _, _, data in hamming.edges(data=True):
        assert data["distance"] == 1.0
        assert data["normalized_distance"] == 0.25
        assert data["weight"] == 1.0

    knn = create_knn_graph(sequences, k=3, backend="balltree")
    assert knn.graph[EDGE_SCHEMA_GRAPH_KEY]["constructor"] == "knn-balltree"
    by_distance = sorted(
        (data["distance"], data["weight"])
        for _, _, data in knn.edges(data=True)
    )
    assert by_distance[0][0] < by_distance[-1][0]
    assert by_distance[0][1] > by_distance[-1][1]
    for _, _, data in knn.edges(data=True):
        assert data["normalized_distance"] == pytest.approx(data["distance"] / 4.0)
        assert data["weight"] == pytest.approx(np.exp(-data["normalized_distance"]))
        assert data["knn_weight"] == data["distance"]


def test_diffusion_and_tda_constructors_never_store_distance_as_weight():
    sequences = [
        BaseNumpySequence([index], sequence_id=f"s{index}") for index in range(4)
    ]
    embeddings = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0], [2.0, 0.0]])

    diffusion = create_diffusion_emb_graph(
        sequences,
        embeddings,
        k=2,
        t=1,
        connectivity_threshold=0.0,
        backend="balltree",
    )
    assert diffusion.graph[EDGE_SCHEMA_GRAPH_KEY]["conductance"]["key"] == "weight"
    for _, _, data in diffusion.edges(data=True):
        assert data["weight"] == data["affinity"] == data["kernel_weight"]

    pytest.importorskip("gudhi")
    tda = create_tda_graph(sequences, embeddings, n_components=2)
    assert tda.graph[EDGE_SCHEMA_GRAPH_KEY]["distance"]["units"] == "pca_euclidean"
    for _, _, data in tda.edges(data=True):
        assert data["distance"] == data["tda_distance"]
        assert data["weight"] == data["affinity"]
        assert data["weight"] == pytest.approx(1.0 / (1.0 + data["distance"]))


def test_auto_resolution_rejects_ambiguous_legacy_weight():
    graph = nx.path_graph(2)
    graph[0][1]["weight"] = 3.0

    with pytest.raises(ValueError, match="semantics are undeclared"):
        resolve_edge_attribute(graph, "conductance")
    assert resolve_edge_attribute(graph, "conductance", "weight") == "weight"
    assert resolve_edge_attribute(graph, "conductance", None) is None


def test_declared_conductance_drives_laplacian_dirichlet_and_resistance():
    sequences = [
        BaseNumpySequence([0], sequence_id="a"),
        BaseNumpySequence([1], sequence_id="b"),
    ]
    graph = nx.Graph()
    graph.add_node("a", sequence=sequences[0])
    graph.add_node("b", sequence=sequences[1])
    graph.add_edge("a", "b", distance=4.0, affinity=0.25, weight=0.25)
    declare_edge_semantics(
        graph,
        constructor="known-answer",
        distance_key="distance",
        distance_units="arbitrary",
        affinity_key="affinity",
        conductance_key="weight",
    )
    landscape = FitnessLandscape(
        sequences,
        graph,
        fitness_layers={"fitness": NumericFitness.from_scalars("fitness", [0.0, 2.0])},
    )

    eigenvalues, _ = eigenmode_decomposition(landscape)
    np.testing.assert_allclose(eigenvalues, [0.0, 0.5], atol=1e-12)

    energy = calculate_ruggedness_dirichlet_energy(
        landscape,
        weighted_laplacian=True,
    )
    assert energy["weight_key"] == "weight"
    assert energy["total_dirichlet_energy"] == pytest.approx(0.5)

    resistance = resistance_distance_matrix(
        landscape,
        compute_resistance_matrix=True,
        weight_epsilon=0.0,
    )
    assert resistance["weight_key"] == "weight"
    assert resistance["resistance_mat"][0, 1] == pytest.approx(4.0)


def test_portable_bundle_migrates_known_legacy_knn_aliases(tmp_path):
    sequences = [
        BaseNumpySequence([0, 0], sequence_id="a"),
        BaseNumpySequence([0, 1], sequence_id="b"),
    ]
    graph = nx.Graph()
    graph.add_node(0, sequence=sequences[0])
    graph.add_node(1, sequence=sequences[1])
    graph.add_edge(0, 1, distance=1.0, weight=1.0, knn_weight=1.0)
    landscape = FitnessLandscape(sequences, graph)

    bundle_dir = tmp_path / "legacy-knn"
    landscape.save_bundle_dir(bundle_dir)
    loaded = FitnessLandscape.load_bundle_dir(bundle_dir)

    schema = loaded.graph.graph[EDGE_SCHEMA_GRAPH_KEY]
    assert schema["constructor"] == "legacy-knn"
    edge = loaded.graph[0][1]
    assert edge["distance"] == 1.0
    assert edge["normalized_distance"] == 0.5
    assert edge["weight"] == pytest.approx(np.exp(-0.5))


def test_portable_bundle_preserves_declared_edge_schema(tmp_path):
    sequences = _binary_sequences()[:2]
    graph = create_hamming_graph(sequences)
    landscape = FitnessLandscape(sequences, graph)

    bundle_dir = tmp_path / "declared-schema"
    landscape.save_bundle_dir(bundle_dir)
    loaded = FitnessLandscape.load_bundle_dir(bundle_dir)

    assert (
        loaded.graph.graph[EDGE_SCHEMA_GRAPH_KEY]
        == graph.graph[EDGE_SCHEMA_GRAPH_KEY]
    )
    assert loaded.graph[0][1]["distance"] == 1.0
    assert loaded.graph[0][1]["weight"] == 1.0


@pytest.mark.parametrize(
    "edge_data,constructor,expected_weight",
    [
        ({"kernel_weight": 0.4}, "legacy-diffusion", 0.4),
        ({"tda_distance": 3.0, "weight": 3.0}, "legacy-tda", 0.25),
    ],
)
def test_known_legacy_aliases_are_migrated_without_distance_conductance_confusion(
    edge_data,
    constructor,
    expected_weight,
):
    graph = nx.Graph()
    graph.add_edge("a", "b", **edge_data)

    assert migrate_legacy_edge_semantics(graph)
    assert graph.graph[EDGE_SCHEMA_GRAPH_KEY]["constructor"] == constructor
    assert graph["a"]["b"]["weight"] == pytest.approx(expected_weight)
