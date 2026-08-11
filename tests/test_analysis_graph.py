import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fitness_landscape.analysis.graph import (
    annotate_louvain_communities,
    resistance_distance_matrix,
    category_diffusion_hierarchy,
)
from fitness_landscape.analysis.random_walk import category_boundary_crossing_times
from fitness_landscape.core.annotation import AnnotationLayer
from fitness_landscape.core.fitness import CategoricalFitness, ProbabilisticCategoricalFitness
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence


def test_annotate_louvain_communities_creates_layer(binary_3bit_landscape):
    landscape = binary_3bit_landscape
    layer = annotate_louvain_communities(
        landscape,
        annotation_name="auto_comm",
        seed=123,
    )

    assert layer.name == "auto_comm"
    frame = layer.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert {"community_id", "community_label", "community_size", "louvain_community"} <= set(frame.columns)
    assert len(frame) == len(landscape.sequences)
    assert frame["community_id"].notna().any()

    for idx in range(len(landscape.sequences)):
        node = landscape._nodes_by_index[idx]
        annotations = landscape.graph.nodes[node].get("annotations", {})
        assert "auto_comm" in annotations
        assert "community_id" in annotations["auto_comm"]


def test_annotate_louvain_requires_overwrite_flag(binary_3bit_landscape):
    landscape = binary_3bit_landscape
    annotate_louvain_communities(landscape, annotation_name="auto_comm", seed=123)

    with pytest.raises(ValueError):
        annotate_louvain_communities(landscape, annotation_name="auto_comm", seed=123)

    layer = annotate_louvain_communities(
        landscape,
        annotation_name="auto_comm",
        seed=321,
        overwrite=True,
    )
    assert landscape.get_annotation_layer("auto_comm") is layer


def _small_categorical_landscape():
    seqs = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
        BaseNumpySequence([2], sequence_id="s2"),
    ]
    G = nx.path_graph(3)
    for idx, seq in enumerate(seqs):
        G.nodes[idx]["sequence"] = seq

    cat = CategoricalFitness(name="cat", values=["A", "A", "B"], categories=["A", "B"])
    annotations = AnnotationLayer(name="anno", data={"label": ["x", "y", "y"]})

    return FitnessLandscape(
        sequences=seqs,
        graph=G,
        fitness_layers={"cat": cat},
        annotation_layers={"anno": annotations},
    )


def test_resistance_distance_matrix_plain_graph_returns_matrix_only():
    G = nx.path_graph(3)
    res = resistance_distance_matrix(G)
    assert set(res.keys()) == {"resistance_mat", "sampled_nodes"}
    with pytest.raises(ValueError):
        resistance_distance_matrix(G, layers=["anything"])


def test_resistance_expected_pairwise_matches_formula():
    landscape = _small_categorical_landscape()
    res = resistance_distance_matrix(
        landscape,
        layers=["cat"],
        aggregation_function="expected_pairwise",
        compute_resistance_matrix=True,
    )
    R = res["resistance_mat"]
    cat_entry = res["cat"]

    categories = [cat_entry["categories"][i] for i in sorted(cat_entry["categories"])]
    assert categories == ["A", "B"]

    P = np.array(
        [
            [1.0, 0.0],  # A
            [1.0, 0.0],  # A
            [0.0, 1.0],  # B
        ]
    )
    masses = P.sum(axis=0)
    expected = (P.T @ R @ P) / (masses[:, None] * masses[None, :])
    expected[np.isnan(expected)] = 0.0
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(cat_entry["distance_mat"], expected)
    np.testing.assert_array_equal(cat_entry["distance_mat"], cat_entry["distance_max"])


def test_resistance_annotation_aggregation_defaults():
    landscape = _small_categorical_landscape()
    res = resistance_distance_matrix(landscape)
    assert "cat" in res
    assert "anno" in res

    anno_entry = res["anno"]
    assert anno_entry.get("column") == "label"
    cats = set(anno_entry["categories"].values())
    assert cats == {"x", "y"}
    mat = anno_entry["distance_mat"]
    assert mat.shape == (len(cats), len(cats))
    assert np.allclose(np.diag(mat), 0.0)


def test_resistance_wasserstein_collapses_identical_distributions():
    seqs = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
    ]
    G = nx.path_graph(2)
    for idx, seq in enumerate(seqs):
        G.nodes[idx]["sequence"] = seq

    probs = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    layer = ProbabilisticCategoricalFitness(
        name="prob",
        probabilities=probs,
        categories=["A", "B"],
    )
    landscape = FitnessLandscape(
        sequences=seqs,
        graph=G,
        fitness_layers={"prob": layer},
    )

    res = resistance_distance_matrix(
        landscape,
        layers=["prob"],
        aggregation_function="ot",
    )
    prob_entry = res["prob"]
    mat = prob_entry["distance_mat"]
    assert mat.shape == (2, 2)
    np.testing.assert_allclose(mat, np.zeros_like(mat), atol=1e-8)


def test_resistance_sampling_reduces_nodes():
    landscape = _small_categorical_landscape()
    res = resistance_distance_matrix(
        landscape,
        layers=["cat"],
        aggregation_function="expected_pairwise",
        compute_resistance_matrix=True,
        sample_nodes=2,
        sample_seed=42,
    )
    sampled = res.get("sampled_nodes")
    assert sampled is not None
    assert len(sampled) == 2
    R = res["resistance_mat"]
    assert R.shape == (2, 2)


def test_category_diffusion_hierarchy_basic():
    landscape = _small_categorical_landscape()
    out = category_diffusion_hierarchy(
        landscape,
        layer="cat",
        embedding_dim=2,
        skip_first=True,
    )
    assert out["embedding"].shape[0] == len(landscape.graph.nodes())
    assert out["centroids"].shape[0] == 2
    dist = out["pairwise_distances"]
    assert dist.shape == (2, 2)
    assert np.allclose(np.diag(dist), 0.0, atol=1e-8)
    assert out["distance_stats"]["min"] >= 0.0


def test_category_diffusion_hierarchy_filters_small_embeddings():
    landscape = _small_categorical_landscape()
    emb = np.array(
        [
            [0.0, 0.0],   # will be filtered
            [0.1, 0.0],
            [0.0, 0.2],
        ]
    )
    out = category_diffusion_hierarchy(
        landscape,
        layer="cat",
        embedding=emb,
        filter_small_embedding=True,
        embedding_norm_threshold=1e-3,
    )
    assert out["filtered_node_count"] == 1
    assert len(out["kept_node_indices"]) == 2
    assert out["embedding"].shape[0] == 2


def test_category_diffusion_hierarchy_filters_coordinate_threshold():
    landscape = _small_categorical_landscape()
    emb = np.array(
        [
            [0.0, 0.5],   # drop due to x=0
            [0.2, 0.0],   # drop due to y=0
            [0.3, 0.4],   # keep
        ]
    )
    out = category_diffusion_hierarchy(
        landscape,
        layer="cat",
        embedding=emb,
        filter_coordinate_threshold=0.05,
        filter_small_embedding=False,
    )
    assert out["filtered_node_count"] == 2
    assert len(out["kept_node_indices"]) == 1
    assert out["embedding"].shape[0] == 1


def test_category_boundary_crossing_times_simple():
    landscape = _small_categorical_landscape()
    out = category_boundary_crossing_times(
        landscape,
        layer="cat",
        n_walks=20,
        max_steps=5,
        seed=0,
    )
    mat = out["mean_crossing_time"]
    assert mat.shape == (2, 2)
    # From A->B should be finite (path_graph with immediate neighbor)
    assert np.isfinite(mat[0, 1])
    assert mat[0, 0] == 0.0
    # B->A also finite in this tiny graph
    assert np.isfinite(mat[1, 0])


def test_category_boundary_crossing_times_respects_edge_weights():
    sequences = [
        BaseNumpySequence([0], sequence_id="s0"),
        BaseNumpySequence([1], sequence_id="s1"),
        BaseNumpySequence([2], sequence_id="s2"),
    ]
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    for index, sequence in enumerate(sequences):
        graph.nodes[index]["sequence"] = sequence
    graph.add_edge(0, 1, transition_weight=1.0)
    graph.add_edge(0, 2, transition_weight=0.0)
    graph.add_edge(1, 2, transition_weight=1.0)
    categorical = CategoricalFitness.from_values(
        "cat",
        ["A", "B", "C"],
        categories=["A", "B", "C"],
    )
    landscape = FitnessLandscape(
        sequences=sequences,
        graph=graph,
        fitness_layers={"cat": categorical},
    )

    out = category_boundary_crossing_times(
        landscape,
        layer="cat",
        n_walks=20,
        max_steps=3,
        seed=0,
        weight_key="transition_weight",
    )

    assert out["mean_crossing_time"][0, 1] == 1.0
    assert out["hit_counts"][0, 1] == 20
    assert out["params"]["weight_key"] == "transition_weight"
