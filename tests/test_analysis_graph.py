import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fitness_landscape.analysis.graph import annotate_louvain_communities, resistance_distance_matrix
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
    assert set(res.keys()) == {"resistance_mat"}
    with pytest.raises(ValueError):
        resistance_distance_matrix(G, layers=["anything"])


def test_resistance_expected_pairwise_matches_formula():
    landscape = _small_categorical_landscape()
    res = resistance_distance_matrix(
        landscape,
        layers=["cat"],
        aggregation_function="expected_pairwise",
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
