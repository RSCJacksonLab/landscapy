import numpy as np
import networkx as nx
import pytest

from fitness_landscape.analysis.graph_induction_alignment import procrustes
from fitness_landscape.analysis.statistics import hypothesis_testing
from fitness_landscape.graph_matching import (
    cosine_similarity_matrix,
    graph_to_length_matrix,
    isorank_with_features,
    normalize_adj_matrix,
)


def test_procrustes_identity_alignment():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    Y = X.copy()
    aligned, rmse = procrustes(X, Y)

    assert aligned.shape == X.shape
    assert rmse == pytest.approx(0.0, abs=1e-8)


def test_graph_to_length_matrix_returns_square_distance():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    lengths = graph_to_length_matrix(G, transform="reciprocal")

    assert lengths.shape == (2, 2)
    assert lengths[0, 1] == pytest.approx(0.5)
    assert lengths[0, 0] == 0.0


def test_normalize_adj_matrix_handles_sink_rows():
    G = nx.Graph()
    G.add_edge(0, 1, weight=2.0)
    G.add_node(2)  # isolated row

    mat = normalize_adj_matrix(G)

    assert mat.shape == (3, 3)
    assert mat[2].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_normalize_adj_matrix_rejects_non_graph_and_handles_empty_graph():
    with pytest.raises(TypeError):
        normalize_adj_matrix("bad")

    empty = normalize_adj_matrix(nx.Graph())
    assert empty.shape == (0, 0)


def test_cosine_similarity_matrix_returns_identity_diagonal_for_matching_features():
    sim = cosine_similarity_matrix(np.eye(2), np.eye(2))
    np.testing.assert_allclose(np.diag(sim), np.ones(2))


def test_isorank_with_features_validates_shapes_and_uses_uniform_prior():
    g1 = nx.path_graph(2)
    g2 = nx.path_graph(2)

    with pytest.raises(ValueError):
        isorank_with_features(g1, g2, np.ones((1, 2)), np.ones((2, 2)))

    zero_prior = isorank_with_features(
        g1,
        g2,
        np.zeros((2, 1)),
        np.zeros((2, 1)),
        alpha=0.7,
        max_iter=5,
        tol=0.0,
    )
    np.testing.assert_allclose(zero_prior, np.full((2, 2), 0.25))

    aligned = isorank_with_features(
        g1,
        g2,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        alpha=0.5,
        max_iter=20,
        tol=1e9,
    )
    assert aligned.shape == (2, 2)
    assert aligned[0, 0] > aligned[0, 1]


def test_hypothesis_testing_basic_groups():
    groups = {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([1.5, 2.0, 2.5]),
    }

    result = hypothesis_testing(groups=groups, run_tests=("ttest",))

    assert set(result.keys()) == {"group_stats", "pairwise_tests"}
    assert set(result["group_stats"].keys()) == {"A", "B"}
    assert "B" in result["pairwise_tests"]["A"]
    t_result = result["pairwise_tests"]["A"]["B"]["t_test"]
    assert {"statistic", "p_value", "significant"} <= t_result.keys()
