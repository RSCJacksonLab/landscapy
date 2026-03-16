import networkx as nx
import numpy as np
import pytest

from fitness_landscape.graph_matching.isorank import (
    cosine_similarity_matrix,
    isorank_with_features,
    normalize_adj_matrix,
)


def test_normalize_adj_matrix_validates_input_and_handles_empty_and_sink_rows():
    with pytest.raises(TypeError):
        normalize_adj_matrix("bad")

    empty = normalize_adj_matrix(nx.Graph())
    assert empty.shape == (0, 0)

    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)

    mat = normalize_adj_matrix(graph)
    np.testing.assert_allclose(mat[1], np.array([1 / 3, 1 / 3, 1 / 3]))
    np.testing.assert_allclose(mat[2], np.array([1 / 3, 1 / 3, 1 / 3]))


def test_cosine_similarity_matrix_wrapper_returns_expected_scores():
    sim = cosine_similarity_matrix(np.eye(2), np.eye(2))
    np.testing.assert_allclose(np.diag(sim), np.ones(2))


def test_isorank_with_features_validates_feature_shapes_and_uniform_prior():
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


def test_isorank_with_features_prefers_aligned_feature_pairs():
    g1 = nx.path_graph(2)
    g2 = nx.path_graph(2)

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
