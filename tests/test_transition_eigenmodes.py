"""Regression tests for random-walk Laplacian eigenmodes."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.transforms.eigenmode import eigenmode_decomposition


def _random_walk_laplacian(graph: nx.Graph, weight=None) -> tuple[np.ndarray, np.ndarray]:
    adjacency = nx.to_numpy_array(graph, weight=weight, dtype=float)
    degrees = adjacency.sum(axis=1)
    inverse = np.zeros_like(degrees)
    inverse[degrees > 0.0] = 1.0 / degrees[degrees > 0.0]
    operator = np.diag((degrees > 0.0).astype(float)) - inverse[:, None] * adjacency
    measure = np.where(degrees > 0.0, degrees, 1.0)
    return operator, measure


def _assert_right_eigensystem(operator, eigenvalues, eigenvectors, *, atol=1e-10):
    residual = operator @ eigenvectors - eigenvectors * eigenvalues[None, :]
    assert np.linalg.norm(residual) <= atol
    assert np.isrealobj(eigenvalues)
    assert np.isrealobj(eigenvectors)
    assert np.all(np.diff(eigenvalues) >= -atol)


def test_transition_path_has_exact_spectrum_and_stationary_right_mode():
    graph = nx.path_graph(4)
    operator, measure = _random_walk_laplacian(graph)
    eigenvalues, eigenvectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key=None,
    )

    assert eigenvalues == pytest.approx([0.0, 0.5, 1.5, 2.0])
    _assert_right_eigensystem(operator, eigenvalues, eigenvectors)
    stationary = eigenvectors[:, 0] / eigenvectors[0, 0]
    assert stationary == pytest.approx(np.ones(4))
    assert eigenvectors.T @ np.diag(measure) @ eigenvectors == pytest.approx(
        np.eye(4)
    )


def test_transition_weighted_irregular_graph_matches_general_eigenvalues():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "b", 1.0), ("b", "c", 2.0), ("c", "d", 4.0)],
        weight="conductance",
    )
    operator, measure = _random_walk_laplacian(graph, weight="conductance")
    expected = np.sort(np.real_if_close(np.linalg.eigvals(operator)))
    eigenvalues, eigenvectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key="conductance",
    )

    assert eigenvalues == pytest.approx(expected)
    _assert_right_eigensystem(operator, eigenvalues, eigenvectors)
    assert eigenvectors.T @ np.diag(measure) @ eigenvectors == pytest.approx(
        np.eye(4)
    )


def test_transition_regular_cycle_exact_spectrum():
    graph = nx.cycle_graph(6)
    operator, _ = _random_walk_laplacian(graph)
    eigenvalues, eigenvectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key=None,
    )

    assert eigenvalues == pytest.approx([0.0, 0.5, 0.5, 1.5, 1.5, 2.0])
    _assert_right_eigensystem(operator, eigenvalues, eigenvectors)


def test_transition_sparse_and_dense_paths_agree_on_low_modes():
    graph = nx.path_graph(20)
    operator, measure = _random_walk_laplacian(graph)
    dense_values, dense_vectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key=None,
        k=None,
    )
    sparse_values, sparse_vectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key=None,
        k=5,
        dense_threshold=0,
    )

    assert sparse_values == pytest.approx(dense_values[:5], abs=1e-10)
    _assert_right_eigensystem(operator, sparse_values, sparse_vectors, atol=1e-8)
    overlap = sparse_vectors.T @ np.diag(measure) @ dense_vectors[:, :5]
    assert np.abs(overlap) == pytest.approx(np.eye(5), abs=1e-7)


def test_transition_isolates_are_stationary_zero_modes():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1)
    operator, measure = _random_walk_laplacian(graph)
    eigenvalues, eigenvectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key=None,
    )

    assert eigenvalues == pytest.approx([0.0, 0.0, 2.0])
    assert np.count_nonzero(np.isclose(eigenvalues, 0.0)) == 2
    _assert_right_eigensystem(operator, eigenvalues, eigenvectors)
    assert eigenvectors.T @ np.diag(measure) @ eigenvectors == pytest.approx(
        np.eye(3)
    )


def test_transition_sparse_stationary_mode_is_first():
    graph = nx.path_graph(30)
    operator, _ = _random_walk_laplacian(graph)
    eigenvalues, eigenvectors = eigenmode_decomposition(
        graph,
        matrix="transition",
        weight_key=None,
        k=1,
        dense_threshold=0,
    )

    assert eigenvalues == pytest.approx([0.0], abs=1e-12)
    _assert_right_eigensystem(operator, eigenvalues, eigenvectors, atol=1e-8)
    assert eigenvectors[:, 0] / eigenvectors[0, 0] == pytest.approx(np.ones(30))


def test_transition_rejects_directed_graphs_and_invalid_mode_counts():
    with pytest.raises(TypeError, match="undirected"):
        eigenmode_decomposition(nx.DiGraph([(0, 1)]), matrix="transition")
    with pytest.raises(ValueError, match="positive integer"):
        eigenmode_decomposition(nx.path_graph(3), matrix="transition", k=0)
    with pytest.raises(TypeError, match="positive integer"):
        eigenmode_decomposition(nx.path_graph(3), matrix="transition", k=1.5)
