"""Known-answer tests for reversible undirected diffusion semantics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fitness_landscape._const import PROT_20
from fitness_landscape.core.graph import (
    _reversible_diffusion_kernel,
    _reversible_lazy_transition,
    _threshold_undirected_kernel,
    create_diffusion_emb_graph,
    create_evol_diffusion_graph,
)
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence


def _as_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_reversible_kernel_matches_exact_three_state_solution(sparse_input):
    affinity = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    source = csr_matrix(affinity) if sparse_input else affinity
    transition, stationary, labels = _reversible_lazy_transition(source)

    expected_transition = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.0 / 3.0, 0.5, 1.0 / 6.0],
            [0.0, 0.5, 0.5],
        ]
    )
    expected_stationary = np.array([1.0 / 3.0, 0.5, 1.0 / 6.0])
    expected_kernel = np.array(
        [
            [0.5, np.sqrt(1.0 / 6.0), 0.0],
            [np.sqrt(1.0 / 6.0), 0.5, np.sqrt(1.0 / 12.0)],
            [0.0, np.sqrt(1.0 / 12.0), 0.5],
        ]
    )

    kernel = _reversible_diffusion_kernel(
        transition,
        stationary,
        labels,
        stationary_limit=False,
        power=1,
    )
    flux = stationary[:, None] * _as_dense(transition)

    assert np.allclose(_as_dense(transition), expected_transition)
    assert np.allclose(stationary, expected_stationary)
    assert np.allclose(flux, flux.T)
    assert np.allclose(_as_dense(kernel), expected_kernel)
    assert np.allclose(_as_dense(kernel), _as_dense(kernel).T)


def test_stationary_limit_is_componentwise_for_reducible_and_isolated_states():
    affinity = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    transition, stationary, labels = _reversible_lazy_transition(affinity)
    kernel = _reversible_diffusion_kernel(
        transition,
        stationary,
        labels,
        stationary_limit=True,
        power=None,
    )

    expected = np.zeros((5, 5))
    expected[:2, :2] = 0.5
    expected[2:4, 2:4] = 0.5
    expected[4, 4] = 1.0

    assert np.allclose(kernel, expected)
    assert np.allclose(_as_dense(transition).sum(axis=1), 1.0)
    assert kernel[0, 2] == kernel[0, 4] == 0.0


def test_lazy_transition_removes_two_cycle_periodicity():
    transition, stationary, labels = _reversible_lazy_transition(
        np.array([[0.0, 1.0], [1.0, 0.0]])
    )
    finite = _reversible_diffusion_kernel(
        transition,
        stationary,
        labels,
        stationary_limit=False,
        power=20,
    )
    stationary_kernel = _reversible_diffusion_kernel(
        transition,
        stationary,
        labels,
        stationary_limit=True,
        power=None,
    )

    assert np.allclose(np.diag(transition), 0.5)
    assert np.allclose(finite, stationary_kernel, atol=1e-12)


def test_threshold_is_applied_after_symmetric_kernel_construction():
    affinity = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    transition, stationary, labels = _reversible_lazy_transition(affinity)
    kernel = _reversible_diffusion_kernel(
        transition,
        stationary,
        labels,
        stationary_limit=False,
        power=1,
    )
    rows, cols, weights = _threshold_undirected_kernel(kernel, threshold=0.35)

    assert list(zip(rows, cols)) == [(0, 1)]
    assert weights == pytest.approx([np.sqrt(1.0 / 6.0)])


def _embedding_edge_weights(graph, sequences):
    result = {}
    for left, right, data in graph.edges(data=True):
        endpoints = tuple(
            sorted(
                (
                    tuple(sequences[left].to_array().tolist()),
                    tuple(sequences[right].to_array().tolist()),
                )
            )
        )
        result[endpoints] = data["weight"]
    return result


@pytest.mark.parametrize("power", [1, 3, None, 0, np.inf])
def test_embedding_diffusion_is_permutation_equivariant(power):
    sequences = [
        BinarySequence([0, 0, 0]),
        BinarySequence([0, 0, 1]),
        BinarySequence([0, 1, 1]),
        BinarySequence([1, 1, 1]),
    ]
    embeddings = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.2], [2.5, 0.1]])
    graph = create_diffusion_emb_graph(
        sequences,
        embeddings,
        k=2,
        t=power,
        connectivity_threshold=0.0,
    )

    order = [2, 0, 3, 1]
    reordered_sequences = [sequences[index] for index in order]
    reordered = create_diffusion_emb_graph(
        reordered_sequences,
        embeddings[order],
        k=2,
        t=power,
        connectivity_threshold=0.0,
    )

    expected = _embedding_edge_weights(graph, sequences)
    observed = _embedding_edge_weights(reordered, reordered_sequences)
    assert expected.keys() == observed.keys()
    assert all(observed[key] == pytest.approx(value) for key, value in expected.items())
    assert graph.graph["diffusion_semantics"]["threshold_units"] == (
        "dimensionless_diffusion_amplitude"
    )


def test_embedding_constructor_matches_manual_one_step_kernel():
    sequences = [BinarySequence([0, 0]), BinarySequence([0, 1]), BinarySequence([1, 1])]
    embeddings = np.array([[0.0], [1.0], [3.0]])
    graph = create_diffusion_emb_graph(
        sequences,
        embeddings,
        k=1,
        t=1,
        connectivity_threshold=0.0,
    )

    # k=1 gives sigma values [1, 1, 2], whose median is one.
    pairwise_squared = (embeddings - embeddings.T) ** 2
    affinity = np.exp(-0.5 * pairwise_squared)
    np.fill_diagonal(affinity, 0.0)
    degrees = affinity.sum(axis=1)
    transition = 0.5 * (np.eye(3) + affinity / degrees[:, None])
    stationary = degrees / degrees.sum()
    expected = (
        np.sqrt(stationary)[:, None]
        * transition
        / np.sqrt(stationary)[None, :]
    )

    observed = nx.to_numpy_array(graph, weight="weight")
    assert np.allclose(observed[np.triu_indices(3, k=1)], expected[np.triu_indices(3, k=1)])


def test_embedding_stationary_kernel_does_not_cross_underflow_disconnection():
    sequences = [BinarySequence([0, 0]), BinarySequence([0, 1]), BinarySequence([1, 1])]
    graph = create_diffusion_emb_graph(
        sequences,
        np.array([[0.0], [1.0], [1.0e6]]),
        k=1,
        t=None,
        connectivity_threshold=0.0,
    )

    assert set(graph.edges()) == {(0, 1)}
    assert graph[0][1]["weight"] == pytest.approx(0.5)
    assert graph.degree[2] == 0


def _protein(symbol: str, sequence_id: str) -> BaseNumpySequence:
    return BaseNumpySequence(
        [symbol, symbol, symbol],
        sequence_id=sequence_id,
        alphabet=PROT_20,
    )


def _protein_edge_weights(graph):
    weights = {}
    for left, right, data in graph.edges(data=True):
        endpoints = tuple(
            sorted((graph.nodes[left]["sequence"].id, graph.nodes[right]["sequence"].id))
        )
        weights[endpoints] = data["weight"]
    return weights


def test_evolutionary_stationary_kernel_preserves_reducible_components_and_order():
    sequences = [
        _protein("A", "a"),
        _protein("R", "r"),
        _protein("N", "n"),
        _protein("D", "d"),
    ]
    embeddings = np.array([[0.0], [0.1], [10.0], [10.1]])
    graph = create_evol_diffusion_graph(
        sequences,
        embeddings,
        backend="balltree",
        k=1,
        t=None,
        connectivity_threshold=0.0,
        cpus=1,
    )

    order = [2, 0, 3, 1]
    reordered = create_evol_diffusion_graph(
        [sequences[index] for index in order],
        embeddings[order],
        backend="balltree",
        k=1,
        t=np.inf,
        connectivity_threshold=0.0,
        cpus=1,
    )

    expected = _protein_edge_weights(graph)
    observed = _protein_edge_weights(reordered)
    assert expected == pytest.approx({("a", "r"): 0.5, ("d", "n"): 0.5})
    assert observed == pytest.approx(expected)
    assert not graph.has_edge(0, 2)


def test_evolutionary_two_state_finite_kernel_has_exact_lazy_weight():
    graph = create_evol_diffusion_graph(
        [_protein("A", "a"), _protein("R", "r")],
        np.array([[0.0], [0.1]]),
        backend="balltree",
        k=1,
        t=1,
        connectivity_threshold=0.0,
        cpus=1,
    )

    assert set(graph.edges()) == {(0, 1)}
    assert graph[0][1]["weight"] == pytest.approx(0.5)
