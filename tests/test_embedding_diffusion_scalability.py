"""Sparse-storage and feasibility regressions for embedding diffusion."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import fitness_landscape.core.graph as graph_module
from fitness_landscape.core.graph import (
    _reversible_diffusion_kernel,
    _reversible_lazy_transition,
    _select_diffusion_knn_candidates,
    create_diffusion_emb_graph,
)
from fitness_landscape.core.sequence import BinarySequence


def _sequences(n: int) -> list[BinarySequence]:
    return [BinarySequence([index % 2]) for index in range(n)]


def test_embedding_diffusion_scale_path_stays_sparse_and_linear_at_t1(monkeypatch):
    n, dimensions, k = 2048, 8, 4
    embeddings = np.random.default_rng(185).normal(size=(n, dimensions))
    original_require_optional = graph_module.require_optional

    def fail_dense_power(*args, **kwargs):
        raise AssertionError("embedding diffusion must not use dense matrix power")

    def reject_dense_pairwise(module_name, **kwargs):
        if module_name == "sklearn.metrics.pairwise":
            raise AssertionError("embedding diffusion must not request dense RBF kernels")
        return original_require_optional(module_name, **kwargs)

    monkeypatch.setattr(graph_module.np.linalg, "matrix_power", fail_dense_power)
    monkeypatch.setattr(graph_module, "require_optional", reject_dense_pairwise)
    graph = create_diffusion_emb_graph(
        _sequences(n),
        embeddings,
        embedding_domain="plm",
        backend="balltree",
        k=k,
        t=1,
        connectivity_threshold=0.0,
    )

    construction = graph.graph["diffusion_construction"]
    assert construction["storage"] == "csr"
    assert construction["diffusion_accuracy"] == "exact"
    assert construction["affinity_nnz"] <= 2 * n * k
    assert construction["transition_nnz"] <= construction["affinity_nnz"] + n
    assert construction["kernel_nnz"] <= construction["transition_nnz"]
    assert construction["estimated_scalar_products"] == construction["transition_nnz"]
    assert graph.number_of_edges() <= construction["affinity_nnz"] // 2


def test_tiebuffer_only_adds_observed_kth_distance_ties():
    features = np.array([[0.0], [1.0], [-1.0], [4.0]])
    neighbours = np.array(
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1],
            [3, 1, 0],
        ]
    )
    selected = _select_diffusion_knn_candidates(
        features,
        neighbours,
        k=1,
        distance_geometry="euclidean",
    )

    assert set(selected[0]) == {1, 2}  # both exactly one unit away
    assert set(selected[1]) == {0}  # buffered node 2 is farther away
    assert set(selected[3]) == {1}  # buffered node 0 is farther away


def test_include_self_does_not_reduce_public_nonself_candidate_count(monkeypatch):
    embeddings = np.arange(5, dtype=float)[:, None]
    captured = {}

    def fake_find(features, k, tiebuffer=0, **kwargs):
        captured["k"] = k
        captured["include_self"] = kwargs["include_self"]
        indices = np.array(
            [
                [0, 1, 2],
                [1, 0, 2],
                [2, 1, 3],
                [3, 2, 4],
                [4, 3, 2],
            ]
        )
        return np.zeros_like(indices, dtype=float), indices

    monkeypatch.setattr(graph_module, "_find_knn_balltree", fake_find)
    graph = create_diffusion_emb_graph(
        _sequences(5),
        embeddings,
        embedding_domain="plm",
        backend="balltree",
        k=2,
        include_self=True,
        t=1,
        connectivity_threshold=0.0,
    )

    assert captured == {"k": 3, "include_self": True}
    assert graph.graph["diffusion_construction"]["effective_k"] == 2


def test_approximate_backend_is_declared_as_part_of_candidate_kernel(monkeypatch):
    embeddings = np.array([[0.0], [1.0], [3.0]])

    def fake_faiss(features, k, **kwargs):
        return (
            np.zeros((3, 2), dtype=float),
            np.array([[0, 1], [1, 0], [2, 1]], dtype=np.int64),
        )

    monkeypatch.setattr(graph_module, "_find_knn_faiss", fake_faiss)
    graph = create_diffusion_emb_graph(
        _sequences(3),
        embeddings,
        embedding_domain="plm",
        backend="faiss",
        index_type="hnsw",
        k=1,
        t=1,
        connectivity_threshold=0.0,
    )

    construction = graph.graph["diffusion_construction"]
    assert construction["candidate_backend_approximate"] is True
    assert construction["tie_rule"] == "all_returned_candidates_at_exact_kth_distance"


def test_exact_sparse_power_raises_before_nonzero_growth_exceeds_budget():
    n = 64
    rows = np.arange(n - 1)
    affinity = sparse.coo_matrix(
        (
            np.ones(2 * (n - 1)),
            (
                np.concatenate([rows, rows + 1]),
                np.concatenate([rows + 1, rows]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    transition, stationary, labels = _reversible_lazy_transition(affinity)

    with pytest.raises(MemoryError, match="max_diffusion_nnz|Reduce `k`, `t`"):
        _reversible_diffusion_kernel(
            transition,
            stationary,
            labels,
            stationary_limit=False,
            power=4,
            max_nnz=300,
            max_work=100_000,
        )


def test_budgeted_sparse_power_matches_dense_exact_power():
    affinity = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    dense_transition, dense_stationary, dense_labels = _reversible_lazy_transition(
        affinity
    )
    sparse_transition, sparse_stationary, sparse_labels = _reversible_lazy_transition(
        sparse.csr_matrix(affinity)
    )
    expected = _reversible_diffusion_kernel(
        dense_transition,
        dense_stationary,
        dense_labels,
        stationary_limit=False,
        power=4,
    )
    observed = _reversible_diffusion_kernel(
        sparse_transition,
        sparse_stationary,
        sparse_labels,
        stationary_limit=False,
        power=4,
        max_nnz=100,
        max_work=1000,
    )

    assert sparse.isspmatrix_csr(observed)
    assert np.allclose(observed.toarray(), expected)


def test_stationary_limit_refuses_quadratic_component_before_allocation():
    n = 40
    embeddings = np.cumsum(np.linspace(1.0, 2.0, n))[:, None]
    with pytest.raises(MemoryError, match="componentwise stationary limit"):
        create_diffusion_emb_graph(
            _sequences(n),
            embeddings,
            embedding_domain="plm",
            backend="balltree",
            k=1,
            t=None,
            connectivity_threshold=0.0,
            max_diffusion_nnz=1000,
        )


def test_exact_sparse_power_enforces_work_budget_before_multiplication():
    n = 32
    embeddings = np.random.default_rng(7).normal(size=(n, 3))
    with pytest.raises(MemoryError, match="max_diffusion_work"):
        create_diffusion_emb_graph(
            _sequences(n),
            embeddings,
            embedding_domain="plm",
            backend="balltree",
            k=3,
            t=2,
            max_diffusion_work=32,
        )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("max_diffusion_nnz", 0, ValueError),
        ("max_diffusion_nnz", 1.5, TypeError),
        ("max_diffusion_work", True, TypeError),
    ],
)
def test_diffusion_resource_budgets_are_strict_positive_integers(name, value, error):
    with pytest.raises(error, match=name):
        create_diffusion_emb_graph(
            [BinarySequence([0])],
            np.array([[0.0]]),
            k=1,
            **{name: value},
        )
