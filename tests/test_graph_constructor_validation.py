"""Regression tests for graph-constructor validation and backend contracts."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import fitness_landscape.core.graph as graph_module
from fitness_landscape._const import PROT_20
from fitness_landscape.core.edge_schema import edge_semantics
from fitness_landscape.core.graph import (
    _find_knn_faiss,
    create_diffusion_emb_graph,
    create_evol_diffusion_graph,
    create_knn_graph,
    create_tda_graph,
)
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence


def _binary_sequences(n: int = 3) -> list[BinarySequence]:
    values = ([0, 0], [0, 1], [1, 1])
    return [BinarySequence(values[index]) for index in range(n)]


def _protein_sequence() -> BaseNumpySequence:
    return BaseNumpySequence(["A", "R"], alphabet=PROT_20)


def test_sequence_constructors_reject_unaligned_or_non_sequence_entries():
    unaligned = [BinarySequence([0]), BinarySequence([0, 1])]
    with pytest.raises(ValueError, match="uniform aligned length"):
        create_knn_graph(unaligned, k=1)

    with pytest.raises(TypeError, match="BaseNumpySequence"):
        create_knn_graph([BinarySequence([0]), object()], k=1)


@pytest.mark.parametrize("backend", ["balltree", "faiss", "auto"])
def test_knn_defines_empty_and_singleton_behavior_without_backend_calls(
    backend,
    monkeypatch,
):
    def fail_optional(*args, **kwargs):
        raise AssertionError("an optional backend should not be imported")

    monkeypatch.setattr(graph_module, "require_optional", fail_optional)
    empty = create_knn_graph([], k=50, backend=backend)
    singleton = create_knn_graph([BinarySequence([0, 1])], k=50, backend=backend)

    assert empty.number_of_nodes() == empty.number_of_edges() == 0
    assert singleton.number_of_nodes() == 1
    assert singleton.number_of_edges() == 0
    assert edge_semantics(empty)["conductance"]["key"] == "weight"
    assert edge_semantics(singleton)["conductance"]["key"] == "weight"


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"k": 0}, ValueError, "`k`"),
        ({"k": 1.5}, TypeError, "`k`"),
        ({"k": True}, TypeError, "`k`"),
        ({"tiebuffer": -1}, ValueError, "tiebuffer"),
        ({"tie_policy": "first"}, ValueError, "tie_policy"),
        ({"index_type": "pq"}, ValueError, "index_type"),
        ({"faiss_metric": "cosine"}, ValueError, "FAISS metric"),
        ({"hnsw_M": 0}, ValueError, "hnsw_M"),
        ({"include_self": 1}, TypeError, "include_self"),
        ({"use_gpu": 1}, TypeError, "use_gpu"),
        ({"backend": "balltree", "use_gpu": True}, ValueError, "FAISS backend"),
        (
            {"backend": "faiss", "use_gpu": True, "index_type": "hnsw"},
            ValueError,
            "index_type='flat'",
        ),
    ],
)
def test_knn_options_fail_before_backend_calls(kwargs, error, message):
    options = {"k": 1, **kwargs}
    with pytest.raises(error, match=message):
        create_knn_graph(_binary_sequences(), **options)


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([0.0, 1.0, 2.0]),
        np.zeros((2, 2)),
        np.empty((3, 0)),
        np.array([[0.0], [np.nan], [1.0]]),
        np.array([[0.0], [np.inf], [1.0]]),
    ],
)
def test_embedding_diffusion_rejects_invalid_embedding_contract(embeddings):
    with pytest.raises(ValueError, match="embeddings"):
        create_diffusion_emb_graph(_binary_sequences(), embeddings=embeddings, k=1)


@pytest.mark.parametrize("t", [np.nan, -np.inf, -1, 1.5])
def test_diffusion_rejects_invalid_powers(t):
    with pytest.raises(ValueError, match="`t`|Finite `t`"):
        create_diffusion_emb_graph(
            [BinarySequence([0, 1])],
            embeddings=np.array([[0.0]]),
            k=1,
            t=t,
        )


@pytest.mark.parametrize("t", [True, "stationary"])
def test_diffusion_rejects_non_numeric_powers(t):
    with pytest.raises(TypeError, match="`t`"):
        create_diffusion_emb_graph(
            [BinarySequence([0, 1])],
            embeddings=np.array([[0.0]]),
            k=1,
            t=t,
        )


@pytest.mark.parametrize("t", [None, 0, 0.0, np.inf])
def test_diffusion_accepts_documented_stationary_sentinels(t):
    graph = create_diffusion_emb_graph(
        [BinarySequence([0, 1])],
        embeddings=np.array([[0.0]]),
        k=1,
        t=t,
    )
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0


@pytest.mark.parametrize("threshold", [None, np.nan, np.inf, -0.1, 1.1])
def test_diffusion_rejects_invalid_connectivity_thresholds(threshold):
    error = TypeError if threshold is None else ValueError
    with pytest.raises(error, match="connectivity_threshold"):
        create_diffusion_emb_graph(
            [BinarySequence([0, 1])],
            embeddings=np.array([[0.0]]),
            k=1,
            connectivity_threshold=threshold,
        )


def test_diffusion_rejects_boolean_connectivity_threshold():
    with pytest.raises(TypeError, match="connectivity_threshold"):
        create_diffusion_emb_graph(
            [BinarySequence([0, 1])],
            embeddings=np.array([[0.0]]),
            k=1,
            connectivity_threshold=True,
        )


def test_evolutionary_diffusion_validates_tau_and_shared_options_early():
    sequence = [_protein_sequence()]
    embedding = np.array([[0.0]])

    for tau in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="tau"):
            create_evol_diffusion_graph(sequence, embedding, k=1, tau=tau)
    with pytest.raises(TypeError, match="tau"):
        create_evol_diffusion_graph(sequence, embedding, k=1, tau=True)
    with pytest.raises(ValueError, match="backend"):
        create_evol_diffusion_graph(sequence, embedding, k=1, backend="unknown")


def test_tda_clips_components_for_small_inputs_and_defines_singletons():
    sequences = _binary_sequences(2)
    graph = create_tda_graph(
        sequences,
        np.array([[0.0], [1.0]]),
        n_components=3,
    )
    assert graph.graph["tda_requested_components"] == 3
    assert graph.graph["tda_effective_components"] == 1

    singleton = create_tda_graph(
        [BinarySequence([0, 1])],
        np.array([[0.0]]),
    )
    assert singleton.number_of_nodes() == 1
    assert singleton.number_of_edges() == 0
    assert singleton.graph["tda_effective_components"] == 0


def test_tda_rejects_duplicate_points_and_invalid_components():
    sequences = _binary_sequences(2)
    with pytest.raises(ValueError, match="duplicate points"):
        create_tda_graph(sequences, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="n_components"):
        create_tda_graph(sequences, np.array([[0.0], [1.0]]), n_components=0)
    with pytest.raises(TypeError, match="n_components"):
        create_tda_graph(sequences, np.array([[0.0], [1.0]]), n_components=1.5)


class _FakeIndex:
    def __init__(self, owner, kind, dimension, metric=None, nlist=None):
        self.owner = owner
        self.owner.created.append(
            {
                "kind": kind,
                "dimension": dimension,
                "metric": metric,
                "nlist": nlist,
            }
        )
        self.nprobe = None

    def train(self, values):
        self.owner.trained_shape = values.shape

    def add(self, values):
        self.owner.added_shape = values.shape

    def search(self, values, query_size):
        self.owner.query_size = query_size
        n = values.shape[0]
        indices = np.tile(np.arange(query_size), (n, 1))
        return np.zeros((n, query_size), dtype=np.float32), indices


class _FakeFaiss(SimpleNamespace):
    METRIC_INNER_PRODUCT = 0
    METRIC_L2 = 1

    def __init__(self):
        super().__init__()
        self.created = []

    def IndexFlatIP(self, dimension):
        return _FakeIndex(self, "flat-ip", dimension, self.METRIC_INNER_PRODUCT)

    def IndexFlatL2(self, dimension):
        return _FakeIndex(self, "flat-l2", dimension, self.METRIC_L2)

    def IndexHNSWFlat(self, dimension, hnsw_M, metric):
        index = _FakeIndex(self, "hnsw", dimension, metric)
        self.hnsw_M = hnsw_M
        return index

    def IndexIVFFlat(self, quantizer, dimension, nlist, metric):
        return _FakeIndex(self, "ivf", dimension, metric, nlist)


@pytest.mark.parametrize(
    ("metric", "expected_metric", "quantizer_kind"),
    [("ip", 0, "flat-ip"), ("l2", 1, "flat-l2")],
)
def test_faiss_ivf_honors_metric_caps_nlist_and_query_size(
    monkeypatch,
    metric,
    expected_metric,
    quantizer_kind,
):
    fake = _FakeFaiss()
    monkeypatch.setattr(graph_module, "require_optional", lambda *args, **kwargs: fake)
    values = np.arange(15, dtype=np.float32).reshape(5, 3)

    distances, indices = _find_knn_faiss(
        values,
        k=50,
        tiebuffer=50,
        index_type="ivf",
        metric=metric,
    )

    ivf = next(created for created in fake.created if created["kind"] == "ivf")
    assert ivf["metric"] == expected_metric
    assert 1 <= ivf["nlist"] <= len(values)
    assert fake.created[0]["kind"] == quantizer_kind
    assert fake.query_size == len(values)
    assert distances.shape == indices.shape == (len(values), len(values))


def test_invalid_faiss_metric_is_reported_before_optional_import(monkeypatch):
    def fail_optional(*args, **kwargs):
        raise AssertionError("FAISS should not be imported for invalid options")

    monkeypatch.setattr(graph_module, "require_optional", fail_optional)
    with pytest.raises(ValueError, match="Unsupported FAISS metric 'cosine'"):
        _find_knn_faiss(np.zeros((3, 2)), k=1, metric="cosine")


def test_missing_faiss_reports_portable_fallback(monkeypatch):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError(
            "FAISS is unavailable on this platform.",
            name="faiss",
        )

    monkeypatch.setattr(graph_module, "require_optional", missing)

    with pytest.raises(ModuleNotFoundError, match="backend='balltree'"):
        _find_knn_faiss(np.zeros((3, 2)), k=1)


def test_cpu_only_faiss_reports_gpu_and_balltree_fallbacks(monkeypatch):
    fake = _FakeFaiss()
    monkeypatch.setattr(graph_module, "require_optional", lambda *args, **kwargs: fake)

    with pytest.raises(RuntimeError, match="use_gpu=False.*backend='balltree'"):
        _find_knn_faiss(
            np.zeros((3, 2)),
            k=1,
            index_type="flat",
            use_gpu=True,
        )


@pytest.mark.parametrize(("metric", "expected_metric"), [("ip", 0), ("l2", 1)])
def test_faiss_hnsw_receives_requested_metric(monkeypatch, metric, expected_metric):
    fake = _FakeFaiss()
    monkeypatch.setattr(graph_module, "require_optional", lambda *args, **kwargs: fake)
    _find_knn_faiss(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        k=2,
        index_type="hnsw",
        metric=metric,
        hnsw_M=7,
    )

    hnsw = next(created for created in fake.created if created["kind"] == "hnsw")
    assert hnsw["metric"] == expected_metric
    assert fake.hnsw_M == 7


def test_knn_flat_faiss_matches_balltree_with_complete_tie_buffer():
    sequences = [
        BinarySequence([a, b, c])
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]
    balltree = create_knn_graph(
        sequences,
        k=2,
        backend="balltree",
        tie_policy="all",
        tiebuffer=len(sequences),
    )
    faiss = create_knn_graph(
        sequences,
        k=2,
        backend="faiss",
        index_type="flat",
        faiss_metric="ip",
        tie_policy="all",
        tiebuffer=len(sequences),
    )

    assert set(balltree.edges()) == set(faiss.edges())
    for u, v in balltree.edges():
        assert faiss[u][v]["distance"] == pytest.approx(
            balltree[u][v]["distance"]
        )
