from __future__ import annotations

from contextlib import contextmanager

from click.testing import CliRunner
import networkx as nx
import numpy as np
import pytest

import fitness_landscape.__main__ as cli_module
import fitness_landscape.core.graph as graph_module
import fitness_landscape.core.landscape as landscape_module
from fitness_landscape._const import PROT_20
from fitness_landscape.core.edge_schema import EDGE_SCHEMA_GRAPH_KEY
from fitness_landscape.core.graph import (
    create_diffusion_emb_graph,
    create_evol_diffusion_graph,
    create_knn_graph,
)
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence


def _protein(text: str, sequence_id: str) -> BaseNumpySequence:
    return BaseNumpySequence.from_string(
        text,
        alphabet=PROT_20,
        sequence_id=sequence_id,
    )


def _sequences() -> list[BaseNumpySequence]:
    return [
        _protein("AAAA", "a"),
        _protein("AAAR", "b"),
        _protein("RRRR", "c"),
    ]


def _plm_embeddings() -> np.ndarray:
    # Embedding neighbours differ deliberately from sequence Hamming neighbours.
    return np.array([[0.0, 0.0], [10.0, 0.0], [0.1, 0.0]])


def _edges(graph: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in graph.edges()}


def test_knn_balltree_uses_euclidean_plm_geometry_and_declares_units():
    sequences = _sequences()
    embeddings = _plm_embeddings()

    hamming = create_knn_graph(
        sequences,
        k=1,
        backend="balltree",
        tiebuffer=0,
        tie_policy="min_index",
    )
    plm = create_knn_graph(
        sequences,
        k=1,
        embeddings=embeddings,
        embedding_domain="plm",
        backend="balltree",
        tiebuffer=0,
        tie_policy="min_index",
    )

    assert _edges(hamming) == {(0, 1), (1, 2)}
    assert _edges(plm) == {(0, 2), (1, 2)}
    assert plm[0][2]["distance"] == pytest.approx(0.1)
    assert plm[0][2]["weight"] == pytest.approx(np.exp(-0.1))
    assert "normalized_distance" not in plm[0][2]
    assert plm.graph[EDGE_SCHEMA_GRAPH_KEY]["distance"]["units"] == "embedding_euclidean"
    assert plm.graph["landscapy_knn_search"] == {
        "role": "graph",
        "backend": "balltree",
        "metric": "euclidean",
        "distance_geometry": "euclidean",
        "embedding_domain": "plm",
    }


def test_knn_faiss_uses_plm_matrix_l2_and_converts_squared_distance(monkeypatch):
    embeddings = _plm_embeddings()
    captured = {}

    def fake_find(features, k, **kwargs):
        captured["features"] = np.asarray(features).copy()
        captured["metric"] = kwargs["metric"]
        squared = np.square(features[:, None, :] - features[None, :, :]).sum(axis=2)
        indices = np.argsort(squared, axis=1, kind="stable")
        return np.take_along_axis(squared, indices, axis=1), indices

    monkeypatch.setattr(graph_module, "_find_knn_faiss", fake_find)
    graph = create_knn_graph(
        _sequences(),
        k=1,
        embeddings=embeddings,
        embedding_domain="plm",
        backend="faiss",
        index_type="flat",
        faiss_metric="ip",
        tiebuffer=0,
        tie_policy="min_index",
    )

    assert np.array_equal(captured["features"], embeddings)
    assert captured["metric"] == "l2"
    assert graph[0][2]["distance"] == pytest.approx(0.1)
    assert graph.graph["landscapy_knn_search"]["metric"] == "l2"


def test_plm_knn_accepts_variable_length_sequences_and_requires_embeddings():
    sequences = [
        _protein("AA", "short"),
        _protein("AAAA", "medium"),
        _protein("AAAAAAAA", "long"),
    ]
    embeddings = np.array([[0.0], [0.2], [2.0]])

    graph = create_knn_graph(
        sequences,
        k=1,
        embeddings=embeddings,
        embedding_domain="plm",
        backend="balltree",
        tiebuffer=0,
        tie_policy="min_index",
    )
    assert graph.number_of_nodes() == 3

    with pytest.raises(ValueError, match="requires an aligned embedding matrix"):
        create_knn_graph(
            sequences,
            k=1,
            embedding_domain="plm",
            backend="balltree",
        )


def test_fitness_landscape_build_passes_selected_plm_domain_to_knn(monkeypatch):
    captured = {}

    def fake_knn(sequences, **kwargs):
        captured.update(kwargs)
        graph = nx.Graph()
        graph.add_nodes_from(
            (index, {"sequence": sequence})
            for index, sequence in enumerate(sequences)
        )
        return graph

    monkeypatch.setattr(landscape_module, "create_knn_graph", fake_knn)
    embeddings = _plm_embeddings()
    landscape = FitnessLandscape.build(
        _sequences(),
        graph="knn",
        embeddings={"plm": embeddings},
        embedding_domain="plm",
        attach_embeddings=False,
        k=1,
    )

    assert captured["embeddings"] is embeddings
    assert captured["embedding_domain"] == "plm"
    assert np.array_equal(landscape.embeddings["plm"], embeddings)


def test_fitness_landscape_build_computes_plm_required_by_knn(monkeypatch):
    embeddings = _plm_embeddings()
    captured = {}

    monkeypatch.setattr(
        landscape_module,
        "_compute_embeddings_from_sequences",
        lambda *args, **kwargs: embeddings,
    )

    def fake_knn(sequences, **kwargs):
        captured.update(kwargs)
        graph = nx.Graph()
        graph.add_nodes_from(
            (index, {"sequence": sequence})
            for index, sequence in enumerate(sequences)
        )
        return graph

    monkeypatch.setattr(landscape_module, "create_knn_graph", fake_knn)
    landscape = FitnessLandscape.build(
        _sequences(),
        graph="knn",
        embedding_domain="plm",
        attach_embeddings=True,
        k=1,
    )

    assert captured["embeddings"] is embeddings
    assert captured["embedding_domain"] == "plm"
    assert landscape.active_embedding_domain == "plm"


def test_knn_cli_passes_loaded_plm_embeddings_to_constructor(tmp_path, monkeypatch):
    sequence_path = tmp_path / "proteins.fasta"
    sequence_path.write_text(">placeholder\nA\n")
    embedding_path = tmp_path / "plm.npy"
    output_path = tmp_path / "knn.pkl"
    embeddings = _plm_embeddings()
    np.save(embedding_path, embeddings)
    captured = {}

    monkeypatch.setattr(
        cli_module,
        "fasta_to_prot20_sequences",
        lambda *args, **kwargs: _sequences(),
    )

    def fake_knn(*, sequences, embeddings, embedding_domain, **kwargs):
        captured["embeddings"] = embeddings
        captured["embedding_domain"] = embedding_domain
        graph = nx.Graph()
        graph.add_nodes_from(
            (index, {"sequence": sequence})
            for index, sequence in enumerate(sequences)
        )
        return graph

    monkeypatch.setattr(cli_module, "create_knn_graph", fake_knn)
    result = CliRunner().invoke(
        cli_module.cli,
        [
            "knn-landscape",
            "--sequences",
            str(sequence_path),
            "--output",
            str(output_path),
            "--k",
            "1",
            "--backend",
            "balltree",
            "--embedding-domain",
            "plm",
            "--embeddings-in",
            str(embedding_path),
            "--no-compute-embeddings",
            "--no-compute-hamming-edges",
        ],
    )

    assert result.exit_code == 0, result.output
    assert np.array_equal(captured["embeddings"], embeddings)
    assert captured["embedding_domain"] == "plm"
    assert output_path.exists()


def test_diffusion_prefilter_uses_euclidean_plm_embeddings(monkeypatch):
    embeddings = _plm_embeddings()
    captured = {}

    def fake_find(features, k, tiebuffer=0, **kwargs):
        captured["features"] = np.asarray(features).copy()
        captured["metric"] = kwargs["metric"]
        indices = np.array([[0, 2], [1, 2], [2, 0]])
        return np.zeros_like(indices, dtype=float), indices

    monkeypatch.setattr(graph_module, "_find_knn_balltree", fake_find)
    graph = create_diffusion_emb_graph(
        _sequences(),
        embeddings=embeddings,
        embedding_domain="plm",
        backend="balltree",
        k=1,
        t=1,
        connectivity_threshold=0.0,
    )

    assert np.array_equal(captured["features"], embeddings)
    assert captured["metric"] == "euclidean"
    assert graph.graph["landscapy_knn_search"]["role"] == "prefilter"
    assert graph.graph["landscapy_knn_search"]["embedding_domain"] == "plm"


class _ImmediateRemote:
    def __init__(self, function):
        self.function = function

    def options(self, **kwargs):
        return self

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _ImmediateRay:
    @staticmethod
    def remote(function):
        return _ImmediateRemote(function)

    @staticmethod
    def wait(pending, num_returns):
        return pending[:num_returns], pending[num_returns:]

    @staticmethod
    def get(ready):
        return ready


@contextmanager
def _immediate_ray_runtime(*args, **kwargs):
    yield _ImmediateRay()


def test_evolutionary_diffusion_prefilter_uses_plm_l2_with_faiss(monkeypatch):
    embeddings = _plm_embeddings()
    captured = {}

    def fake_find(features, k, **kwargs):
        captured["features"] = np.asarray(features).copy()
        captured["metric"] = kwargs["metric"]
        indices = np.array([[0, 2], [1, 2], [2, 0]])
        return np.zeros_like(indices, dtype=float), indices

    monkeypatch.setattr(graph_module, "_find_knn_faiss", fake_find)
    monkeypatch.setattr(graph_module, "ray_runtime", _immediate_ray_runtime)
    graph = create_evol_diffusion_graph(
        _sequences(),
        embeddings=embeddings,
        embedding_domain="plm",
        backend="faiss",
        faiss_metric="ip",
        k=1,
        t=1,
        connectivity_threshold=0.0,
        cpus=1,
    )

    assert np.array_equal(captured["features"], embeddings)
    assert captured["metric"] == "l2"
    assert graph.graph["landscapy_knn_search"] == {
        "role": "prefilter",
        "backend": "faiss",
        "metric": "l2",
        "distance_geometry": "euclidean",
        "embedding_domain": "plm",
    }
