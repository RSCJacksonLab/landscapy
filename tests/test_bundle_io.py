import json
import sys
from pathlib import Path
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pytest

from fitness_landscape.core.annotation import AnnotationLayer
from fitness_landscape.core.fitness import (
    CategoricalFitness,
    NumericFitness,
    ProbabilisticCategoricalFitness,
)
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.io import BundleValidationError, ChecksumMismatchError


ROOT = Path(__file__).resolve().parents[1]
LANDSCAPE_STORE_SRC = ROOT / "landscape-store" / "src"
if str(LANDSCAPE_STORE_SRC) not in sys.path:
    sys.path.insert(0, str(LANDSCAPE_STORE_SRC))

try:  # pragma: no cover - optional compatibility validation
    from landscape_store.serialization import load_bundle_object, validate_bundle
except Exception:  # pragma: no cover - optional dependency mismatch
    load_bundle_object = None
    validate_bundle = None


def _make_landscape() -> FitnessLandscape:
    alphabet = ["A", "B"]
    seq_a = BaseNumpySequence.from_string("AAA", alphabet=alphabet, sequence_id="seq-a")
    seq_b = BaseNumpySequence.from_string("AAB", alphabet=alphabet, sequence_id="seq-b")
    seq_c = BaseNumpySequence.from_string("ABB", alphabet=alphabet, sequence_id="seq-c")

    graph = nx.Graph()
    graph.add_node("node-b", sequence=seq_b)
    graph.add_node("node-a", sequence=seq_a)
    graph.add_node("node-c", sequence=seq_c)
    graph.add_edge("node-b", "node-a", weight=1.5, relation="near")
    graph.add_edge("node-a", "node-c", weight=2.5, mutations=["A2B"])

    numeric = NumericFitness.from_replicates(
        "score",
        [[3.0, 4.0], [1.0], [2.0, 5.0]],
        metadata={"units": "a.u."},
    )
    categorical = CategoricalFitness.from_values(
        "label",
        ["high", "low", "medium"],
        categories=["low", "medium", "high"],
        metadata={"palette": "example"},
    )
    probabilistic = ProbabilisticCategoricalFitness.from_probabilities(
        "posterior",
        np.array(
            [
                [0.1, 0.2, 0.7],
                [0.8, 0.1, 0.1],
                [0.2, 0.6, 0.2],
            ]
        ),
        categories=["low", "medium", "high"],
        metadata={"kind": "posterior"},
    )
    annotations = AnnotationLayer(
        "taxonomy",
        {
            "group": ["g3", "g2", "g1"],
            "source": ["paper-c", "paper-b", "paper-a"],
        },
        metadata={"curated": True},
    )
    embeddings = {
        "ohe": np.array(
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ]
        ),
        "plm": np.array(
            [
                [0.1, 0.2, 0.3],
                [1.1, 1.2, 1.3],
                [2.1, 2.2, 2.3],
            ]
        ),
    }

    landscape = FitnessLandscape(
        sequences=[seq_c, seq_b, seq_a],
        graph=graph,
        fitness_layers={
            "score": numeric,
            "label": categorical,
            "posterior": probabilistic,
        },
        annotation_layers={"taxonomy": annotations},
        embeddings=embeddings,
        active_embedding_domain="plm",
    )
    landscape.view("label")
    return landscape


def _node_embedding_map(landscape: FitnessLandscape, domain: str) -> dict[str, np.ndarray]:
    node_order = list(landscape.graph.nodes())
    matrix = np.asarray(landscape.embeddings[domain])
    return {node: matrix[idx] for idx, node in enumerate(node_order)}


def _read_dir_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_bundle_dir_round_trip_restores_portable_landscape(tmp_path: Path):
    landscape = _make_landscape()
    bundle_dir = tmp_path / "bundle"
    metadata = {
        "dataset_name": "example-dataset",
        "source_name": "synthetic",
        "protein_gene": "GENE1",
        "assay_type": "DMS",
        "organism": "human",
        "version": "v1",
        "tags": ["portable", "test"],
        "provenance": {"builder": "pytest"},
        "metadata": {"note": "round-trip"},
    }

    expected_plm = _node_embedding_map(landscape, "plm")
    expected_ohe = _node_embedding_map(landscape, "ohe")

    landscape.save_bundle_dir(bundle_dir, metadata=metadata)
    loaded = FitnessLandscape.load_bundle_dir(bundle_dir)

    assert isinstance(loaded, FitnessLandscape)
    assert loaded.active_layer_name == "label"
    assert loaded.active_embedding_domain == "plm"
    assert sorted(loaded.fitness_layers) == ["label", "posterior", "score"]
    assert sorted(loaded.annotation_layers) == ["taxonomy"]
    assert set(loaded.graph.nodes()) == set(landscape.graph.nodes())

    for node in loaded.graph.nodes():
        np.testing.assert_array_equal(
            loaded.graph.nodes[node]["sequence"].to_array(),
            landscape.graph.nodes[node]["sequence"].to_array(),
        )
        assert loaded.graph.nodes[node]["sequence"].id == landscape.graph.nodes[node]["sequence"].id
        assert loaded.graph.nodes[node]["fitness_score"] == landscape.graph.nodes[node]["fitness_score"]
        assert loaded.graph.nodes[node]["fitness_label"] == landscape.graph.nodes[node]["fitness_label"]
        loaded_posterior = loaded.graph.nodes[node]["fitness_posterior"]
        expected_posterior = landscape.graph.nodes[node]["fitness_posterior"]
        assert loaded_posterior.keys() == expected_posterior.keys()
        for category in loaded_posterior:
            assert loaded_posterior[category] == pytest.approx(expected_posterior[category])
        assert loaded.graph.nodes[node]["annotations"]["taxonomy"] == landscape.graph.nodes[node]["annotations"]["taxonomy"]

    for edge in loaded.graph.edges():
        assert loaded.graph.edges[edge] == landscape.graph.edges[edge]

    loaded_order = list(loaded.graph.nodes())
    for idx, node in enumerate(loaded_order):
        np.testing.assert_array_equal(loaded.embeddings["plm"][idx], expected_plm[node])
        np.testing.assert_array_equal(loaded.embeddings["ohe"][idx], expected_ohe[node])
    expected_numeric_scalars = {
        node: float(np.mean(landscape.graph.nodes[node]["fitness_score"]))
        for node in landscape.graph.nodes()
    }
    expected_labels = {
        node: landscape.graph.nodes[node]["fitness_label"] for node in landscape.graph.nodes()
    }
    for idx, node in enumerate(loaded_order):
        assert loaded.fitness_layers["score"].to_scalar()[idx] == pytest.approx(expected_numeric_scalars[node])
        assert loaded.fitness_layers["label"].get_value(idx) == expected_labels[node]
    assert loaded.fitness_layers["label"].categories == ["low", "medium", "high"]
    assert loaded.fitness_layers["posterior"].categories == ["low", "medium", "high"]

    assert getattr(loaded, "_bundle_metadata")["dataset_name"] == "example-dataset"


def test_bundle_dir_can_exclude_embeddings(tmp_path: Path):
    landscape = _make_landscape()
    bundle_dir = tmp_path / "bundle-no-emb"

    landscape.save_bundle_dir(bundle_dir, include_embeddings=False)
    loaded = FitnessLandscape.load_bundle_dir(bundle_dir)

    assert loaded.embeddings == {}
    assert loaded.active_embedding_domain is None


def test_bundle_dir_output_is_deterministic(tmp_path: Path):
    landscape = _make_landscape()
    bundle_a = tmp_path / "bundle-a"
    bundle_b = tmp_path / "bundle-b"
    metadata = {"dataset_name": "same", "protein_gene": "GENE1", "assay_type": "DMS", "version": "v1"}

    landscape.save_bundle_dir(bundle_a, metadata=metadata)
    landscape.save_bundle_dir(bundle_b, metadata=metadata)

    assert _read_dir_bytes(bundle_a) == _read_dir_bytes(bundle_b)


def test_bundle_dir_writes_optional_legacy_pickle(tmp_path: Path):
    landscape = _make_landscape()
    bundle_dir = tmp_path / "bundle-legacy"

    landscape.save_bundle_dir(bundle_dir, include_legacy_pickle=True)

    legacy_path = bundle_dir / "legacy" / "landscape.pkl"
    assert legacy_path.exists()

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["legacy_pickle"]["path"] == "legacy/landscape.pkl"


def test_bundle_dir_detects_checksum_failures(tmp_path: Path):
    landscape = _make_landscape()
    bundle_dir = tmp_path / "bundle"
    landscape.save_bundle_dir(bundle_dir)

    sequences_path = bundle_dir / "sequences.npy"
    sequences_path.write_bytes(sequences_path.read_bytes() + b"corruption")

    with pytest.raises(ChecksumMismatchError):
        FitnessLandscape.load_bundle_dir(bundle_dir)


def test_bundle_dir_rejects_malformed_manifest(tmp_path: Path):
    landscape = _make_landscape()
    bundle_dir = tmp_path / "bundle"
    landscape.save_bundle_dir(bundle_dir)

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("landscape_class")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleValidationError):
        FitnessLandscape.load_bundle_dir(bundle_dir)


def test_portable_lsbundle_export_is_created_and_deterministic(tmp_path: Path):
    landscape = _make_landscape()
    bundle_a = tmp_path / "portable-a.lsbundle"
    bundle_b = tmp_path / "portable-b.lsbundle"

    landscape.export_lsbundle(bundle_a, metadata={"dataset_name": "portable"})
    landscape.export_lsbundle(bundle_b, metadata={"dataset_name": "portable"})

    assert bundle_a.read_bytes() == bundle_b.read_bytes()

    with ZipFile(bundle_a, "r") as archive:
        names = sorted(archive.namelist())
        assert "manifest.json" in names
        assert "metadata.json" in names
        assert "sequences.npy" in names
        assert "graph_edges.parquet" in names
        assert any(name.startswith("layers/") for name in names)


@pytest.mark.skipif(validate_bundle is None or load_bundle_object is None, reason="landscape-store v1 is not importable")
def test_pickle_lsbundle_export_matches_landscape_store_v1(tmp_path: Path):
    landscape = _make_landscape()
    bundle_path = tmp_path / "compat.lsbundle"
    metadata = {
        "landscape_id": "compat-1",
        "dataset_name": "compat-dataset",
        "source_name": "compat-source",
        "protein_gene": "GENE1",
        "assay_type": "DMS",
        "organism": "human",
        "version": "v1",
        "tags": ["compat"],
        "metadata": {"note": "pickle"},
        "provenance": {"kind": "pytest"},
    }

    landscape.export_lsbundle(bundle_path, metadata=metadata, backend="pickle")

    validated = validate_bundle(bundle_path)
    assert validated.manifest.serialization_backend == "pickle"
    assert validated.metadata.landscape_id == "compat-1"
    assert validated.metadata.default_active_layer == "label"

    loaded = load_bundle_object(
        bundle_path,
        allow_unsafe_pickle=True,
        enforce_package_versions=False,
    )
    assert isinstance(loaded, FitnessLandscape)
    assert loaded.active_layer_name == "label"
