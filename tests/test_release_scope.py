import inspect
import importlib.util
import json

import networkx as nx
import pytest
from click.testing import CliRunner

import fitness_landscape
from fitness_landscape.__main__ import cli
from fitness_landscape.core.graph import create_phylo_graph
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BinarySequence
from fitness_landscape.io import BundleValidationError
from fitness_landscape.phylo import ASRConstructor
from fitness_landscape.utils import geodesic_distance_matrix


def test_release_api_excludes_deferred_features():
    assert not hasattr(fitness_landscape, "DirectedFitnessLandscape")
    assert not hasattr(fitness_landscape.analysis, "cross_spectral_coherence")
    assert not hasattr(fitness_landscape.analysis, "calculate_local_bottleneck")
    assert not hasattr(FitnessLandscape, "plot")

    assert importlib.util.find_spec("fitness_landscape.core.digraph") is None
    assert importlib.util.find_spec("fitness_landscape.analysis.bottleneck") is None
    assert importlib.util.find_spec("fitness_landscape.analysis.coupling") is None
    visualization_spec = importlib.util.find_spec("fitness_landscape.visualization")
    assert visualization_spec is None or visualization_spec.loader is None


def test_release_landscape_rejects_directed_graphs():
    sequence = BinarySequence.from_bits([0, 1])
    graph = nx.DiGraph()
    graph.add_node(0, sequence=sequence)

    with pytest.raises(TypeError, match="undirected"):
        FitnessLandscape([sequence], graph)
    with pytest.raises(TypeError, match="undirected"):
        geodesic_distance_matrix(graph)


def test_release_cli_excludes_directed_commands():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "dilandscape" not in result.output


def test_release_keeps_portable_export_api():
    assert callable(fitness_landscape.export_lsbundle)
    assert callable(fitness_landscape.save_bundle_dir)
    assert callable(fitness_landscape.load_bundle_dir)


def test_phylogenetic_api_only_constructs_undirected_topologies():
    assert not hasattr(ASRConstructor, "construct_dag")
    signature = inspect.signature(ASRConstructor.construct_topology)
    assert list(signature.parameters) == ["self"]
    assert signature.return_annotation is nx.Graph
    assert "directed acyclic" not in inspect.getdoc(
        ASRConstructor.construct_topology
    ).lower()
    assert "DiGraph" not in str(inspect.signature(create_phylo_graph))


def test_portable_bundle_manifest_has_no_directed_schema_path(tmp_path):
    sequences = [
        BinarySequence.from_bits([0, 0]),
        BinarySequence.from_bits([0, 1]),
    ]
    landscape = FitnessLandscape.build(sequences, graph="hamming")
    bundle = tmp_path / "bundle"

    landscape.save_bundle_dir(bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "graph_directed" not in manifest
    assert "graph_class" not in manifest
    assert not FitnessLandscape.load_bundle_dir(bundle).graph.is_directed()

    manifest["graph_directed"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="Directed graph bundles"):
        FitnessLandscape.load_bundle_dir(bundle)


def test_portable_bundle_writer_rejects_mutated_directed_state(tmp_path):
    sequences = [
        BinarySequence.from_bits([0, 0]),
        BinarySequence.from_bits([0, 1]),
    ]
    landscape = FitnessLandscape.build(sequences, graph="hamming")
    landscape.graph = nx.DiGraph(landscape.graph)

    with pytest.raises(BundleValidationError, match="undirected"):
        landscape.save_bundle_dir(tmp_path / "directed")
