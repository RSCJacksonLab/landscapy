import importlib.util

import networkx as nx
from click.testing import CliRunner

import fitness_landscape
from fitness_landscape.__main__ import cli
from fitness_landscape.core.landscape import DirectedFitnessLandscape, FitnessLandscape
from fitness_landscape.core.sequence import BinarySequence


def test_feature_api_excludes_other_deferred_features():
    assert fitness_landscape.DirectedFitnessLandscape is DirectedFitnessLandscape
    assert not hasattr(fitness_landscape.analysis, "cross_spectral_coherence")
    assert not hasattr(fitness_landscape.analysis, "calculate_local_bottleneck")
    assert not hasattr(FitnessLandscape, "plot")

    assert importlib.util.find_spec("fitness_landscape.core.digraph") is not None
    assert importlib.util.find_spec("fitness_landscape.analysis.bottleneck") is None
    assert importlib.util.find_spec("fitness_landscape.analysis.coupling") is None
    visualization_spec = importlib.util.find_spec("fitness_landscape.visualization")
    assert visualization_spec is None or visualization_spec.loader is None


def test_directed_feature_accepts_directed_graphs():
    sequence = BinarySequence.from_bits([0, 1])
    graph = nx.DiGraph()
    graph.add_node(0, sequence=sequence)

    landscape = DirectedFitnessLandscape([sequence], graph)

    assert landscape.graph is graph


def test_directed_feature_registers_cli_commands():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "dilandscape" in result.output


def test_release_keeps_portable_export_api():
    assert callable(fitness_landscape.export_lsbundle)
    assert callable(fitness_landscape.save_bundle_dir)
    assert callable(fitness_landscape.load_bundle_dir)
