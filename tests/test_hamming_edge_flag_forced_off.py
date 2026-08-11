import networkx as nx

import fitness_landscape.core.digraph as digraph_mod
import fitness_landscape.core.graph as graph_mod
from fitness_landscape._const import PROT_20
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence


def _fail_if_called(*args, **kwargs):
    raise AssertionError("Hamming edge annotation should be disabled.")


class _FakeASRConstructor:
    def __init__(self, *args, **kwargs):
        self.tip_names = {"tip"}

    def construct_dag(self, graph_type="undirected"):
        sequence = BaseNumpySequence.from_string(
            "AA",
            alphabet=PROT_20,
            moltype="protein",
            sequence_id="tip",
        )
        graph = nx.Graph() if graph_type == "undirected" else nx.DiGraph()
        graph.add_node("root", sequence=sequence)
        graph.add_node("tip", sequence=sequence)
        graph.add_edge("root", "tip")
        return graph


def test_create_hamming_graph_binary_ignores_hamming_edge_flag(monkeypatch):
    monkeypatch.setattr(graph_mod, "attach_expected_hamming_to_edges", _fail_if_called)

    sequences = [
        BinarySequence.from_bits([0, 0]),
        BinarySequence.from_bits([0, 1]),
    ]

    graph = graph_mod.create_hamming_graph_binary(sequences, _compute_hamming_edges=True)

    assert graph.number_of_edges() == 1


def test_create_phylo_graph_ignores_hamming_edge_flag(monkeypatch):
    monkeypatch.setattr(graph_mod, "ASRConstructor", _FakeASRConstructor)
    monkeypatch.setattr(graph_mod, "compute_edge_mutations_star", _fail_if_called)

    graph = graph_mod.create_phylo_graph("unused", _compute_hamming_edges=True)

    assert graph.number_of_edges() == 1


def test_create_phylo_digraph_ignores_hamming_edge_flag(monkeypatch):
    monkeypatch.setattr(digraph_mod, "ASRConstructor", _FakeASRConstructor)
    monkeypatch.setattr(digraph_mod, "compute_edge_mutations_star", _fail_if_called)

    digraph = digraph_mod.create_phylo_digraph("unused", _compute_hamming_edges=True)

    assert digraph.number_of_edges() == 1
