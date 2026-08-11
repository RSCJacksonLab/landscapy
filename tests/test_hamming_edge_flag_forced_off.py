import networkx as nx

import fitness_landscape.core.graph as graph_mod
import fitness_landscape.phylo.phylogenetic_asr as phylo_mod
from fitness_landscape._const import PROT_20
from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence


class _FakeASRConstructor:
    def __init__(self, *args, **kwargs):
        self.tip_names = {"tip"}

    def construct_topology(self):
        sequence = BaseNumpySequence.from_string(
            "AA",
            alphabet=PROT_20,
            moltype="protein",
            sequence_id="tip",
        )
        graph = nx.Graph()
        graph.add_node("root", sequence=sequence)
        graph.add_node("tip", sequence=sequence)
        graph.add_edge("root", "tip")
        return graph


def test_create_hamming_graph_binary_exposes_hamming_attributes_when_flagged():
    sequences = [
        BinarySequence.from_bits([0, 0]),
        BinarySequence.from_bits([0, 1]),
    ]

    graph = graph_mod.create_hamming_graph_binary(sequences, _compute_hamming_edges=True)

    assert graph.number_of_edges() == 1
    edge = next(iter(graph.edges(data=True)))[2]
    assert edge["distance"] == 1.0
    assert edge["normalized_distance"] == 0.5


def test_create_phylo_graph_honors_hamming_edge_flag(monkeypatch):
    monkeypatch.setattr(phylo_mod, "ASRConstructor", _FakeASRConstructor)
    called = []

    def _annotate(graph, **kwargs):
        called.append(True)
        for u, v in graph.edges():
            graph[u][v]["hamming_distance"] = 0.0
            graph[u][v]["normalized_distance"] = 0.0

    monkeypatch.setattr(graph_mod, "_annotate_existing_edges_hamming", _annotate)

    graph = graph_mod.create_phylo_graph("unused", _compute_hamming_edges=True)

    assert graph.number_of_edges() == 1
    assert called == [True]
    assert graph["root"]["tip"]["hamming_distance"] == 0.0
