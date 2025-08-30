import pickle
import numpy as np
import networkx as nx

from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import generate_sequences
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.superscape import FitnessSuperscape


def _make_small_landscape():
    seqs = generate_sequences(length=2, alphabet=[0, 1])
    vals = [[float(i)] for i in range(len(seqs))]
    layers = {"default": NumericFitness(name="default", values=vals)}
    return FitnessLandscape.from_sequences(seqs, fitness_layers=layers, graph_type="hamming")


def test_single_graph_fastpath_identity_mapping():
    L = _make_small_landscape()
    ss = FitnessSuperscape([L], burn_in=1, samples=1)

    # latent graph equals original (up to node relabel order)
    assert isinstance(ss.latent_graph, nx.Graph)
    assert ss.latent_graph.number_of_nodes() == L.graph.number_of_nodes()
    assert ss.latent_graph.number_of_edges() == L.graph.number_of_edges()

    # mappings is identity
    M = ss._latent_mappings[0]
    assert M.shape == (L.graph.number_of_nodes(), L.graph.number_of_nodes())
    assert np.allclose(M, np.eye(M.shape[0]))


def test_single_graph_fastpath_picklable(tmp_path):
    L = _make_small_landscape()
    ss = FitnessSuperscape([L], burn_in=1, samples=1)

    out = tmp_path / "ss.pkl"
    with open(out, "wb") as f:
        pickle.dump(ss, f)
    with open(out, "rb") as f:
        ss2 = pickle.load(f)

    assert isinstance(ss2, FitnessSuperscape)
    assert ss2.latent_graph.number_of_nodes() == ss.latent_graph.number_of_nodes()

