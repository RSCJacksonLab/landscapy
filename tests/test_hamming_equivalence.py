import networkx as nx
import numpy as np

from fitness_landscape.core.sequence import BaseNumpySequence, BinarySequence
from fitness_landscape.core.graph import (
    create_hamming_graph,
    create_hamming_graph_binary,
    create_hamming_graph_multiallele,
)


def _edges_as_set(G: nx.Graph):
    return {tuple(sorted(e)) for e in G.edges()}

def _brute_hamming_edges(seqs):
    X = np.stack([s.to_array() for s in seqs])
    edges = set()

    for i in range(len(seqs)):
        dists = (X[i + 1:] != X[i]).sum(axis=1)
        for offset in np.flatnonzero(dists == 1):
            edges.add((i, i + 1 + int(offset)))

    return edges


def test_hamming_binary_xor_equals_masked_on_binary_with_duplicates():
    # Construct a small binary dataset with intentional duplicates
    bits = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],  # duplicate of 0
        [0, 0, 0, 1],
        [0, 0, 0, 1],  # duplicate of 2
        [0, 0, 1, 1],
    ]
    seqs = [BinarySequence.from_bits(b) for b in bits]

    G_xor = create_hamming_graph_binary(seqs, _compute_hamming_edges=False)
    G_mask = create_hamming_graph_multiallele(seqs, _compute_hamming_edges=False)

    # Basic node checks
    assert G_xor.number_of_nodes() == len(seqs)
    assert G_mask.number_of_nodes() == len(seqs)

    # Edge set equality
    assert _edges_as_set(G_xor) == _edges_as_set(G_mask)


def test_hamming_dispatch_backend_equivalence_on_binary():
    rng = np.random.default_rng(42)
    bits = (rng.random((20, 8)) < 0.5).astype(int).tolist()
    # Introduce some duplicates deliberately
    bits += bits[:3]
    seqs = [BinarySequence.from_bits(b) for b in bits]

    G_auto = create_hamming_graph(seqs, _backend='auto', _compute_hamming_edges=False)
    G_xor = create_hamming_graph(seqs, _backend='binary_xor', _compute_hamming_edges=False)

    assert _edges_as_set(G_auto) == _edges_as_set(G_xor)


def test_hamming_multiallele_matches_bruteforce_for_long_sequences():
    alphabet = list("ABCDEFGHIJKLMNOPQRST")
    wt = "".join(alphabet[i % len(alphabet)] for i in range(174))
    seqs = [wt]

    # This shape used to trigger dense false-positive graphs via int64 overflow
    # in the radix-key masked backend.
    for pos in range(33):
        wt_aa = wt[pos]
        for aa in [sym for sym in alphabet if sym != wt_aa][:3]:
            mutant = list(wt)
            mutant[pos] = aa
            seqs.append("".join(mutant))

    sequences = [BaseNumpySequence(seq) for seq in seqs]

    G = create_hamming_graph_multiallele(sequences, _compute_hamming_edges=False)

    assert _edges_as_set(G) == _brute_hamming_edges(sequences)
