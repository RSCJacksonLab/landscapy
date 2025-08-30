import numpy as np

from fitness_landscape.core.superscape import FitnessSuperscape
from fitness_landscape.core.sequence import generate_sequences
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.core.landscape import FitnessLandscape


def test_superscape_to_graph_and_sequence_tensors():
    # Single-graph fast path
    seqs = generate_sequences(length=2, alphabet=[0, 1])
    layers = {"default": NumericFitness(name="default", values=[[float(i)] for i in range(len(seqs))])}
    L = FitnessLandscape.from_sequences(seqs, fitness_layers=layers, graph_type='hamming')
    ss = FitnessSuperscape([L], burn_in=1, samples=1)

    Gt = ss.to_graph_tensor()
    # PyG Data object
    from torch_geometric.data import Data
    assert isinstance(Gt, Data)
    # Node features exist and match node count
    assert hasattr(Gt, 'x')
    assert Gt.x.shape[0] == len(seqs)

    st = ss.to_sequence_tensors()
    assert isinstance(st, list) and len(st) == len(seqs)
    # each item has sequence_tensor and fitness_tensors
    assert 'sequence_tensor' in st[0]
    assert 'fitness_tensors' in st[0]
