import numpy as np
import pytest
import networkx as nx

from fitness_landscape.core.sequence import *
from fitness_landscape.core.graph import *
from fitness_landscape.core.landscape import *


def test_sequence_creation_and_distance():
    """
    Tests sequence creation and distance calculation.
    
    Raises
    ------
    AssertionError
        If the distance calculation does not match expected values.
    """
    seq1 = BinarySequence([0, 1, 0, 1])
    seq2 = BinarySequence([0, 1, 1, 0])
    assert seq1.distance(seq2, metric="hamming") == 2
    # Test with BaseNumpySequence
    seq3 = BaseNumpySequence(['A', 'C', 'G'])
    seq4 = BaseNumpySequence(['A', 'T', 'G'])
    assert sequence_distance(seq3, seq4) == 1


def test_sequence_mutation():
    """
    Tests the mutation method of sequence objects.
    
    Raises
    ------
    AssertionError
        If the mutated sequence does not match expected values or
        distance.
    """
    seq = BaseNumpySequence([0, 0, 0, 0], alphabet=[0, 1])
    mutated_seq = seq.mutate(positions=[1, 3], values=[1, 1])
    assert np.array_equal(mutated_seq.to_array(), [0, 1, 0, 1])
    assert seq.distance(mutated_seq) == 2


def test_generate_sequences():
    """
    Tests the generation of all combinatorial sequences.

    Raises
    ------
    AssertionError
        If the generated sequences do not match expected values.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    assert len(sequences) == 8  # 2^3
    # Check for a specific sequence
    assert BaseNumpySequence([1, 0, 1]) in sequences


def test_create_complete_hamming_graph():
    """
    Tests the creation of a complete Hamming graph.

    Raises
    ------
    AssertionError
        If the graph does not have the expected number of nodes or
        edges.
    """
    # First, generate the sequences for N=3
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    
    # Then, create the graph from the sequences
    graph = create_hamming_graph(sequences=sequences)
    
    assert graph.number_of_nodes() == 8
    assert graph.number_of_edges() == 12
    assert graph.degree[0] == 3


def test_create_knn_graph():
    """
    Tests the creation of a k-nearest neighbor graph.
    
    Raises
    ------
    AssertionError
        If the graph does not have the expected number of edges or
        if nodes do not have the expected degree.
    """
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    k = 3
    graph = create_knn_graph(sequences=sequences, k=k)
    # In a complete Hamming space, each node should have k neighbors
    for node in graph.nodes():
        
        # Equidistance connections are included thus the true degree is >= k (depedning on graph structure).
        assert graph.degree[node] >= k


def test_fitness_landscape_initialization():
    """
    Tests FitnessLandscape can be initialized correctly.
    
    Raises
    ------
    AssertionError
        If the landscape does not have the expected number of nodes or
        if fitness values do not match expected values.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    fitnesses = np.random.rand(8)

    # Test initialization from sequences and fitness values
    landscape = FitnessLandscape(sequences=sequences, fitness_values=fitnesses, emb_nodes=False)
    assert len(landscape) == 8
    assert landscape.get_fitness(sequences[0]) == fitnesses[0]
    assert landscape.graph is not None

    graph_sequences = generate_sequences(length=3, alphabet=[0, 1])
    graph = create_hamming_graph(sequences=graph_sequences) 
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['fitness'] = fitnesses[i]
        graph.nodes[node]['sequence'] = sequences[i]
        
        # Dummy data for the required attributes
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21))
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20))

    landscape_from_graph = FitnessLandscape.from_graph(graph, emb_nodes=False)
    assert len(landscape_from_graph) == 8
    assert landscape_from_graph.get_fitness(sequences[0]) == fitnesses[0]

def test_multiallele_sequence():
    """
    Tests the MultialleleSequence class.
    """
    alphabet = ['A', 'C', 'G', 'T']
    seq_data = ['A', 'G', 'T', 'C']
    seq = MultialleleSequence(seq_data, alphabet=alphabet)
    assert np.array_equal(seq.to_array(), seq_data)
    assert seq.alphabet == alphabet

    # Test that it raises an error for invalid characters
    with pytest.raises(ValueError):
        MultialleleSequence(['A', 'X', 'G', 'T'], alphabet=alphabet)
    
    seq2 = MultialleleSequence(['A', 'C', 'G', 'T'], alphabet=alphabet)
    assert seq.distance(seq2) == 3

def test_soft_sequence():
    """
    Tests the SoftSequence class.
    """
    alphabet = ['A', 'C', 'G', 'T']
    
    # Posterior probabilities for a sequence of length 3
    posterior = np.array([
        [0.8, 0.1, 0.05, 0.05], # Most likely A
        [0.1, 0.1, 0.7, 0.1],  # Most likely G
        [0.25, 0.25, 0.25, 0.25] # Completely uncertain
    ])

    # Test with argmax rule
    soft_seq_argmax = SoftSequence(posterior, alphabet=alphabet, hard_rule='argmax')
    expected_hard_seq = ['A', 'G', 'A'] # 'A' is the first max at index 2
    assert np.array_equal(soft_seq_argmax.to_array(), expected_hard_seq)

    rng = np.random.default_rng(42)
    soft_seq_sample = SoftSequence(posterior, alphabet=alphabet, hard_rule='sample', rng=rng)
    assert len(soft_seq_sample) == 3
    assert all(c in alphabet for c in soft_seq_sample.to_array())

    # Test map_values and entropy
    map_values = soft_seq_argmax.map_values()
    entropy = soft_seq_argmax.entropy()
    assert map_values.shape == (3,)
    assert entropy.shape == (3,)
    assert np.allclose(map_values, [0.8, 0.7, 0.25])
    assert entropy[0] > 0 and entropy[2] > entropy[0]