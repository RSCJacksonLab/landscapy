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
    graph = create_hamming_graph(N=3)
    assert graph.number_of_nodes() == 8
    
    # Each of 8 nodes has 3 neighbors
    assert graph.number_of_edges() == 12  
    # Check degree of a node
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
        assert graph.degree[node] == k


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
    landscape = FitnessLandscape(sequences=sequences, fitness_values=fitnesses)
    assert len(landscape) == 8
    assert landscape.get_fitness(sequences[0]) == fitnesses[0]
    assert landscape.graph is not None
    
    # Test initialization from a pre-made graph
    graph = create_hamming_graph(N=3)
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['fitness'] = fitnesses[i]
        # Required attributes for from_graph initialization
        graph.nodes[node]['sequence'] = sequences[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21)) # Dummy data
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20)) # Dummy data

    landscape_from_graph = FitnessLandscape.from_graph(graph, emb_nodes=False)
    assert len(landscape_from_graph) == 8
    assert landscape_from_graph.get_fitness(sequences[0]) == fitnesses[0]
