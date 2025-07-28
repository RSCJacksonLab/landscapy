import numpy as np
import pytest
import networkx as nx
from fitness_landscape.core.sequence import *
from fitness_landscape.core.graph import *
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.fitness import NumericFitness, CategoricalFitness


def test_sequence_creation_and_distance():
    """
    Tests sequence creation and distance calculation.
    """
    seq1 = BinarySequence([0, 1, 0, 1])
    seq2 = BinarySequence([0, 1, 1, 0])
    assert seq1.distance(seq2, metric="hamming") == 2
    seq3 = BaseNumpySequence(['A', 'C', 'G'])
    seq4 = BaseNumpySequence(['A', 'T', 'G'])
    assert sequence_distance(seq3, seq4) == 1

def test_sequence_mutation():
    """
    Tests the mutation method of sequence objects.
    """
    seq = BaseNumpySequence([0, 0, 0, 0], alphabet=[0, 1])
    mutated_seq = seq.mutate(positions=[1, 3], values=[1, 1])
    assert np.array_equal(mutated_seq.to_array(), [0, 1, 0, 1])
    assert seq.distance(mutated_seq) == 2

def test_generate_sequences():
    """
    Tests the generation of all combinatorial sequences.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    assert len(sequences) == 8
    assert BaseNumpySequence([1, 0, 1]) in sequences

def test_multiallele_sequence():
    """
    Tests the MultialleleSequence class.
    """
    alphabet = ['A', 'C', 'G', 'T']
    seq_data = ['A', 'G', 'T', 'C']
    seq = MultialleleSequence(seq_data, alphabet=alphabet)
    assert np.array_equal(seq.to_array(), seq_data)
    assert seq.alphabet == alphabet
    with pytest.raises(ValueError):
        MultialleleSequence(['A', 'X', 'G', 'T'], alphabet=alphabet)
    seq2 = MultialleleSequence(['A', 'C', 'G', 'T'], alphabet=alphabet)
    assert seq.distance(seq2) == 3

def test_soft_sequence():
    """
    Tests the SoftSequence class.
    """
    alphabet = ['A', 'C', 'G', 'T']
    posterior = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.7, 0.1],
        [0.25, 0.25, 0.25, 0.25]
    ])
    soft_seq_argmax = SoftSequence(posterior, alphabet=alphabet, hard_rule='argmax')
    expected_hard_seq = ['A', 'G', 'A']
    assert np.array_equal(soft_seq_argmax.to_array(), expected_hard_seq)
    map_values = soft_seq_argmax.map_values()
    assert np.allclose(map_values, [0.8, 0.7, 0.25])

def test_create_complete_hamming_graph():
    """
    Tests the creation of a complete Hamming graph.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    graph = create_hamming_graph(sequences=sequences)
    assert graph.number_of_nodes() == 8
    assert graph.number_of_edges() == 12
    assert graph.degree[0] == 3

def test_create_knn_graph():
    """
    Tests the creation of a k-nearest neighbor graph.
    """
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    k = 3
    graph = create_knn_graph(sequences=sequences, k=k)
    for node in graph.nodes():
        assert graph.degree[node] >= k

@pytest.fixture
def basic_landscape():
    """Provides a basic FitnessLandscape with a numeric layer for testing."""
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    fitness_values = [[val] for val in np.random.rand(8)]
    fitness_layers = {
        'default': NumericFitness(name='default', values=fitness_values)
    }
    return FitnessLandscape(sequences=sequences, fitness_layers=fitness_layers)

def test_landscape_initialization_with_layers(basic_landscape):
    """
    Tests that FitnessLandscape initializes correctly with the layer
    system.
    """
    assert len(basic_landscape) == 8
    assert 'default' in basic_landscape.fitness_layers
    assert basic_landscape.get_signal().shape == (8,)
    assert basic_landscape.graph is not None

def test_fitness_free_landscape_initialization():
    """
    Tests that a landscape can be initialized without any fitness layers.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    landscape = FitnessLandscape(sequences=sequences, fitness_layers={})
    assert len(landscape) == 8
    assert landscape.graph.number_of_nodes() == 8
    with pytest.raises(ValueError):
        landscape.get_signal()

def test_attach_and_detach_layer(basic_landscape):
    """
    Tests that a new fitness layer can be attached and detached.
    """
    cat_values = ['A'] * 4 + ['B'] * 4
    new_layer = CategoricalFitness(name='activity', values=cat_values, categories=['A', 'B'])
    basic_landscape.attach(new_layer)
    
    assert 'activity' in basic_landscape.fitness_layers
    assert 'fitness_activity' in basic_landscape.graph.nodes[0]
    assert basic_landscape.graph.nodes[0]['fitness_activity'] == 'A'
    
    basic_landscape.detach('activity')
    
    assert 'activity' not in basic_landscape.fitness_layers
    assert 'fitness_activity' not in basic_landscape.graph.nodes[0]

def test_from_graph_initialization():
    """
    Tests that a landscape can be correctly initialized from a graph.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    graph = create_hamming_graph(sequences=sequences)
    
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['fitness_stability'] = [np.random.rand()]
        graph.nodes[node]['fitness_activity'] = 'high' if i % 2 == 0 else 'low'

    landscape = FitnessLandscape.from_graph(graph)

    assert len(landscape) == 8
    assert 'stability' in landscape.fitness_layers
    assert 'activity' in landscape.fitness_layers
    assert isinstance(landscape.view('stability'), NumericFitness)
    assert isinstance(landscape.view('activity'), CategoricalFitness)