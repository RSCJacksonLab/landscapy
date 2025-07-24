import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch

from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.core.superscape import FitnessSuperscape

AMINO_ACID_ALPHABET = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

@pytest.fixture
def simple_landscape_factory():
    """
    Creates a simple but valid fitness landscape using a 20-amino-acid
    alphabet.
    """
    def _factory():
        alphabet = AMINO_ACID_ALPHABET
        alphabet_size = len(alphabet)
        seq_length = 4

        seq1_list = np.random.choice(alphabet, seq_length).tolist()
        seq1 = BaseNumpySequence(seq1_list, alphabet=alphabet)
        seq2 = seq1.mutate(positions=[1])
        sequences = [seq1, seq2]
        fitnesses = [0.5, 0.8]
        
        landscape = FitnessLandscape(sequences=sequences, fitness_values=fitnesses, graph_type='hamming')
        
        for i, node in enumerate(landscape.graph.nodes()):
            landscape.graph.nodes[node]['emb_arr'] = np.random.rand(10)
            landscape.graph.nodes[node]['ungapped_arr'] = np.random.rand(seq_length, alphabet_size)
            landscape.graph.nodes[node]['gapped_arr'] = np.random.rand(seq_length, alphabet_size + 1)
            
        return landscape
    return _factory

@pytest.fixture
def two_landscapes(simple_landscape_factory):
    """
    Provides two distinct landscape instances for testing.

    Raises
    ------
    AssertionError
        If the landscapes are not distinct.
    """
    return [simple_landscape_factory(), simple_landscape_factory()]

@patch('fitness_landscape.core.superscape.RJMCMCAligner')
def test_superscape_initialization(MockRJMCMCAligner, two_landscapes):
    """
    Tests that FitnessSuperscape initializes correctly with a full
    alphabet.

    Raises
    ------
    AssertionError
        If the RJMCMCAligner is not called with the expected
        parameters.
    """
    mock_aligner_instance = MockRJMCMCAligner.return_value
    mock_aligner_instance.sample.return_value = None
    
    superscape = FitnessSuperscape(two_landscapes, alpha=0.7)
    
    MockRJMCMCAligner.assert_called_once()
    assert superscape.alphabet == AMINO_ACID_ALPHABET

@patch('fitness_landscape.core.superscape.RJMCMCAligner')
@patch('fitness_landscape.core.superscape.align_soft_sequences')
def test_construct_latent_landscape_with_full_alphabet(mock_align_soft,
                                                       MockRJMCMCAligner,
                                                       two_landscapes):
    """
    Tests the latent landscape construction logic with a full alphabet.

    Raises
    ------
    AssertionError
        If the latent landscape is not constructed correctly or if the
        latent graph does not have the expected properties.
    """
    mock_aligner_instance = MockRJMCMCAligner.return_value
    
    latent_graph = nx.Graph()
    latent_graph.add_edge(0, 1)
    mock_aligner_instance.latent_blueprint_graph.return_value = latent_graph
    
    mock_mappings = {
        0: np.array([[0.9, 0.1], [0.2, 0.8]]),
        1: np.array([[0.95, 0.05], [0.15, 0.85]])
    }
    mock_aligner_instance.get_node_to_latent_mapping.return_value = mock_mappings
    
    def dynamic_align_side_effect(sequences, alphabet):
        num_sequences = len(sequences)
        aligned_length = 5
        alphabet_size = len(alphabet)
        aligned_sequences = [np.random.rand(aligned_length, alphabet_size + 1) for _ in range(num_sequences)]
        return (aligned_sequences, -10.0)
    
    mock_align_soft.side_effect = dynamic_align_side_effect

    superscape = FitnessSuperscape(two_landscapes)
    superscape.construct_latent_landscape()

    assert hasattr(superscape, 'latent_landscape')
    assert superscape.latent_graph.number_of_nodes() == 2
    
    latent_node_0_data = superscape.latent_graph.nodes[0]
    assert 'gapped_arr' in latent_node_0_data
    assert latent_node_0_data['gapped_arr'].shape == (5, len(AMINO_ACID_ALPHABET) + 1)