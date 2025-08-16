import numpy as np
import pytest
import networkx as nx

from fitness_landscape.core.sequence import BaseNumpySequence, generate_sequences
from fitness_landscape.models.nk import create_nk_binary_landscape
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.transforms.walsh_hadamard import *
from fitness_landscape.transforms.graph_fourier import *
from fitness_landscape.core.fitness import NumericFitness



@pytest.fixture
def diffusion_landscape():
    """Provides a basic FitnessLandscape for diffusion fourier transform testing."""
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    fitness_values = [[val] for val in np.random.rand(8)]
    fitness_layers = {
        'default': NumericFitness(name='default', values=fitness_values)
    }
    # Using a complete graph ensures the transition matrix is well-defined
    graph = nx.complete_graph(8)
    for i, seq in enumerate(sequences):
        graph.nodes[i]['sequence'] = seq

    return FitnessLandscape(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph=graph
    )

def test_walsh_coefficients_extraction():
    """
    Tests that Walsh coefficients are extracted and labeled correctly.
    
    Raises
    ------
    AssertionError
        If the coefficients dictionary does not contain expected keys
        or if high-order terms are not zero for a K=0 landscape.
    """
    landscape = create_nk_binary_landscape(N=4, K=0 , seed=42)
    coeffs_dict = walsh_coefficients(landscape, order=3)
    
    # Check for expected keys
    assert 'intercept' in coeffs_dict
    assert '0' in coeffs_dict
    assert '1,2' in coeffs_dict
    
    # For a K=0 landscape, high-order terms should be zero
    assert np.isclose(coeffs_dict['0,1,2'], 0.0)

def test_graph_fourier_transform_laplacian():
    """
    """
    graph = nx.cycle_graph(8)
    signal = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['sequence'] = BaseNumpySequence([i])
        graph.nodes[node]['fitness_default'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21))
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20))

    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    eigenvectors, _, coefficients = graph_fourier_transform(landscape)
    assert eigenvectors is not None
    assert coefficients is not None

def test_graph_fourier_transform_norm_laplacian():
    """
    """
    graph = nx.cycle_graph(8)
    signal = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['sequence'] = BaseNumpySequence([i])
        graph.nodes[node]['fitness_default'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21))
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20))

    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    eigenvectors, _, coefficients = graph_fourier_transform(landscape, matrix='norm_laplacian')
    assert eigenvectors is not None
    assert coefficients is not None

