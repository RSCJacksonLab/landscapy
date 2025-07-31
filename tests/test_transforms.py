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

def test_walsh_transform_reconstruction():
    """
    Tests that a signal can be perfectly reconstructed via inverse
    Walsh transform.
    
    Raises
    ------
    AssertionError
        If the reconstructed signal does not match the original
        fitness signal.
    """
    landscape = create_nk_binary_landscape(N=4, K=1, seed=42)
    fitness_signal = landscape.get_signal()
    
    # Perform Walsh transform
    coeffs = walsh_transform(landscape)
    
    # Reconstruct the signal
    reconstructed_signal = inverse_walsh_transform(coeffs, sequences=landscape.sequences)
    
    assert np.allclose(fitness_signal, reconstructed_signal)

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

def test_graph_fourier_transform_reconstruction():
    """
    Tests that a signal can be perfectly reconstructed via inverse GFT.
    This test uses a cycle graph with a sine wave signal.
    """
    graph = nx.cycle_graph(8)
    signal = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    for i, node in enumerate(graph.nodes()):

        graph.nodes[node]['sequence'] = BaseNumpySequence([i]) 
        graph.nodes[node]['fitness_default'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21)) # Dummy data
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20)) # Dummy data
    
    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    
    eigenvectors, _, coefficients = graph_fourier_transform(landscape)
    reconstructed_signal = inverse_graph_fourier_transform(eigenvectors, coefficients)
    
    assert np.allclose(signal, reconstructed_signal, atol=1e-9)

def test_walsh_transform_torch():
    """Tests Walsh transform with torch backend."""
    landscape = create_nk_binary_landscape(N=4, K=1, seed=42)
    coeffs = walsh_transform(landscape, backend='torch')
    assert coeffs is not None

def test_inverse_walsh_transform_torch():
    """Tests inverse Walsh transform with torch backend."""
    landscape = create_nk_binary_landscape(N=4, K=1, seed=42)
    fitness_signal = landscape.get_signal()
    coeffs = walsh_transform(landscape, backend='torch')
    reconstructed_signal = inverse_walsh_transform(coeffs, sequences=landscape.sequences, backend='torch')
    assert np.allclose(fitness_signal, reconstructed_signal.numpy())

# New tests for graph_fourier.py
def test_graph_fourier_transform_torch():
    """Tests GFT with torch backend."""
    graph = nx.cycle_graph(8)
    signal = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['sequence'] = BaseNumpySequence([i])
        graph.nodes[node]['fitness_default'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21))
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20))

    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    eigenvectors, _, coefficients = graph_fourier_transform(landscape, backend='torch')
    assert eigenvectors is not None
    assert coefficients is not None

def test_inverse_graph_fourier_transform_torch():
    """Tests inverse GFT with torch backend."""
    graph = nx.cycle_graph(8)
    signal = np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    for i, node in enumerate(graph.nodes()):
        graph.nodes[node]['sequence'] = BaseNumpySequence([i])
        graph.nodes[node]['fitness_default'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21))
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20))

    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    eigenvectors, _, coefficients = graph_fourier_transform(landscape, backend='torch')
    reconstructed_signal = inverse_graph_fourier_transform(eigenvectors, coefficients, backend='torch')
    assert np.allclose(signal, reconstructed_signal.numpy(), atol=1e-6)
