import numpy as np
import pytest
import networkx as nx

from fitness_landscape.models import NKFitnessLandscape
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.transforms.walsh_hadamard import *
from fitness_landscape.transforms.graph_fourier import *


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
    landscape = NKFitnessLandscape(N=4, K=1, alphabet_size=2, seed=42)
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
    landscape = NKFitnessLandscape(N=3, K=0, alphabet_size=2, seed=42)
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
        graph.nodes[node]['fitness'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21)) # Dummy data
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20)) # Dummy data
    
    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    
    eigenvectors, _, coefficients = graph_fourier_transform(landscape)
    reconstructed_signal = inverse_graph_fourier_transform(eigenvectors, coefficients)
    
    assert np.allclose(signal, reconstructed_signal, atol=1e-9)

