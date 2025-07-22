import numpy as np
import pytest
import networkx as nx

from fitness_landscape.models.nk import *
from fitness_landscape.models.rmf import *
from fitness_landscape.models.elementary_landscape import *
from fitness_landscape.core.sequence import generate_sequences
from fitness_landscape.analysis.epistasis import calculate_epistasis_walsh
from fitness_landscape.analysis.eigenmode import eigenmode_decomposition


def test_nk_landscape_additive_case():
    """
    Tests that an NK landscape with K=0 is purely additive.
    
    Raises
    ------
    AssertionError
        If the fitness values do not match expected additive values.
    """
    landscape = NKFitnessLandscape(N=4, K=0, alphabet_size=2, seed=42)
    epistasis_results = calculate_epistasis_walsh(landscape, order=4)
    
    for order, terms in epistasis_results['by_order'].items():
        if order > 1:
            for term, value in terms.items():
                assert np.isclose(value, 0), f"Order {order} term {term} should be zero, but is {value}"

def test_rmf_landscape_smooth_component():
    """
    Tests that RMF fitness correlates with distance from the optimum.
    
    Raises
    ------
    AssertionError
        If the fitness values do not correlate with distance from the optimum.
    """
    landscape = RMFFitnessLandscape(N=8, slope=1.0, sigma=0.0, seed=42) # No noise
    fitnesses = landscape.get_signal()
    
    # Optimum is all zeros, distance is number of ones
    distances = [np.sum(seq.to_array()) for seq in landscape.sequences]
    
    # Fitness should be perfectly anti-correlated with distance
    correlation = np.corrcoef(fitnesses, distances)[0, 1]
    assert np.isclose(correlation, -1.0)

def test_elementary_landscape_is_eigenfunction():
    """
    Tests that the fitness signal of an Elementary landscape is a
    Laplacian eigenvector.
    
    Raises
    ------
    AssertionError
        If the fitness signal is not an eigenvector of the Laplacian.
    """
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    j = 3 # Use the 4th eigenvector
    k = 2 # kNN parameter
    
    landscape = ElementaryFitnessLandscape(sequences=sequences, j=j, k=k, seed=42, emb_nodes=False)
    fitness_signal = landscape.get_signal()
    
    # Get the Laplacian of the landscape's graph
    L = nx.laplacian_matrix(landscape.graph).toarray()
    
    Lv = L @ fitness_signal
    
    norm_v = fitness_signal / np.linalg.norm(fitness_signal)
    norm_Lv = Lv / np.linalg.norm(Lv)
    
    assert np.allclose(np.abs(norm_v), np.abs(norm_Lv))

