import numpy as np
import pytest
import networkx as nx

from fitness_landscape.models.nk import *
from fitness_landscape.models.rmf import *
from fitness_landscape.models.elementary_landscape import *
from fitness_landscape.core.sequence import generate_sequences, BaseNumpySequence

def test_rmf_landscape_smooth_component():
    """
    Tests that RMF fitness correlates with distance from the optimum.
    This is a good self-contained test.
    """
    landscape = RMFFitnessLandscape(N=8, slope=1.0, sigma=0.0, seed=42) # No noise
    fitnesses = landscape.get_signal()
    
    # Optimum is all zeros, distance is number of ones
    distances = [np.sum(seq.to_array().astype(int)) for seq in landscape.sequences]
    
    # Fitness should be perfectly anti-correlated with distance
    correlation = np.corrcoef(fitnesses, distances)[0, 1]
    assert np.isclose(correlation, -1.0)

def test_elementary_landscape_is_eigenfunction():
    """
    Tests that the fitness signal of an Elementary landscape is a
    Laplacian eigenvector. This is also a good self-contained test.
    """
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    j = 3 # Use the 4th eigenvector
    k = 2 # kNN parameter
    
    landscape = ElementaryFitnessLandscape(sequences=sequences, j=j, k=k, seed=42, emb_nodes=False)
    fitness_signal = landscape.get_signal()
    
    # Get the Laplacian of the landscape's graph
    L = nx.laplacian_matrix(landscape.graph).toarray()
    
    # Calculate L*v
    Lv = L @ fitness_signal
    
    # The eigenvalue is the ratio of the norms (or element-wise division)
    # Avoid division by zero for elements that are zero in the eigenvector
    non_zero_indices = np.where(fitness_signal != 0)[0]
    eigenvalues = Lv[non_zero_indices] / fitness_signal[non_zero_indices]
    
    # All non-zero elements should yield the same eigenvalue
    assert np.allclose(eigenvalues, eigenvalues[0])

def test_nk_landscape_initialization_and_seeding():
    """
    Tests the basic initialization and reproducibility of the NK model.
    """
    # Test that the landscape size is correct (2^N)
    landscape1 = NKFitnessLandscape(N=5, K=1, seed=123, alphabet_size=2)
    assert len(landscape1.sequences) == 2**5
    assert len(landscape1.get_signal()) == 2**5

    # Test that the same seed produces the exact same landscape
    landscape2 = NKFitnessLandscape(N=5, K=1, seed=123, alphabet_size=2)
    assert np.array_equal(landscape1.get_signal(), landscape2.get_signal())

    # Test that a different seed produces a different landscape
    landscape3 = NKFitnessLandscape(N=5, K=1, seed=456, alphabet_size=2)
    assert not np.array_equal(landscape1.get_signal(), landscape3.get_signal())

def test_nk_landscape_is_additive_for_k0():
    """
    Tests the additive property of an NK landscape with K=0 directly,
    without using the epistasis module.
    """
    N = 4
    landscape = NKFitnessLandscape(N=N, K=0, seed=42, alphabet_size=2)
    
    # Get the fitness of the all-zeros sequence (the reference)
    ref_seq = BaseNumpySequence([0] * N)
    ref_fitness = landscape.get_fitness(ref_seq)
    
    # Calculate the fitness effect of flipping each bit individually
    single_mut_effects = []
    for i in range(N):
        mut_seq_arr = np.zeros(N, dtype=int)
        mut_seq_arr[i] = 1
        mut_seq = BaseNumpySequence(mut_seq_arr)
        effect = landscape.get_fitness(mut_seq) - ref_fitness
        single_mut_effects.append(effect)
        
    # Test a sequence with two mutations ('1100')
    double_mut_seq = BaseNumpySequence([1, 1, 0, 0])
    
    # The expected fitness should be the reference plus the sum of individual effects
    expected_fitness = ref_fitness + single_mut_effects[0] + single_mut_effects[1]
    actual_fitness = landscape.get_fitness(double_mut_seq)
    
    assert np.isclose(expected_fitness, actual_fitness)

def test_rmf_landscape_optimum_and_noise():
    """
    Tests that the optimum sequence has the highest fitness and that noise works.
    """
    landscape_smooth = RMFFitnessLandscape(N=5, slope=2.0, sigma=0.0, seed=42)
    fitness_smooth = landscape_smooth.get_signal()
    optimum_seq = landscape_smooth.optimum
    
    assert landscape_smooth.get_fitness(optimum_seq) == np.max(fitness_smooth)
    
    landscape_noisy = RMFFitnessLandscape(N=5, slope=2.0, sigma=10, seed=42)
    fitness_noisy = landscape_noisy.get_signal()
    
    assert not np.array_equal(fitness_smooth, fitness_noisy)
    

    new_optimum_idx = np.argmax(fitness_noisy)
    new_optimum_seq = landscape_noisy.sequences[new_optimum_idx]
    
    # Assert that the noise was significant enough to change the location of the peak.
    assert not np.array_equal(optimum_seq.to_array(), new_optimum_seq.to_array())

def test_elementary_landscapes_are_orthogonal():
    """
    Tests that elementary landscapes (as eigenvectors of the same graph)
    are orthogonal to each other.
    """
    sequences = generate_sequences(length=5, alphabet=[0, 1])
    
    # Create two landscapes based on different eigenvectors (j=1 and j=4)
    landscape1 = ElementaryFitnessLandscape(sequences=sequences, j=1, emb_nodes=False)
    landscape4 = ElementaryFitnessLandscape(sequences=sequences, j=4, emb_nodes=False)
    
    signal1 = landscape1.get_signal()
    signal4 = landscape4.get_signal()
    
    # The dot product of orthogonal vectors should be zero (or very close to it)
    dot_product = np.dot(signal1, signal4)
    assert np.isclose(dot_product, 0.0)