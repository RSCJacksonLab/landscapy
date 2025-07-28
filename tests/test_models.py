import numpy as np
import pytest
import networkx as nx

from fitness_landscape.models.nk import create_nk_landscape
from fitness_landscape.models.rmf import create_rmf_landscape
from fitness_landscape.models.elementary_landscape import create_elementary_landscape
from fitness_landscape.core.sequence import generate_sequences, BaseNumpySequence

def test_rmf_landscape_smooth_component():
    """
    Tests that RMF fitness correlates with distance from the optimum.
    """
    landscape = create_rmf_landscape(N=8, slope=1.0, sigma=0.0, seed=42) # No noise
    fitnesses = landscape.get_signal()
    
    # Optimum is all zeros, distance is number of ones
    distances = [np.sum(seq.to_array().astype(int)) for seq in landscape.sequences]
    
    correlation = np.corrcoef(fitnesses, distances)[0, 1]
    assert np.isclose(correlation, -1.0)

def test_elementary_landscape_is_eigenfunction():
    """
    Tests that the fitness signal of an Elementary landscape is a
    Laplacian eigenvector.
    """
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    j = 3 # Use the 4th eigenvector
    
    landscape = create_elementary_landscape(sequences=sequences, j=j, emb_nodes=False)
    fitness_signal = landscape.get_signal()
    
    L = nx.laplacian_matrix(landscape.graph).toarray()
    Lv = L @ fitness_signal
    
    non_zero_indices = np.where(np.abs(fitness_signal) > 1e-9)[0]
    eigenvalues = Lv[non_zero_indices] / fitness_signal[non_zero_indices]
    
    assert np.allclose(eigenvalues, eigenvalues[0])

def test_nk_landscape_initialization_and_seeding():
    """
    Tests the basic initialization and reproducibility of the NK model.
    """
    landscape1 = create_nk_landscape(N=5, K=1, seed=123, alphabet_size=2)
    assert len(landscape1.sequences) == 2**5
    assert len(landscape1.get_signal()) == 2**5

    landscape2 = create_nk_landscape(N=5, K=1, seed=123, alphabet_size=2)
    assert np.array_equal(landscape1.get_signal(), landscape2.get_signal())

    landscape3 = create_nk_landscape(N=5, K=1, seed=456, alphabet_size=2)
    assert not np.array_equal(landscape1.get_signal(), landscape3.get_signal())

def test_nk_landscape_is_additive_for_k0():
    """
    Tests the additive property of an NK landscape with K=0.
    """
    N = 4
    landscape = create_nk_landscape(N=N, K=0, seed=42, alphabet_size=2)
    
    ref_seq = BaseNumpySequence([0] * N)
    ref_fitness = landscape.get_fitness(ref_seq)
    
    single_mut_effects = []
    for i in range(N):
        mut_seq_arr = np.zeros(N, dtype=int)
        mut_seq_arr[i] = 1
        mut_seq = BaseNumpySequence(mut_seq_arr)
        effect = landscape.get_fitness(mut_seq) - ref_fitness
        single_mut_effects.append(effect)
        
    double_mut_seq = BaseNumpySequence([1, 1, 0, 0])
    
    expected_fitness = ref_fitness + single_mut_effects[0] + single_mut_effects[1]
    actual_fitness = landscape.get_fitness(double_mut_seq)
    
    assert np.isclose(expected_fitness, actual_fitness)

def test_rmf_landscape_optimum_and_noise():
    """
    Tests that the optimum sequence has the highest fitness and that
    noise works.
    """
    optimum_sequence = np.zeros(5, dtype=int)
    landscape_smooth = create_rmf_landscape(N=5, slope=2.0, sigma=0.0, seed=42, optimum=optimum_sequence)
    fitness_smooth = landscape_smooth.get_signal()
    
    assert landscape_smooth.get_fitness(BaseNumpySequence(optimum_sequence)) == np.max(fitness_smooth)
    
    landscape_noisy = create_rmf_landscape(N=5, slope=2.0, sigma=10, seed=42, optimum=optimum_sequence)
    fitness_noisy = landscape_noisy.get_signal()
    
    assert not np.array_equal(fitness_smooth, fitness_noisy)

    new_optimum_idx = np.argmax(fitness_noisy)
    new_optimum_seq = landscape_noisy.sequences[new_optimum_idx]
    
    assert not np.array_equal(optimum_sequence, new_optimum_seq.to_array())

def test_elementary_landscapes_are_orthogonal():
    """
    Tests that elementary landscapes are orthogonal.
    """
    sequences = generate_sequences(length=5, alphabet=[0, 1])
    
    landscape1 = create_elementary_landscape(sequences=sequences, j=1, emb_nodes=False)
    landscape4 = create_elementary_landscape(sequences=sequences, j=4, emb_nodes=False)
    
    signal1 = landscape1.get_signal()
    signal4 = landscape4.get_signal()
    
    dot_product = np.dot(signal1, signal4)
    assert np.isclose(dot_product, 0.0)
