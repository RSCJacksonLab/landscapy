import numpy as np
import pytest
import networkx as nx

from fitness_landscape.models.nk import create_gnk_landscape, create_nk_binary_landscape
from fitness_landscape.models.rmf import create_rmf_landscape
from fitness_landscape.models.elementary_landscape import create_elementary_landscape
from fitness_landscape.core.sequence import (
    BaseNumpySequence,
    BinarySequence,
    MultialleleSequence,
    generate_sequences
)


AMINO_ACID_ALPHABET = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

def test_gnk_binary_default():
    """
    Tests the gNK landscape with default binary alphabet.
    """
    landscape = create_gnk_landscape(N=4, K=1, alphabet=[0, 1], seed=42)
    assert len(landscape.sequences) == 16  # 2^4
    # Updated Assertion: Check for the base class
    assert isinstance(landscape.sequences[0], BaseNumpySequence)
    assert len(landscape.get_signal()) == 16

def test_gnk_amino_acid_alphabet():
    """
    Tests the gNK landscape with a 4-letter amino acid alphabet.
    """
    aa_alphabet = AMINO_ACID_ALPHABET[:4] # Use a small subset for speed
    landscape = create_gnk_landscape(N=3, K=1, alphabet=aa_alphabet, seed=42)
    assert len(landscape.sequences) == 64  # 4^3
    # Updated Assertion: Check for the base class
    assert isinstance(landscape.sequences[0], BaseNumpySequence)
    assert landscape.sequences[0].alphabet == aa_alphabet
    assert len(landscape.get_signal()) == 64

def test_gnk_with_base_sequence():
    """
    Tests the gNK landscape with a fixed base sequence and variable
    sites.
    """
    base_seq = ['A', 'C', 'X', 'G', 'X']
    variable_sites = [2, 4]
    alphabet = ['A', 'C', 'G', 'T']
    
    # Replace 'X' with a valid character from the alphabet for the base sequence
    base_seq = [c if c in alphabet else alphabet[0] for c in base_seq]

    landscape = create_gnk_landscape(
        N=2,
        K=1,
        alphabet=alphabet,
        base_sequence=base_seq,
        variable_sites=variable_sites,
        seed=42
    )
    
    assert len(landscape.sequences) == 16
    test_seq = landscape.sequences[0].to_array()
    assert list(test_seq) == ['A', 'C', 'A', 'G', 'A']
    test_seq_last = landscape.sequences[-1].to_array()
    assert list(test_seq_last) == ['A', 'C', 'T', 'G', 'T']

def test_gnk_with_adjacency_matrix():
    """
    Tests the gNK landscape with a custom adjacency matrix.
    """
    adj_mat = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    
    landscape = create_gnk_landscape(N=4, K=2, alphabet=[0, 1], adj_mat=adj_mat, seed=42)
    default_landscape = create_gnk_landscape(N=4, K=2, alphabet=[0, 1], seed=42)
    
    assert not np.array_equal(landscape.get_signal(), default_landscape.get_signal())

def test_gnk_with_variable_sites():
    """
    Test the gNK generating with variable sites provided.
    """
    wt_seq = "ACDEFGHIKLM"
    alphabet = ["C", "E", "F"]
    landscape = create_gnk_landscape(N=3, 
                                     K=2, 
                                     alphabet=alphabet,
                                     seed=42,
                                     base_sequence=wt_seq,
                                     variable_sites=[1, 3, 4])
    assert len(landscape.sequences) == 3**len(alphabet)
    assert isinstance(landscape.sequences[0], BaseNumpySequence)
    assert len(landscape.get_signal()) == 3**len(alphabet)

def test_gnk_with_variable_sites_and_adjacency_matrix():
    """
    Test the gNK generating with variable sites provided.
    """
    N = 4
    wt_seq = "ACDEFGHIKLM"
    alphabet = ["C", "E", "F", "G"]
    adj_mat = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    landscape = create_gnk_landscape(N=N, 
                                     alphabet=alphabet,
                                     seed=42,
                                     adj_mat=adj_mat,
                                     base_sequence=wt_seq,
                                     variable_sites=[1, 3, 4, 5])
    assert len(landscape.sequences) == N**len(alphabet)
    assert isinstance(landscape.sequences[0], BaseNumpySequence)
    assert len(landscape.get_signal()) == N**len(alphabet)

def test_gnk_invalid_arguments():
    """
    Tests that the gNK landscape raises appropriate errors for invalid
    arguments.
    """
    with pytest.raises(ValueError):
        # N is longer than base sequence
        create_gnk_landscape(
            N=4, K=1, alphabet=['A', 'C'],
            base_sequence=['A', 'C', 'G'], variable_sites=[0]
        )
        
    with pytest.raises(IndexError):
        # Variable site index out of bounds
        create_gnk_landscape(
            N=1, K=1, alphabet=['A', 'C', 'G'],
            base_sequence=['A', 'C', 'G'], variable_sites=[3]
        )

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
    landscape1 = create_gnk_landscape(N=5, K=1, seed=123, alphabet=['A', 'B'])
    assert len(landscape1.sequences) == 2**5
    assert len(landscape1.get_signal()) == 2**5

    landscape2 = create_gnk_landscape(N=5, K=1, seed=123, alphabet=['A', 'B'])
    assert np.array_equal(landscape1.get_signal(), landscape2.get_signal())

    landscape3 = create_gnk_landscape(N=5, K=1, seed=456, alphabet=['A', 'B'])
    assert not np.array_equal(landscape1.get_signal(), landscape3.get_signal())

def test_nk_landscape_is_additive_for_k0():
    """
    Tests the additive property of an NK landscape with K=0.
    """
    N = 4
    alphabet = ['A', 'B']
    landscape = create_gnk_landscape(N=N, K=0, seed=42, alphabet=alphabet)

    # Use the correct alphabet for the reference sequence
    ref_seq = BaseNumpySequence([alphabet[0]] * N)
    ref_fitness = landscape.get_fitness(ref_seq)

    single_mut_effects = []
    for i in range(N):
        # Create the mutated sequence using the correct alphabet
        mut_seq_arr = np.array([alphabet[0]] * N)
        mut_seq_arr[i] = alphabet[1]
        mut_seq = BaseNumpySequence(mut_seq_arr)
        effect = landscape.get_fitness(mut_seq) - ref_fitness
        single_mut_effects.append(effect)

    # Create the double mutant with the correct alphabet
    double_mut_seq = BaseNumpySequence([alphabet[1], alphabet[1], alphabet[0], alphabet[0]])

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
