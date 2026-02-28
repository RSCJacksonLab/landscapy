import math
import numpy as np
import pytest
import networkx as nx

from fitness_landscape.models.nk import create_gnk_landscape, create_nk_binary_landscape
from fitness_landscape.models.rmf import create_rmf_landscape
from fitness_landscape.models.elementary_landscape import create_elementary_landscape
from fitness_landscape.models.dms_dimension import (
    create_ranked_dms_landscape,
    _generate_ranked_effect_matrix,
)
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

def test_binary_gnk_with_variable_sites_and_adjacency_matrix():
    """
    Test the gNK generating with variable sites provided.
    """
    N = 4
    adj_mat = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    landscape = create_nk_binary_landscape(N=N,
                                           seed=42,
                                           adj_mat=adj_mat)
    assert len(landscape.sequences) == N**2
    assert isinstance(landscape.sequences[0], BinarySequence)
    assert len(landscape.get_signal()) == N**2

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

@pytest.mark.parametrize("K", [0, 1, 2, 3, 4])
def test_nk_landscape_initialization_and_seeding_across_k(K):
    """
    Tests initialization and seeding for NK landscapes across K values.
    """
    N = 5
    alphabet = ['A', 'B']
    landscape1 = create_gnk_landscape(N=N, K=K, seed=123, alphabet=alphabet)
    assert len(landscape1.sequences) == 2**N
    assert len(landscape1.get_signal()) == 2**N

    landscape2 = create_gnk_landscape(N=N, K=K, seed=123, alphabet=alphabet)
    assert np.array_equal(landscape1.get_signal(), landscape2.get_signal())

    landscape3 = create_gnk_landscape(N=N, K=K, seed=456, alphabet=alphabet)
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

def _prod(ns):
    p = 1
    for x in ns:
        p *= int(x)
    return p


def test_gnk_dict_alphabet_simple():
    """
    Per-site alphabet with no base_sequence: ensure sequence count = product of site sizes
    and each position uses only its site's alphabet.
    """
    per_site = {
        0: ['A', 'B'],
        1: ['A', 'B', 'C'],
        2: ['B', 'C'],
    }
    N = 3  # number of variable sites; variable_sites defaults to [0,1,2]
    landscape = create_gnk_landscape(N=N, K=1, alphabet=per_site, seed=123)

    expected = _prod(len(per_site[i]) for i in range(N))
    assert len(landscape.sequences) == expected
    assert isinstance(landscape.sequences[0], BaseNumpySequence)
    assert len(landscape.get_signal()) == expected

    # Validate symbols per position
    for seq in landscape.sequences:
        arr = seq.to_array()
        assert len(arr) == N
        for i in range(N):
            assert arr[i] in per_site[i]


def test_gnk_dict_alphabet_with_base_sequence_and_variable_sites():
    """
    Per-site alphabet + base_sequence + variable_sites.
    Non-variable positions must remain fixed; variable ones must draw from their site alph.
    """
    base_seq = list("ABCDE")  # length 5
    variable_sites = [1, 3, 4]  # global indices
    per_site = {
        1: ['X', 'Y'],
        3: ['G', 'H', 'I'],
        4: ['M', 'N'],
    }
    # Make sure base sequence characters at variable sites are allowed
    base_seq[1] = per_site[1][0]
    base_seq[3] = per_site[3][0]
    base_seq[4] = per_site[4][0]

    N = len(variable_sites)
    landscape = create_gnk_landscape(
        N=N,
        K=1,
        alphabet=per_site,
        base_sequence=base_seq,
        variable_sites=variable_sites,
        seed=7,
    )

    expected = _prod(len(per_site[i]) for i in variable_sites)
    assert len(landscape.sequences) == expected
    assert isinstance(landscape.sequences[0], BaseNumpySequence)
    assert len(landscape.get_signal()) == expected

    # Validate fixed vs variable positions
    for seq in landscape.sequences:
        arr = list(seq.to_array())
        # non-variable positions should match base_seq
        for pos in range(len(base_seq)):
            if pos not in variable_sites:
                assert arr[pos] == base_seq[pos]
        # variable positions must be from their per-site alphabets
        for pos in variable_sites:
            assert arr[pos] in per_site[pos]


def test_gnk_dict_alphabet_with_adjacency_matrix_changes_signal():
    """
    With a per-site alphabet, an explicit adjacency matrix should alter the signal
    compared to a different adjacency structure (same seed).
    """
    base_seq = list("ABCDE")  # length 5
    variable_sites = [0, 2, 4]
    per_site = {
        0: ['A', 'B'],
        2: ['C', 'D', 'E'],
        4: ['M', 'N'],
    }
    # ensure base is valid at variable sites
    base_seq[0] = per_site[0][0]
    base_seq[2] = per_site[2][0]
    base_seq[4] = per_site[4][0]

    N = len(variable_sites)
    adj_mat_1 = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    adj_mat_2 = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ])

    L1 = create_gnk_landscape(
        N=N, alphabet=per_site, base_sequence=base_seq, variable_sites=variable_sites,
        adj_mat=adj_mat_1, seed=42
    )
    L2 = create_gnk_landscape(
        N=N, alphabet=per_site, base_sequence=base_seq, variable_sites=variable_sites,
        adj_mat=adj_mat_2, seed=42
    )
    # Same combinatorics
    expected = _prod(len(per_site[i]) for i in variable_sites)
    assert len(L1.sequences) == expected == len(L2.sequences)
    # But different fitness signals due to different adjacency
    assert not np.array_equal(L1.get_signal(), L2.get_signal())


def test_gnk_dict_alphabet_k0_is_additive_per_site():
    """
    K=0 should be additive with per-site contributions.
    Check additivity for a pair of single mutations vs the double mutation.
    """
    per_site = {
        0: ['A', 'B'], # size 2
        1: ['X', 'Y'], # size 2
        2: ['C', 'D', 'E'],  # size 3 (different base size)
    }
    N = 3
    landscape = create_gnk_landscape(N=N, K=0, alphabet=per_site, seed=99)

    ref = BaseNumpySequence(['A', 'X', 'C'])
    ref_fit = landscape.get_fitness(ref)

    mut0 = BaseNumpySequence(['B', 'X', 'C'])
    mut1 = BaseNumpySequence(['A', 'Y', 'C'])
    mut01 = BaseNumpySequence(['B', 'Y', 'C'])

    eff0 = landscape.get_fitness(mut0) - ref_fit
    eff1 = landscape.get_fitness(mut1) - ref_fit
    expected = ref_fit + eff0 + eff1
    actual = landscape.get_fitness(mut01)

    assert np.isclose(expected, actual, atol=1e-8)


def test_gnk_dict_alphabet_missing_site_raises():
    """
    If a dict alphabet is missing a variable site, we expect a ValueError.
    """
    per_site_incomplete = {
        0: ['A', 'B'],
        2: ['C', 'D'],
        # 1 missing
    }
    with pytest.raises(ValueError):
        create_gnk_landscape(N=3, K=1, alphabet=per_site_incomplete, seed=1)


def test_ranked_dms_sequence_count_and_mutation_depth():
    """
    Validate combinatorics and mutation depth for ranked DMS landscapes.
    """
    L = 6
    A = 4
    n_components = 2
    n_mutation_layers = 2
    n_mutable_sites = 3
    seed = 7
    wildtype = [0, 1, 2, 3, 1, 0]

    landscape = create_ranked_dms_landscape(
        L=L,
        A=A,
        n_components=n_components,
        n_mutation_layers=n_mutation_layers,
        n_mutable_sites=n_mutable_sites,
        noise_scale=0.0,
        seed=seed,
        wildtype=wildtype,
    )

    expected = 1
    for d in range(1, n_mutation_layers + 1):
        expected += math.comb(n_mutable_sites, d) * (A - 1) ** d
    assert len(landscape.sequences) == expected

    metadata = landscape.active_layer.metadata
    wt = np.array(metadata["wildtype"], dtype=int)
    mutable_positions = set(metadata["mutable_positions"])

    for seq in landscape.sequences:
        arr = seq.to_array().astype(int)
        diffs = np.where(arr != wt)[0]
        assert len(diffs) <= n_mutation_layers
        assert set(diffs).issubset(mutable_positions)


def test_ranked_dms_effect_matrix_rank_matches_components():
    """
    The effect matrix should have rank equal to n_components when noise is zero.
    """
    L = 10
    A = 6
    n_components = 3
    rng = np.random.default_rng(11)

    M = _generate_ranked_effect_matrix(
        L=L,
        A=A,
        n_components=n_components,
        noise_scale=0.0,
        rng=rng,
    )

    singular_values = np.linalg.svd(M, compute_uv=False)
    tol = singular_values.max() * 1e-8 if singular_values.size else 0.0
    rank = int(np.sum(singular_values > tol))
    assert rank == n_components
