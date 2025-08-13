import numpy as np
import pytest
import networkx as nx

from fitness_landscape.core.sequence import *
from fitness_landscape.core.graph import *
from fitness_landscape.core.landscape import *
from fitness_landscape.analysis.epistasis import *
from fitness_landscape.analysis.statistics import *
from fitness_landscape.analysis.adaptive_walk import *
from fitness_landscape.analysis.random_walk import *
from fitness_landscape.analysis.bottleneck import *
from fitness_landscape.analysis.bottleneck import _ensure_affinity, _outward_cut_leakage
from fitness_landscape.analysis.dirichlet_energy import *
from fitness_landscape.analysis.graph_induction_alignment import *
from fitness_landscape.transforms.graph_fourier import *
from fitness_landscape.transforms.eigenmode import *
from fitness_landscape.analysis.graph import *
from fitness_landscape.analysis.diffusion_scale import (
    _precompute_GMRF_stats,
    compute_log_likelihood_H0,
    fit_t_bayesian_laplace,
    _compute_variances,
    compute_ruggedness_diffusion_scale,
    compute_ruggedness_variance_energy,
)
from fitness_landscape.models.elementary_landscape import *
from fitness_landscape.models.nk import *
from fitness_landscape.models.rmf import *
from math import factorial
from fitness_landscape.analysis.persistent_homology import (
    vietoris_rips_complex,
    delauny_cech_complex,
    compute_persistent_homology,
    compute_betti_curves,
)
from fitness_landscape.core.fitness import NumericFitness
from fitness_landscape.utils import make_latent_geometric_graph_connected, sample_observed_induced_connected
from scipy.sparse import issparse


@pytest.fixture
def simple_graph_bottleneck():
    """A simple graph for testing."""
    G = nx.path_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.9
    return G


@pytest.fixture
def envelope_graph_and_subgraph_bottleneck():
    """An envelope graph and a subgraph S for testing leakage."""
    G_env = nx.Graph()
    # Add weights to all edges
    G_env.add_edge(0, 1, weight=0.9)
    G_env.add_edge(1, 2, weight=0.9)
    G_env.add_edge(2, 3, weight=0.9)
    G_env.add_edge(3, 4, weight=0.9)
    G_env.add_edge(1, 5, weight=0.9)
    G_env.add_edge(2, 6, weight=0.9)
    
    S = [0, 1, 2, 3]
    G_obs = G_env.subgraph(S).copy()
    return G_env, G_obs, S

@pytest.fixture
def homology_landscape():
    """Provides a basic FitnessLandscape for homology testing."""
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    fitness_values = [[val] for val in np.random.rand(16)]
    fitness_layers = {
        'default': NumericFitness(name='default', values=fitness_values)
    }
    return FitnessLandscape.from_sequences(
        sequences=sequences, 
        fitness_layers=fitness_layers,
        graph_type='hamming'
    )

@pytest.fixture
def additive_landscape():
    """
    A purely additive NK landscape with K=0, N=4.
    """
    return create_nk_binary_landscape(N=4, K=0, seed=42)

@pytest.fixture
def epistatic_landscape():
    """
    An epistatic NK landscape with K=2, N=4.
    """
    return create_nk_binary_landscape(N=4, K=2, seed=42)

@pytest.fixture(params=[1, 5, 10]) # Test with 3 different eigenvectors
def elementary_landscape(request):
    """
    Creates an ElementaryFitnessLandscape based on the j-th
    eigenvector where j is parameterized by pytest.
    """
    j = request.param
    sequences = generate_sequences(length=5, alphabet=[0, 1]) # N=5 -> 32 nodes.
    landscape = create_elementary_landscape(
        sequences=sequences, j=j, graph_type='hamming', seed=42
    )
    return landscape, j

@pytest.fixture
def complete_hamming_graph_n3():
    """
    Provides a complete Hamming graph for N=3.
    """
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    return create_hamming_graph(sequences=sequences)

@pytest.fixture
def random_rmf_landscape():
    """
    An uncorrelated RMF landscape (pure noise).
    """
    return create_rmf_landscape(N=8, slope=0.0, sigma=10.0, seed=42)

@pytest.fixture
def linear_rmf_landscape():
    """
    An RMF landscape with no noise and a perfect linear relationship.
    """
    return create_rmf_landscape(N=8, slope=1.0, sigma=0.0, seed=42)


# Adaptive walk tests
def test_number_of_local_optima_vs_K(additive_landscape: additive_landscape,
                                     epistatic_landscape: epistatic_landscape):
    """
    Tests that the number of local optima increases with K.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).
    
    Raises
    ------
    AssertionError
        If the number of local optima does not match the expected
        values.
    """
    # For K=0, there should be exactly one local (and global) maximum.
    smooth_results = analyze_path_accessibility(additive_landscape)
    assert smooth_results['maxima_count'] == 1

    # For K>0, the number of maxima should be greater than 1.
    rugged_results = analyze_path_accessibility(epistatic_landscape )
    assert rugged_results['maxima_count'] > 1


def test_number_of_greedy_paths_vs_K(additive_landscape: additive_landscape,
                                     epistatic_landscape: epistatic_landscape):
    """
    Tests that the number of accessible paths decreases with K. For
    K=0, paths between antipodal nodes (distance d) = d!

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).

    Raises
    ------
    AssertionError
        If the number of paths does not match the expected values.
    """
    # For K=0, N=4, the number of paths between "0000" and "1111" should be 4! = 24
    fitness_values = additive_landscape.get_signal()
    start_idx = np.argmin(fitness_values)
    end_idx = np.argmax(fitness_values)
    start_smooth = additive_landscape.sequences[start_idx]
    end_smooth = additive_landscape.sequences[end_idx]
    
    smooth_paths = find_greedy_accessible_paths(additive_landscape, start_smooth, end_smooth)
    # On a K=0 landscape, there should be at least one greedy path from min to max.
    assert smooth_paths['path_count'] == factorial(4)

    # For the rugged landscape, check between antipodal points where paths are unlikely.
    start_rugged = BinarySequence([0, 0, 0, 0])
    end_rugged = BinarySequence([1, 1, 1, 1])
    rugged_paths = find_greedy_accessible_paths(epistatic_landscape, start_rugged, end_rugged)
    assert rugged_paths['path_count'] < factorial(4)

def test_basin_of_attraction_vs_K(additive_landscape: additive_landscape,
                                  epistatic_landscape: epistatic_landscape):
    """
    Tests that the basin of the global optimum shrinks with K. For K=0,
    the basin is the entire space. For K>0, it's a fraction.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).

    Raises
    ------
    AssertionError
        If the basin sizes do not match the expected values.
    """
    smooth_fitnesses = additive_landscape.get_signal()
    smooth_optimum_idx = np.argmax(smooth_fitnesses)
    smooth_optimum_seq = additive_landscape.sequences[smooth_optimum_idx]
    smooth_basin = calculate_basin_of_attraction_greedy(additive_landscape, smooth_optimum_seq)
    assert smooth_basin['basin_size'] == len(additive_landscape.sequences)

    rugged_fitnesses = epistatic_landscape.get_signal()
    rugged_optimum_idx = np.argmax(rugged_fitnesses)
    rugged_optimum_seq = epistatic_landscape.sequences[rugged_optimum_idx]
    rugged_basin = calculate_basin_of_attraction_greedy(epistatic_landscape, rugged_optimum_seq)
    assert rugged_basin['basin_size'] < len(epistatic_landscape.sequences)

def test_adaptive_walk_length_vs_K(additive_landscape: additive_landscape,
                                   epistatic_landscape: epistatic_landscape):
    """
    Tests that the average adaptive walk length increases with K.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).

    Raises
    ------
    AssertionError
        If the average walk lengths do not match the expected values.
    """
    # Average over a few walks to get a stable estimate
    n_walks = 100
    smooth_lengths = [adaptive_walk_stochastic(additive_landscape, strategy='greedy')['steps_taken'] for _ in range(n_walks)]
    rugged_lengths = [adaptive_walk_stochastic(epistatic_landscape, strategy='greedy')['steps_taken'] for _ in range(n_walks)]
    
    # Code terminates at local optima - does the rugged landscape reach local optima before the smooth one does? That is what this tests for. 
    # Not an ideal test as it is not analytically derived, but it is a sanity check.
    assert np.mean(smooth_lengths) > np.mean(rugged_lengths)

def test_basin_of_attraction_stochastic_runs(epistatic_landscape: epistatic_landscape):
    """
    Tests that the stochastic basin calculation runs and returns a
    plausible result.

    Raises
    ------
    AssertionError
        If the basin size is not greater than 0.
    """
    rugged_fitnesses = epistatic_landscape.get_signal()
    optimum_idx = np.argmax(rugged_fitnesses)
    optimum_seq = epistatic_landscape.sequences[optimum_idx]

    # Use a low number of simulations for a fast test
    stochastic_basin = calculate_basin_of_attraction_stochastic(
        epistatic_landscape,
        optimum_seq,
        n_simulations=10,
        beta=1.0
    )
    assert 'basin_size' in stochastic_basin
    assert stochastic_basin['basin_size'] > 0

def test_neutral_network_analysis_runs(epistatic_landscape: epistatic_landscape):
    """
    Tests that neutral network analysis correctly identifies
    components.

    Parameters
    ----------
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).
    
    Raises
    ------
    AssertionError
        If the analysis does not find any networks or the largest
        network size is 0.
    """
    # A rugged landscape is likely to have some neutral or near-neutral connections
    neutral_results = neutral_network_analysis(epistatic_landscape, threshold=0.01)
    assert 'network_count' in neutral_results
    assert neutral_results['network_count'] > 0
    assert neutral_results['largest_network_size'] > 0
    

# Dirichelet energy tests
def test_total_dirichlet_energy_equals_eigenvalue(elementary_landscape):
    """
    Tests that the total Dirichlet energy of an elementary landscape equals its
    defining eigenvalue.
    """
    landscape, j = elementary_landscape
    energy_results = calculate_ruggedness_dirichlet_energy(landscape)
    calculated_energy_per_node = energy_results['total_dirichlet_energy']
    eigenvalues, _ = eigenmode_decomposition(landscape.graph, matrix='laplacian')
    eigenvalues.sort()
    true_eigenvalue = eigenvalues[j]
    expected_energy_per_node = true_eigenvalue / len(landscape.sequences)
    assert np.isclose(calculated_energy_per_node, expected_energy_per_node)

def test_sum_of_local_contributions_equals_total_energy(elementary_landscape):
    """
    Tests that the sum of local Dirichlet energy contributions equals the
    total Dirichlet energy.
    
    Theory: The total Dirichlet energy E = Σ E_i, where E_i is the local
    contribution of each node i.
    """
    landscape, _ = elementary_landscape
    
    # 1. Calculate the total Dirichlet energy.
    total_energy_results = calculate_ruggedness_dirichlet_energy(landscape)
    # The function returns E/N, so we multiply by N to get the total E.
    total_energy = total_energy_results['total_dirichlet_energy'] * len(landscape.sequences)

    # 2. Calculate the local contributions for all nodes.
    local_contributions = local_dirichlet_energy_contribution(landscape)
    
    # 3. Sum the local contributions.
    sum_of_locals = sum(local_contributions.values())
    
    # 4. Assert that the sum of locals equals the total energy.
    assert np.isclose(sum_of_locals, total_energy)




# Epistasis tests
def test_walsh_on_additive_landscape(additive_landscape):
    """
    Tests that Walsh-Hadamard transform finds no high-order epistasis for a K=0 landscape.
    """
    results = calculate_epistasis_walsh(additive_landscape, order=4)
    # For a K=0 landscape, all coefficients for orders > 1 must be zero.
    for order, coeffs in results['by_order'].items():
        if order > 1:
            for term, value in coeffs.items():
                assert np.isclose(value, 0), f"Walsh term {term} should be zero"

def test_regression_on_additive_landscape(additive_landscape):
    """
    Tests that regression finds no high-order epistasis and has a perfect fit for a K=0 landscape.
    """
    # Test with order=2 to check for second-order terms
    results = calculate_epistasis_regression(additive_landscape, order=2)
    
    # The R2 score should be 1.0, indicating a perfect linear fit.
    assert np.isclose(results['model']['r2_score'], 1.0, atol=0.05)
    
    # All second-order coefficients must be zero.
    if 2 in results['by_order']:
        for term, value in results['by_order'][2].items():
            assert np.isclose(value, 0, atol=0.05), f"Regression term {term} should be zero"

def test_reference_free_on_additive_landscape(additive_landscape):
    """
    Tests that the reference-free method finds no high-order epistasis for a K=0 landscape.
    """
    results = calculate_epistasis_reference_free(additive_landscape, order=2)
    # All second-order coefficients must be zero.
    if 2 in results['by_order']:
        for term, value in results['by_order'][2].items():
            assert np.isclose(value, 0), f"Reference-free term {term} should be zero"

def test_ensemble_on_additive_landscape(additive_landscape):
    """
    Tests that the ensemble method finds no high-order epistasis for a K=0 landscape.
    """
    results = calculate_epistasis_ensemble(additive_landscape, order=2)
    # All second-order coefficients must be zero.
    if 2 in results['by_order']:
        for term, value in results['by_order'][2].items():
            assert np.isclose(value, 0), f"Ensemble term {term} should be zero"

# Sanity Check for Epistatic Landscapes (K>0)

@pytest.mark.parametrize("epistasis_func", [
    calculate_epistasis_walsh,
    calculate_epistasis_regression,
    calculate_epistasis_reference_free,
    calculate_epistasis_ensemble
])
def test_epistasis_detection_on_k1_landscape(epistatic_landscape, epistasis_func):
    """
    Tests that all methods can detect non-zero epistasis on a K=1 landscape.
    """
    results = epistasis_func(epistatic_landscape, order=2)
    
    # For a K=1 landscape, we expect at least one non-zero second-order term.
    assert 2 in results['by_order'], f"{epistasis_func.__name__} did not compute second-order terms."
    
    second_order_coeffs = results['by_order'][2].values()
    assert any(not np.isclose(v, 0) for v in second_order_coeffs), \
        f"{epistasis_func.__name__} failed to detect any second-order epistasis."



# Fourier transform tests
def test_eigenmode_reconstruction():
    """
    Tests that a matrix can be reconstructed from its eigenmodes.

    Raises
    ------
    AssertionError
        If the reconstructed matrix does not match the original matrix.
    
    """
    graph = nx.path_graph(4)
    L = nx.laplacian_matrix(graph).toarray()
    
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='laplacian')
    
    # Reconstruct using all modes
    reconstructed_L = reconstruct_from_eigenmodes(eigenvectors, eigenvalues)
    
    assert np.allclose(L, reconstructed_L, atol=1e-9)

def test_gft_reconstruction():
    """
    Tests that a signal can be perfectly reconstructed via inverse GFT.

    Raises
    ------
    AssertionError
        If the reconstructed signal does not match the original signal.
    """
    graph = nx.cycle_graph(4)
    signal = np.sin(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    for i, node in enumerate(graph.nodes()):

        graph.nodes[node]['sequence'] = BaseNumpySequence([i])
        graph.nodes[node]['fitness_default'] = signal[i]
        graph.nodes[node]['gapped_arr'] = np.zeros((1, 21))
        graph.nodes[node]['ungapped_arr'] = np.zeros((1, 20))
    
    landscape = FitnessLandscape.from_graph(graph, emb_nodes=False)
    eigenvectors, _, coefficients = graph_fourier_transform(landscape)
    reconstructed_signal = inverse_graph_fourier_transform(eigenvectors, coefficients)
    assert np.allclose(signal, reconstructed_signal, atol=1e-9)




# Graph analysis tests
def test_graph_properties_on_hamming_graph(complete_hamming_graph_n3: complete_hamming_graph_n3):
    """
    Validates graph_properties against the known theoretical values
    for a complete N=3 Hamming graph.

    Parameters
    ----------
    complete_hamming_graph_n3 : networkx.Graph
        A complete Hamming graph for N=3.

    Raises
    ------
    AssertionError
        If the calculated properties do not match the expected values.
    """
    properties = graph_properties(complete_hamming_graph_n3)

    # Theoretical properties of a 3-cube graph:
    # 8 nodes, 12 edges, each node has degree 3.
    num_nodes = 8
    num_edges = 12
    
    assert properties['degree']['mean'] == 3.0
    assert properties['degree']['min'] == 3.0
    assert properties['degree']['max'] == 3.0
    assert properties['clustering'] == 0.0  # No triangles in a cube graph
    assert properties['components']['count'] == 1
    
    expected_density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    assert np.isclose(properties['density'], expected_density)
    
    # Avg shortest path length for a 3-cube is (1*3 + 2*3 + 3*1) / 7
    expected_path_length = (3 + 6 + 3) / 7.0
    assert np.isclose(properties['path_length'], expected_path_length)


def test_local_optima_on_smooth_landscape(additive_landscape: additive_landscape):
    """
    Validates that a purely additive (K=0) NK landscape has exactly one
    local (and global) optimum.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).

    Raises
    ------
    AssertionError
        If the number of local optima is not exactly 1.
    """
    results = calculate_ruggedness_local_optima(additive_landscape)
    
    assert results['local_optima_count'] == 1, \
        "A K=0 landscape must have exactly one local optimum."

def test_local_optima_on_rugged_landscape(epistatic_landscape: epistatic_landscape):
    """
    Validates that a rugged (K>0) NK landscape has more than one
    local optimum.

    Parameters
    ----------
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).
    Raises
    ------
    AssertionError
        If the number of local optima is not greater than 1.
    """
    results = calculate_ruggedness_local_optima(epistatic_landscape)
    
    assert results['local_optima_count'] > 1, \
        "A K>0 landscape is expected to have multiple local optima."

def test_local_optima_fitness_statistics(epistatic_landscape: epistatic_landscape):
    """
    Sanity check for the fitness statistics of the found local optima.

    Parameters
    ----------
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0). 
    Raises
    ------
    AssertionError
        If the statistics do not match the expected properties of a
        rugged landscape.
    """
    results = calculate_ruggedness_local_optima(epistatic_landscape)
    
    # The max fitness of the optima should be the global max fitness of the landscape
    global_max_fitness = np.max(epistatic_landscape.get_signal())
    
    assert 'mean_fitness' in results
    assert np.isclose(results['max_fitness'], global_max_fitness)
    assert results['min_fitness'] <= results['max_fitness']


# Random walk analysis tests
def test_stochastic_correlation_length_vs_ruggedness(additive_landscape: additive_landscape,
                                                     epistatic_landscape: epistatic_landscape):
    """
    Tests that the correlation length is greater for a smooth landscape
    than a rugged one.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    epistatic_landscape : NKFitnessLandscape
        An epistatic landscape (K>0).
    
    Raises
    ------
    AssertionError
        If the correlation length for the smooth landscape is not greater
        than that for the rugged landscape.
    """
    smooth_results = calculate_ruggedness_autocorrelation_stochastic(additive_landscape)
    rugged_results = calculate_ruggedness_autocorrelation_stochastic(epistatic_landscape)
    
    assert smooth_results['correlation_length'] > rugged_results['correlation_length']

def test_autocorrelation_on_random_landscape(random_rmf_landscape: random_rmf_landscape):
    """
    Tests that autocorrelation is near zero for a completely random
    landscape.

    Parameters
    ----------
    random_rmf_landscape : RMFFitnessLandscape
        An RMF landscape with no slope and high noise.
    Raises
    ------
    AssertionError
        If the autocorrelation at lag 1 is not close to zero.
    """
    results = calculate_ruggedness_autocorrelation_stochastic(random_rmf_landscape, lag_max=5)
    
    # The autocorrelation at lag 1 should be very close to zero.
    assert np.isclose(results['autocorrelation'][1], 0, atol=0.1)


# Statistics tests
def test_analyze_fitness_distribution(additive_landscape: additive_landscape):
    """
    Tests that the fitness distribution of an additive landscape is
    correctly identified as approximately normal.

    Parameters
    ---------- 
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    Raises
    ------
    AssertionError
        If the normality test fails or the mean is not close to the
        expected value.
    """
    results = analyze_fitness_distribution(additive_landscape)
    
    # For K=0, fitness is a sum of i.i.d. variables, so it should be normal-like.
    # The Shapiro-Wilk test p-value should be > 0.05.
    assert results['normality_test']['is_normal'] == True
    assert results['mean'] == pytest.approx(0.5, abs=0.1)

def test_correlation_analysis(linear_rmf_landscape: linear_rmf_landscape):
    """
    Tests that correlation analysis correctly identifies the perfect
    linear relationship in a noise-free RMF landscape.

    Parameters
    ----------
    additive_landscape : RMFFitnessLandscape
        An RMF landscape with a linear fitness signal.
    Raises
    ------
    AssertionError
        If the Pearson correlation is not close to -1.0, indicating a
        perfect anti-correlation with distance from the optimum.
    """
    # Create a feature that is the Hamming distance from the optimum ("00...0")
    distances = [np.sum(seq.to_array().astype(int)) for seq in linear_rmf_landscape.sequences]
    features = {'distance_from_optimum': distances}
    
    results = correlation_analysis(linear_rmf_landscape, features)
    
    pearson_corr = results['pearson']['distance_from_optimum']['correlation']
    assert np.isclose(pearson_corr, -1.0)

def test_regression_analysis(additive_landscape: additive_landscape):
    """
    Tests that linear regression can perfectly model a purely additive
    landscape.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    Raises
    ------
    AssertionError
        If the R2 score is not close to 1.0.
    """
    # For a K=0 landscape, fitness is a linear function of the sequence bits.
    features = {f'pos_{i}': [s.to_array()[i] for s in additive_landscape.sequences] for i in range(4)}
    
    results = regression_analysis(additive_landscape, features)
    
    assert np.isclose(results['models']['linear']['test_r2'], 1.0)
    
    # A simple linear model should explain all variance, so R² should be 1.0
    assert np.isclose(results['models']['linear']['test_r2'], 1.0)

def test_hypothesis_testing(additive_landscape: additive_landscape):
    """
    Tests that hypothesis testing correctly identifies a significant
    difference between two distinct groups of sequences.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    Raises
    ------
    AssertionError
        If the t-test does not find a significant difference between
        the two groups.
    """
    group1_indices = [i for i, seq in enumerate(additive_landscape.sequences) if np.sum(seq.to_array().astype(int)) <= 1]
    group2_indices = [i for i, seq in enumerate(additive_landscape.sequences) if np.sum(seq.to_array().astype(int)) >= 3]
    
    groups = {'high_fitness': group1_indices, 'low_fitness': group2_indices}
    
    results = hypothesis_testing(additive_landscape, groups)
    
    ttest_results = results['pairwise_tests']['high_fitness']['low_fitness']['t_test']
    assert ttest_results['significant'] == True
    assert ttest_results['p_value'] < 0.05

def test_bootstrap_analysis(additive_landscape: additive_landscape):
    """
    Tests that bootstrap analysis produces a confidence interval that
    contains the observed statistic.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    Raises
    ------
    AssertionError
        If the confidence interval does not contain the observed mean.
    """
    # Bootstrap the mean of the fitness distribution
    results = bootstrap_analysis(additive_landscape, statistic_func=np.mean, n_bootstrap=100)
    
    observed_mean = results['observed']
    lower_ci, upper_ci = results['confidence_interval']
    
    assert lower_ci <= observed_mean <= upper_ci

def test_permutation_test(additive_landscape: additive_landscape):
    """
    Tests that a permutation test confirms the significant difference
    found by the standard hypothesis test.

    Parameters
    ----------
    additive_landscape : NKFitnessLandscape
        A purely additive landscape (K=0).
    Raises
    ------
    AssertionError
        If the permutation test does not confirm the significance of
        the difference between the two groups.
    """
    # CORRECTED: Adjusted group definitions for an N=4 landscape.
    group1_indices = [i for i, seq in enumerate(additive_landscape.sequences) if np.sum(seq.to_array().astype(int)) <= 1]
    group2_indices = [i for i, seq in enumerate(additive_landscape.sequences) if np.sum(seq.to_array().astype(int)) >= 3]
    groups = {'high_fitness': group1_indices, 'low_fitness': group2_indices}
    
    def diff_in_means(a, b):
        return np.mean(a) - np.mean(b)
        
    results = permutation_test(additive_landscape, groups, statistic_func=diff_in_means, n_permutations=500)
    
    assert results['significant'] == True
    assert results['p_value'] < 0.05

def test_vietoris_rips_complex(homology_landscape):
    """Tests the Vietoris-Rips complex computation."""
    simplex_tree = vietoris_rips_complex(homology_landscape, max_dim=2)
    assert simplex_tree.num_simplices() > 0
    assert simplex_tree.dimension() == 2

def test_compute_persistent_homology(homology_landscape):
    """Tests the persistent homology computation."""
    persistence = compute_persistent_homology(homology_landscape, max_dim=2)
    assert "persistence_intervals" in persistence
    assert "betti_numbers" in persistence
    assert "stats" in persistence
    assert len(persistence["betti_numbers"]) > 0

def test_compute_betti_curves(homology_landscape):
    """Tests the Betti curve computation."""
    persistence = compute_persistent_homology(homology_landscape, max_dim=2)
    betti_curves, filtration_range = compute_betti_curves(persistence["persistence_intervals"], max_dim=2)
    assert 0 in betti_curves
    assert 1 in betti_curves
    assert 2 in betti_curves
    assert len(betti_curves[0]) == len(filtration_range)

def test_eigenmode_decomposition_torch():
    """Tests eigenmode decomposition with torch backend."""
    graph = nx.path_graph(4)
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='laplacian', backend='torch')
    assert eigenvalues is not None
    assert eigenvectors is not None

def test_reconstruct_from_eigenmodes_torch():
    """Tests reconstruction from eigenmodes with torch backend."""
    graph = nx.path_graph(4)
    L = nx.laplacian_matrix(graph).toarray()
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='laplacian', backend='torch')
    reconstructed_L = reconstruct_from_eigenmodes(eigenvectors, eigenvalues, backend='torch')
    assert np.allclose(L, reconstructed_L.numpy(), atol=1e-6)

def test_graph_spectral_analysis(additive_landscape):
    """Tests graph spectral analysis."""
    results = graph_spectral_analysis(additive_landscape, matrix='laplacian')
    assert 'eigenvalues' in results
    assert 'participation_ratios' in results
    assert 'localization' in results
    assert 'node_centralities' in results

# New tests for dirichlet_energy.py
def test_calculate_ruggedness_dirichlet_energy_weighted(additive_landscape):
    """Tests Dirichlet energy with a weighted Laplacian."""
    results = calculate_ruggedness_dirichlet_energy(additive_landscape, weighted_laplacian=True)
    assert 'total_dirichlet_energy' in results

def test_local_dirichlet_energy_contribution(additive_landscape):
    """Tests local Dirichlet energy contribution."""
    local_energies = local_dirichlet_energy_contribution(additive_landscape)
    assert len(local_energies) == len(additive_landscape.sequences)
    assert sum(local_energies.values()) > 0

def test_find_greedy_accessible_paths_no_path(additive_landscape):
    """Tests that no path is found when start and end are reversed."""
    fitness_values = additive_landscape.get_signal()
    start_idx = np.argmax(fitness_values)
    end_idx = np.argmin(fitness_values)
    start_seq = additive_landscape.sequences[start_idx]
    end_seq = additive_landscape.sequences[end_idx]
    paths = find_greedy_accessible_paths(additive_landscape, start_seq, end_seq)
    assert paths['path_count'] == 0

def test_dirichlet_energy_with_bins(additive_landscape):
    """Tests Dirichlet energy with edge weight bins."""
    results = calculate_ruggedness_dirichlet_energy(additive_landscape, edge_weight_bins=[(0, 2)])
    assert 'edge_weight_bins' in results
    assert len(results['edge_weight_bins']) > 0

def test_eigenmode_decomposition_variants():
    """Tests different matrices for eigenmode decomposition."""
    graph = nx.path_graph(5)
    adj_vals, _ = eigenmode_decomposition(graph, matrix='adjacency')
    trans_vals, _ = eigenmode_decomposition(graph, matrix='transition')
    assert adj_vals is not None
    assert trans_vals is not None

def test_graph_properties_disconnected():
    """Tests graph properties on a disconnected graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)
    properties = graph_properties(graph)
    assert properties['components']['count'] == 3
    assert 'path_length_note' in properties

def test_walsh_variance_explained(additive_landscape):
    """
    Tests the new variation_explained functionality in the Walsh-Hadamard epistasis calculation.
    """
    results = calculate_epistasis_walsh(additive_landscape, order=4)
    
    assert 'variance_explained' in results
    
    # The sum of explained variances should be close to 1.0
    total_explained = sum(results['variance_explained'].values())
    assert np.isclose(total_explained, 1.0)
    
    assert results['variance_explained'][1] > 0.95

def test_get_epistasis_matrix_variance(epistatic_landscape):
    """
    Tests that the get_epistasis_matrix function returns a matrix of
    variances (floats) for pairwise interactions.
    """
    epistasis_matrix = get_epistasis_matrix(epistatic_landscape)
    assert isinstance(epistasis_matrix, np.ndarray)
    n = len(epistatic_landscape.sequences[0])
    assert epistasis_matrix.shape == (n, n)
    assert epistasis_matrix.dtype == float

    assert np.all(np.diag(epistasis_matrix) == 0)
    assert np.allclose(epistasis_matrix, epistasis_matrix.T)
    off_diagonal_mask = ~np.eye(n, dtype=bool)
    assert np.any(epistasis_matrix[off_diagonal_mask] > 0)

def test_graph_reconstruction_analysis():
    """
    Tests that the graph matching returns a correctly structured dict.
    Results are not analytical, thus analytical checks are ommitted.
    """

    G = make_latent_geometric_graph_connected(n_latent = 20,
                                              d_target = 4,
                                              k_edges = 16,
                                              seed = 42)

    G_ind = sample_observed_induced_connected(G, node_keep=0.5, edge_keep=0.5, seed=42)
    
    results = evaluate_reconstruction(G, G_ind, G)
    assert np.isclose(results["edge_precision"], 1)
    assert np.isclose(results["edge_recall"], 1)
    assert np.isclose(results["edge_F1"], 1)
    assert np.isclose(results["sp_RMSE_recon_vs_truth"], 0)
    assert np.isclose(results["spectral_RMSE"], 0)

def test_outward_cut_leakage_fixed(envelope_graph_and_subgraph_bottleneck):
    G_env, _, S = envelope_graph_and_subgraph_bottleneck
    b = _outward_cut_leakage(G_env, S, weight_key="weight")
    assert b.get(0, 0) == 0
    assert np.isclose(b.get(1, 0), 0.9)  # Edge to node 5
    assert np.isclose(b.get(2, 0), 0.9)  # Edge to node 6
    assert np.isclose(b.get(3, 0), 0.9)  # Edge to node 4


def test_local_cheeger_sweep_fixed(envelope_graph_and_subgraph_bottleneck):
    G_env, _, S = envelope_graph_and_subgraph_bottleneck
    nodes_S = list(S)
    f = np.array([-0.5, -0.4, 0.4, 0.5])
    _ensure_affinity(G_env, length_key="weight", sim_key="sim")

    h_est, T_star = local_cheeger_sweep(G_env, S, f, nodes_S, weight_key="sim")

    assert isinstance(h_est, float)
    assert T_star == {3}

def test_calculate_local_bottleneck_without_latent_graph_fixed(simple_graph_bottleneck):

    results = calculate_local_bottleneck(simple_graph_bottleneck, return_latent_graph=True)

    assert "first_dirichlet_eigenvalue" in results
    assert "local_cheeger_constant" in results
    assert "latent_graph" in results
    assert isinstance(results["latent_graph"], nx.Graph)

def test_precompute_gmrf_stats_shapes(additive_landscape):
    """
    Ensures _precompute_GMRF_stats returns correctly shaped outputs and
    a strictly positive empirical variance after centering.
    """
    G = additive_landscape.graph
    signal = additive_landscape.get_signal()
    f_hat, eigenvalues, sigma2 = _precompute_GMRF_stats(G, signal)

    n = len(signal)
    assert f_hat.shape == (n,)
    assert eigenvalues.shape == (n,)
    assert sigma2 > 0.0


def test_log_likelihood_finite(additive_landscape):
    """
    The Gaussian GMRF log-likelihood should be finite for reasonable t
    values and monotonically better-conditioned with epsilon.
    """
    G = additive_landscape.graph
    signal = additive_landscape.get_signal()
    f_hat, eigenvalues, sigma2 = _precompute_GMRF_stats(G, signal)

    for t in [0.01, 0.1, 1.0]:
        ll, logdet, qf = compute_log_likelihood_H0(
            f_hat=f_hat,
            eigenvalues=eigenvalues,
            t=t,
            sigma_squared=sigma2,
            epsilon=1e-8,
        )
        assert np.isfinite(ll)
        assert np.isfinite(logdet)
        assert qf >= 0.0


def test_fit_t_bayesian_laplace_returns_interval(additive_landscape):
    """
    MAP t should lie in [t_min, t_max], with a finite log-posterior and a positive
    Laplace variance approximation producing a sensible 95% CI.
    """
    G = additive_landscape.graph
    signal = additive_landscape.get_signal()

    t_min, t_max = 0.01, 10.0
    t_map, ci_lo, ci_hi, logpost_map, var_approx = fit_t_bayesian_laplace(
        G, signal, t_min=t_min, t_max=t_max, epsilon=1e-8
    )

    assert t_min <= t_map <= t_max
    assert t_min <= ci_lo <= ci_hi <= t_max
    assert np.isfinite(logpost_map)
    assert var_approx > 0.0


def test_covariance_psd_and_variance_match(additive_landscape):
    """
    Sigma(t) should be symmetric PSD and its average marginal variance should
    match the empirical variance used to scale the kernel.
    """
    G = additive_landscape.graph
    signal = additive_landscape.get_signal()
    # Precompute spectral stats & empirical variance (centered)
    f_hat, eigenvalues, sigma2 = _precompute_GMRF_stats(G, signal)
    _, U = eigenmode_decomposition(G, matrix='norm_laplacian')

    t = 0.3
    Sigma = _compute_variances(eigenvectors=U,
                               eigenvalues=eigenvalues,
                               sigma_squared=sigma2,
                               t=t,
                               epsilon=1e-8)

    # Symmetry
    assert np.allclose(Sigma, Sigma.T, atol=1e-10)

    # PSD: all eigenvalues >= -tiny_num
    w = np.linalg.eigvalsh(Sigma)
    assert w.min() > -1e-10

    # Mean of diagonal equals sigma^2 (by construction of the scaling)
    avg_var = float(np.mean(np.diag(Sigma)))
    assert np.isclose(avg_var, sigma2, rtol=1e-5, atol=1e-8)


def test_energy_local_global_consistency(additive_landscape):
    """
    The expected global Dirichlet energy should equal the sum of expected
    local contributions, within numerical tolerance.
    """
    # Use normalized Laplacian by default (matches heat-kernel prior)
    res = compute_ruggedness_variance_energy(additive_landscape, t=0.5, normalized=True)
    sigma = res['covariance_matrix']
    local = res['expected_local_energy']
    global_E = res['expected_global_energy']

    # Sum of locals should match the global energy
    assert np.isclose(local.sum(), global_E, rtol=1e-6, atol=1e-9)

def test_energy_monotone_smoothing_in_t(additive_landscape):
    """
    As diffusion time t increases, the expected global Dirichlet energy
    under the heat-kernel GMRF prior should decrease (smoother).
    """
    # Two scales t1 < t2
    t1, t2 = 0.05, 0.5
    res1 = compute_ruggedness_variance_energy(additive_landscape, t=t1, normalized=True)
    res2 = compute_ruggedness_variance_energy(additive_landscape, t=t2, normalized=True)

    E1 = res1['expected_global_energy']
    E2 = res2['expected_global_energy']

    # Expect smoothing: E(t2) <= E(t1)
    assert E2 <= E1 + 1e-9  # small tolerance


def test_compute_ruggedness_diffusion_scale_wrapper(additive_landscape):
    """
    Wrapper should return a valid t_map and confidence interval dictionary with
    expected keys. t_map must be inside [t_min, t_max].
    """
    t_min, t_max = 0.01, 2.0
    out = compute_ruggedness_diffusion_scale(additive_landscape, t_min=t_min, t_max=t_max)
    for key in [
        't_map',
        't_lower_confidence_interval',
        't_upper_confidence_interval',
        't_logposterior_map',
        'variance_approximate',
    ]:
        assert key in out

    assert t_min <= out['t_map'] <= t_max
    assert t_min <= out['t_lower_confidence_interval'] <= out['t_upper_confidence_interval'] <= t_max
    assert out['variance_approximate'] > 0.0


def test_variance_energy_honors_user_t(additive_landscape):
    """
    compute_ruggedness_variance_energy should honor a user-specified t exactly.
    """
    t_forced = 0.123
    res = compute_ruggedness_variance_energy(additive_landscape, t=t_forced)
    assert np.isclose(res['t_used'], t_forced)