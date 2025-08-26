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
from fitness_landscape.core.superscape import FitnessSuperscape
from unittest.mock import MagicMock
from fitness_landscape.analysis.coupling import *
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
from fitness_landscape._const import PROT_20


@pytest.fixture
def mock_superscape_for_posterior_analysis(mocker):
    """
    Provides a FitnessSuperscape instance where sample_latent_landscapes
    is mocked to return a predictable ensemble of simple landscapes.
    """
    # 1. Create a dummy superscape object. Its contents don't matter
    #    as we will mock the sampling method.
    superscape = MagicMock(spec=FitnessSuperscape)
    superscape.latent_landscape = True # To pass the initial check

    # 2. Create the predictable ensemble of landscapes to be "returned" by the mock
    ensemble = []
    # Sample 1: A 4-node path graph
    g1 = nx.path_graph(4)
    s1 = generate_sequences(length=2, alphabet=[0,1])
    for i, seq in enumerate(s1): g1.nodes[i]['sequence'] = seq
    ensemble.append(FitnessLandscape(sequences=s1, graph=g1))

    # Sample 2: A 5-node complete graph
    g2 = nx.complete_graph(5)
    s2 = generate_sequences(length=3, alphabet=[0,1])[:5] # just need 5
    for i, seq in enumerate(s2): g2.nodes[i]['sequence'] = seq
    ensemble.append(FitnessLandscape(sequences=s2, graph=g2))

    # 3. Mock the sample_latent_landscapes method
    mocker.patch.object(
        superscape,
        'sample_latent_landscapes',
        return_value=ensemble
    )
    
    return superscape

@pytest.fixture
def mock_superscape_with_posterior(mocker):
    """
    """
    alphabet = PROT_20
    sequences1 = generate_sequences(length=1, alphabet=alphabet)
    num_seq = len(sequences1)
    fitness1 = NumericFitness(name="default", values=[[v] for v in np.random.rand(num_seq)])
    landscape1 = FitnessLandscape.from_sequences(sequences1, fitness_layers={"default": fitness1})
    for i, node in enumerate(landscape1.graph.nodes()):
        seq = landscape1.sequences[i]
        landscape1.graph.nodes[node]['emb_arr'] = np.random.rand(5)
        landscape1.graph.nodes[node]['ungapped_arr'] = seq.to_one_hot()


    sequences2 = generate_sequences(length=1, alphabet=alphabet)
    fitness2 = NumericFitness(name="default", values=[[v] for v in np.random.rand(num_seq)])
    landscape2 = FitnessLandscape.from_sequences(sequences2, fitness_layers={"default": fitness2})
    for i, node in enumerate(landscape2.graph.nodes()):
        seq = landscape2.sequences[i]
        landscape2.graph.nodes[node]['emb_arr'] = np.random.rand(5)
        landscape2.graph.nodes[node]['ungapped_arr'] = seq.to_one_hot()

    mock_aligner = MagicMock()
    
    L_sample1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    L_sample2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    mock_aligner.full_posterior_L = [L_sample1, L_sample2]

    num_latent_nodes = 3
    mapping1 = np.random.rand(num_seq, num_latent_nodes)
    mapping1 /= mapping1.sum(axis=1, keepdims=True)
    mapping2 = np.random.rand(num_seq, num_latent_nodes)
    mapping2 /= mapping2.sum(axis=1, keepdims=True)

    mock_aligner.full_posterior_mappings = [
        {0: mapping1, 1: mapping2},
        {0: mapping1, 1: mapping2}
    ]
    mock_aligner.directed = False

    mean_graph = nx.complete_graph(num_latent_nodes)
    mean_mappings = {0: mapping1, 1: mapping2}
    mock_aligner.run_alignment.return_value = (mean_graph, mean_mappings)

    mocker.patch(
        'fitness_landscape.core.superscape.HierarchicalRJMCMCAligner',
        return_value=mock_aligner
    )

    superscape = FitnessSuperscape([landscape1, landscape2], **{'burn_in': 1, 'samples': 1})
    
    graph = nx.complete_graph(num_latent_nodes)
    sequences = [SoftSequence(np.random.rand(2, 20), alphabet=PROT_20) for _ in range(num_latent_nodes)]
    for i, seq in enumerate(sequences):
        graph.nodes[i]['sequence'] = seq

    superscape.latent_graph = graph
    superscape.latent_landscape = FitnessLandscape(
        sequences=sequences,
        graph=superscape.latent_graph,
        fitness_layers={'default': NumericFitness('default', [[i] for i in range(num_latent_nodes)])}
    )

    return superscape

def _make_landscape_from_values(values, name="default"):
    """
    Build a tiny landscape with N=4 (16 nodes) and a single numeric layer
    from a 1D array `values` of length 16.
    """
    assert values.ndim == 1
    sequences = generate_sequences(length=4, alphabet=[0, 1])
    fitness_values = [[float(v)] for v in values]
    fitness_layers = {name: NumericFitness(name=name, values=fitness_values)}
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type="hamming",
    )

@pytest.fixture
def two_groups_far_apart():
    rng = np.random.default_rng(123)
    a = rng.normal(loc=0.0, scale=1.0, size=200)
    b = rng.normal(loc=1.5, scale=1.0, size=200)
    return {"A": a, "B": b}

@pytest.fixture
def two_groups_equal():
    rng = np.random.default_rng(456)
    a = rng.normal(loc=0.5, scale=1.0, size=200)
    b = rng.normal(loc=0.5, scale=1.0, size=200)
    return {"A": a, "B": b}

@pytest.fixture
def two_landscapes_different():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, size=16)
    y = x + 1.2  # strong shift to ensure significance
    L1 = _make_landscape_from_values(x, name="default")
    L2 = _make_landscape_from_values(y, name="default")
    return {"L1": L1, "L2": L2}

@pytest.fixture
def one_landscape_two_layers():
    rng = np.random.default_rng(8)
    base = rng.normal(0.0, 1.0, size=16)
    alt  = 1.0 + 1.1 * base

    sequences = generate_sequences(length=4, alphabet=[0, 1])
    layer1 = NumericFitness(name="default", values=[[float(v)] for v in base])
    layer2 = NumericFitness(name="copy",    values=[[float(v)] for v in alt])
    L = FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers={"default": layer1, "copy": layer2},
        graph_type="hamming",
    )
    return L


@pytest.fixture
def cubic_only_landscape():
    """
    Synthetic landscape on N=4 that contains ONLY up to cubic interactions
    in z_i = (-1)^{x_i}. Therefore its Walsh spectrum has no order-4 mass.
    """
    rng = np.random.default_rng(12345)
    N = 4
    seqs = generate_sequences(N, [0, 1])  # full 2^N set

    # Map each 0/1 sequence x -> z in {+1,-1}^N by z_i = (-1)^{x_i}
    Z = np.array([1 - 2 * s.to_array().astype(int) for s in seqs])  # shape (16, 4)

    # Build f(z) = a0 + a^T z + z^T B z + sum_{i<j<k} c_{ijk} z_i z_j z_k
    a0 = 0.1
    a = rng.normal(0, 0.2, size=N)

    B = rng.normal(0, 0.1, size=(N, N))
    B = np.triu(B, 1)
    B = B + B.T  # symmetric, zero diagonal

    triples = [(0, 1, 2), (1, 2, 3)]
    c = {t: rng.normal(0, 0.05) for t in triples}

    f = np.full(Z.shape[0], a0, dtype=float)
    f += Z @ a
    f += 0.5 * np.sum((Z @ B) * Z, axis=1)  # quadratic form
    for (i, j, k), w in c.items():
        f += w * (Z[:, i] * Z[:, j] * Z[:, k])

    layer = NumericFitness(name="default", values=[[float(x)] for x in f])
    L = FitnessLandscape.from_sequences(
        sequences=seqs, fitness_layers={"default": layer}, graph_type="hamming"
    )
    return L

@pytest.fixture
def two_layer_additive(additive_landscape):
    """
    Add a second numeric layer identical to the default layer.
    For a proportional copy, Walsh/GFT coherence should be ~1 wherever power>0.
    """
    vals = additive_landscape.active_layer.to_scalar()
    copy_layer = NumericFitness(name='copy', values=[[float(v)] for v in vals])
    additive_landscape.attach(copy_layer)
    return additive_landscape


@pytest.fixture
def simple_gft_landscape():
    """
    Build a small non-Hamming landscape on a path graph (n=6) with two numeric layers.
    """
    G = nx.path_graph(6)
    # annotate nodes with sequences and two numeric layers
    for i in G.nodes():
        G.nodes[i]['sequence'] = BaseNumpySequence([i])  # trivial sequence
        G.nodes[i]['fitness_default'] = float(i) # linear ramp
        G.nodes[i]['fitness_copy'] = float(i) * 2.0  # proportional copy
        G.nodes[i]['gapped_arr'] = np.zeros((1, 21))
        G.nodes[i]['ungapped_arr'] = np.zeros((1, 20))
    L = FitnessLandscape.from_graph(G, emb_nodes=False)

    assert 'default' in L.fitness_layers and 'copy' in L.fitness_layers
    return L


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

def test_walsh_on_epistatic_landscape(epistatic_landscape):
    """
    For an NK landscape with N=4, K=2, there should be non-zero higher-order
    (>= 2) Walsh coefficients (interactions up to order K+1 = 3 expected).
    """
    results = calculate_epistasis_walsh(epistatic_landscape, order=4)
    nonzero_higher = []
    for order, coeffs in results['by_order'].items():
        if order >= 2:
            nonzero_higher.extend([abs(v) for v in coeffs.values()])
    assert any(v > 1e-8 for v in nonzero_higher), \
        "Expected at least one non-zero higher-order Walsh coefficient for K=2."

    ve = results.get('variance_explained')
    if ve is not None:
        higher_mass = sum(ve.get(o, 0.0) for o in range(2, 5))

        assert higher_mass > 0.05, \
            f"Expected >5% variance in orders >=2 for K=2, got {higher_mass:.3f}."
        assert ve.get(1, 0.0) < 0.95, \
            f"Order-1 variance should not explain ~all signal for K=2, got {ve.get(1, 0.0):.3f}."

# NK "leaks" higher order interactions.
def test_walsh_quartic_zero_on_cubic_landscape(cubic_only_landscape):
    """
    A cubic-only polynomial in z_i = (-1)^{x_i} has no 4th-order Walsh content.
    """
    results = calculate_epistasis_walsh(cubic_only_landscape, order=4)

    # Ensure order-4 entries are present and (numerically) zero
    order4 = results['by_order'].get(4, {})
    assert order4, "Order-4 bucket should exist when requesting order=4."

    mags = [abs(v) for v in order4.values()]
    assert np.isclose(max(mags), 0), f"Unexpected 4th-order mass: max {max(mags):.3e}"

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


def test_eigenmode_decomposition_laplacian():
    """Tests eigenmode decomposition."""
    graph = nx.path_graph(4)
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='laplacian')
    assert eigenvalues is not None
    assert eigenvectors is not None

def test_eigenmode_decomposition_norm_laplacian():
    """Tests eigenmode decomposition."""
    graph = nx.path_graph(4)
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='norm_laplacian')
    assert eigenvalues is not None
    assert eigenvectors is not None

def test_eigenmode_decomposition_adj():
    """Tests eigenmode decomposition."""
    graph = nx.path_graph(4)
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='adjacency')
    assert eigenvalues is not None
    assert eigenvectors is not None

def test_eigenmode_decomposition_transition():
    """Tests eigenmode decomposition."""
    graph = nx.path_graph(4)
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix='transition')
    assert eigenvalues is not None
    assert eigenvectors is not None


def test_graph_spectral_analysis(additive_landscape):
    """Tests graph spectral analysis."""
    results = graph_spectral_analysis(additive_landscape, matrix='laplacian')
    assert 'eigenvalues' in results
    assert 'participation_ratios' in results
    assert 'localization' in results
    assert 'node_centralities' in results

    results = graph_spectral_analysis(additive_landscape, matrix='norm_laplacian')
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

def test_coherence_auto_picks_walsh(two_layer_additive):
    """
    On a full Hamming cube, basis='auto' should select Walsh; evals should be None.
    Bands should be returned per epistatic order.
    """
    res = cross_spectral_coherence(two_layer_additive, ['default', 'copy'], basis='auto', walsh_aggregate='order')
    assert 'evals' in res and res['evals'] is None
    assert 'bands' in res

    # Orders 0..L present
    L = int(np.log2(len(two_layer_additive.sequences)))
    for r in range(L + 1):
        assert f'order_{r}' in res['bands']


def test_coherence_walsh_identity_structure(two_layer_additive):
    """
    For identical layers on an additive (K=0) landscape, order-0 and order-1 coherence ~1,
    higher-order coherence ~0.
    """
    res = cross_spectral_coherence(two_layer_additive, ['default', 'copy'], basis='walsh', walsh_aggregate='order')
    L = int(np.log2(len(two_layer_additive.sequences)))

    # Off-diagonal coherence between the two layers
    c0 = res['bands']['order_0'][0, 1]
    c1 = res['bands']['order_1'][0, 1]
    assert c0 == pytest.approx(1.0, abs=1e-12)
    assert c1 == pytest.approx(1.0, abs=1e-12)

    for r in range(2, L + 1):
        # No power in higher orders for K=0 -> coherence should average to ~0
        assert res['bands'][f'order_{r}'][0, 1] == pytest.approx(0.0, abs=1e-12)


def test_coherence_walsh_per_mode_shapes(two_layer_additive):
    """
    Per-mode coherence list length should equal number of Walsh modes (=2^L),
    and each item is an (N x N) matrix with N=number of layers.
    """
    res = cross_spectral_coherence(two_layer_additive, ['default', 'copy'], basis='walsh', walsh_aggregate='none')
    K = 1 << int(np.log2(len(two_layer_additive.sequences)))
    assert len(res['coherence']) == K
    for mat in res['coherence']:
        assert mat.shape == (2, 2)


def test_coherence_walsh_phase_option(two_layer_additive):
    """
    return_phase=True should provide a phase array per mode with the same shape as coherence.
    """
    res = cross_spectral_coherence(two_layer_additive, ['default', 'copy'], basis='walsh', walsh_aggregate='none', return_phase=True)
    assert 'phase' in res
    assert len(res['phase']) == len(res['coherence'])
    for ph, coh in zip(res['phase'], res['coherence']):
        assert ph.shape == coh.shape


def test_coherence_gft_default_bands(simple_gft_landscape):
    """
    On a non-Hamming graph with two proportional layers, basis='gft' should return
    eigenvalues and default 'low','mid','high' band aggregation.
    """
    res = cross_spectral_coherence(simple_gft_landscape, ['default', 'copy'], basis='gft')
    assert res['evals'] is not None
    assert isinstance(res['evals'], np.ndarray) and res['evals'].ndim == 1

    assert 'bands' in res
    for name in ['low', 'mid', 'high']:
        assert name in res['bands']
        assert res['bands'][name].shape == (2, 2)

    band_vals = [res['bands'][b][0, 1] for b in ['low', 'mid', 'high']]
    assert max(band_vals) > 0.9


def test_coherence_gft_per_mode_shapes(simple_gft_landscape):
    """
    Per-mode coherence length equals number of retained eigenmodes; each (N x N).
    """
    res = cross_spectral_coherence(simple_gft_landscape, ['default', 'copy'], basis='gft', n_eigs=None)
    K = len(res['coherence'])
    n = simple_gft_landscape.graph.number_of_nodes()
    assert K == n
    for mat in res['coherence']:
        assert mat.shape == (2, 2)


def test_coherence_gft_with_phase(simple_gft_landscape):
    """
    Phase returned with GFT path has same per-mode shape as coherence.
    """
    res = cross_spectral_coherence(simple_gft_landscape, ['default', 'copy'], basis='gft', return_phase=True)
    assert 'phase' in res
    assert len(res['phase']) == len(res['coherence'])
    for ph, coh in zip(res['phase'], res['coherence']):
        assert ph.shape == coh.shape

def test_hypothesis_testing_on_groups(two_groups_far_apart):
    res = hypothesis_testing(groups=two_groups_far_apart)
    # structure
    assert "group_stats" in res and "pairwise_tests" in res
    # significance (t-test, U, KS should all be significant for far apart groups)
    pair = res["pairwise_tests"]["A"]["B"]
    assert pair["t_test"]["significant"]
    assert pair["mann_whitney"]["significant"]
    assert pair["ks_test"]["significant"]

def test_hypothesis_testing_on_groups_nonsignificant(two_groups_equal):
    res = hypothesis_testing(groups=two_groups_equal)
    pair = res["pairwise_tests"]["A"]["B"]
    # at least t-test should be non-significant
    assert not pair["t_test"]["significant"]

def test_hypothesis_testing_on_landscapes(two_landscapes_different):
    # value_fn extracts the active layer's scalar signal
    def value_fn(L):
        return L.get_signal().astype(float)

    res = hypothesis_testing(landscapes=two_landscapes_different, value_fn=value_fn)
    assert "group_stats" in res and "pairwise_tests" in res
    pair = res["pairwise_tests"]["L1"]["L2"]
    assert pair["t_test"]["significant"]

def test_hypothesis_testing_on_layers(one_landscape_two_layers):
    L = one_landscape_two_layers

    def value_fn_layers(landscape, layer_name):
        # switch active layer and read scalars
        prev = landscape._active_view_name
        try:
            landscape.view(layer_name)
            return landscape.active_layer.to_scalar().astype(float)
        finally:
            if prev is not None and prev in landscape.fitness_layers:
                landscape.view(prev)

    res = hypothesis_testing(
        landscape=L,
        layer_names=["default", "copy"],
        value_fn_layers=value_fn_layers,
    )
    assert "pairwise_tests" in res
    assert res["pairwise_tests"]["default"]["copy"]["t_test"]["significant"]

def test_hypothesis_testing_raises_on_conflicting_inputs(two_groups_far_apart, two_landscapes_different):
    # Passing groups and landscapes together should raise
    with pytest.raises(ValueError):
        hypothesis_testing(groups=two_groups_far_apart,
                           landscapes=two_landscapes_different,
                           value_fn=lambda L: L.get_signal())



def _value_fn_layers(L: FitnessLandscape, lname: str) -> np.ndarray:
    return L.get_layer(lname).to_scalar()

def _value_fn_landscape(L: FitnessLandscape) -> np.ndarray:
    # Use the landscape's active layer (default)
    return L.active_layer.to_scalar()

# --- Tests ---

def test_permutation_test_on_groups_two_sided():
    rng = np.random.default_rng(42)
    g1 = rng.normal(0.0, 1.0, size=200)
    g2 = rng.normal(0.5, 1.0, size=200)

    res = permutation_test(groups={"A": g1, "B": g2}, n_permutations=1000, alpha=0.05, alternative="two-sided")
    out = res[("A", "B")]

    assert "observed" in out and "p_value" in out and "significant" in out
    assert out["p_value"] <= 0.05
    assert out["significant"] is True

def test_permutation_test_on_groups_directional():
    rng = np.random.default_rng(123)
    g1 = rng.normal(0.0, 1.0, size=300)
    g2 = rng.normal(0.5, 1.0, size=300)

    # statistic_func = mean(g1) - mean(g2), so we expect it to be negative on average.
    stat = lambda a, b: float(np.mean(a) - np.mean(b))

    # "less": mean(g1)-mean(g2) is often < 0, so p should be small and significant True.
    res = permutation_test(groups={"A": g1, "B": g2}, statistic_func=stat, n_permutations=1000, alpha=0.05, alternative="less")
    out = res[("A", "B")]

    assert out["observed"] < 0.0
    assert out["p_value"] <= 0.05
    assert out["significant"] is True

def test_permutation_test_nonsignificant():
    rng = np.random.default_rng(7)
    g1 = rng.normal(1.0, 1.0, size=250)
    g2 = rng.normal(1.0, 1.0, size=250)

    res = permutation_test(groups={"A": g1, "B": g2}, n_permutations=800, alpha=0.05, alternative="two-sided")
    out = res[("A", "B")]

    assert out["p_value"] > 0.05
    assert out["significant"] is False

def test_permutation_test_on_landscapes(additive_landscape, epistatic_landscape):
    # Structure check: ensure we return an entry for the pair and the keys exist.
    landscapes = {
        "add": additive_landscape,
        "epi": epistatic_landscape,
    }
    res = permutation_test(landscapes=landscapes, value_fn=_value_fn_landscape, n_permutations=100, alpha=0.05)
    assert ("add", "epi") in res
    out = res[("add", "epi")]
    for key in ["observed", "p_value", "significant", "n_permutations", "alternative"]:
        assert key in out

def test_permutation_test_on_layers(two_layer_additive):
    # Compare default vs copy layer. Means differ -> often significant.
    res = permutation_test(
        landscape=two_layer_additive,
        layer_names=["default", "copy"],
        value_fn_layers=_value_fn_layers,
        n_permutations=500,
        alpha=0.05,
        alternative="two-sided",
    )
    out = res[("default", "copy")]
    for key in ["observed", "p_value", "significant", "n_permutations", "alternative"]:
        assert key in out
    assert out["significant"] in (True, False)

def test_subsample_scalar_dirichlet_energy_structure(additive_landscape):
    """
    subsample_analysis should handle a scalar-returning analysis_fn and
    return a 'samples' list and a 'summary' dict with mean/std/ci.
    """
    out = subsample_analysis(
        landscape=additive_landscape,
        analysis_func=lambda L: calculate_ruggedness_dirichlet_energy(L)["total_dirichlet_energy"],
        n_samples=25,
        subsample_node_prop=0.9,
        subsample_edge_prop=0.9,
        seed=123,  # if your subsampler accepts seed; otherwise drop
    )

    assert "results" in out and isinstance(out["results"], list)
    assert len(out["results"]) == 25
    assert all(np.isscalar(x) for x in out["results"])

    assert "summary" in out and isinstance(out["summary"], dict)
    for k in ["mean", "std", "ci_low", "ci_high", "alpha"]:
        assert k in out["summary"]

    # Energy should be finite and non-negative
    assert np.isfinite(out["summary"]["mean"])
    assert out["summary"]["mean"] >= 0.0


def test_subsample_dict_dirichlet_energy_aggregation(additive_landscape):
    """
    If analysis_fn returns a dict of numeric metrics, subsample_analysis
    should report 'per_key' aggregation with per-metric summary stats.
    """
    out = subsample_analysis(
        landscape=additive_landscape,
        analysis_func=lambda L: calculate_ruggedness_dirichlet_energy(L, weighted_laplacian=True),
        n_samples=15,
        subsample_node_prop=0.8,
        subsample_edge_prop=0.8,
        seed=7,
    )

    assert "per_key" in out and isinstance(out["per_key"], dict)
    # At minimum we expect total_dirichlet_energy aggregated
    assert "total_dirichlet_energy" in out["per_key"]
    per = out["per_key"]["total_dirichlet_energy"]
    for k in ["mean", "std", "ci_low", "ci_high", "alpha"]:
        assert k in per

def test_subsample_node_fraction_tracks_keep(additive_landscape):
    """
    If we use analysis_fn that returns number of nodes, the average of the
    subsample should be close to node_keep * N.
    """
    N = additive_landscape.graph.number_of_nodes()

    node_keep = 0.5
    out = subsample_analysis(
        landscape=additive_landscape,
        analysis_func=lambda L: L.graph.number_of_nodes(),
        n_samples=40,
        subsample_node_prop=node_keep,
        subsample_edge_prop=0.9,
        seed=2024,
    )

    avg_nodes = out["summary"]["mean"]
    # Within a reasonable tolerance of node_keep * N
    assert np.isclose(avg_nodes, node_keep * N, rtol=0.15, atol=1.0)

def test_subsample_local_dirichlet_mean(additive_landscape):
    """
    Show that vector outputs can be reduced in the lambda (mean over nodes).
    """
    out = subsample_analysis(
        landscape=additive_landscape,
        analysis_func=lambda L: float(np.mean(list(local_dirichlet_energy_contribution(L).values()))),
        n_samples=20,
        subsample_node_prop=0.85,
        subsample_edge_prop=0.9,
        seed=11,
    )

    assert "results" in out and len(out["results"]) == 20
    assert np.isfinite(out["summary"]["mean"])

def test_get_expected_latent_landscape(mock_superscape_with_posterior):
    """
    Tests that the expected latent landscape has a weighted graph where edge
    weights are the posterior probabilities of edge existence.
    """
    superscape = mock_superscape_with_posterior
    expected_landscape = superscape.sample_latent_landscapes(n_samples=1)[0]

    assert isinstance(expected_landscape, FitnessLandscape)
    
    graph = expected_landscape.graph
    assert graph.number_of_nodes() == 3
    assert graph[0][1]['weight'] == pytest.approx(1.0)

def test_sample_latent_landscapes(mock_superscape_with_posterior):
    """
    Tests that sampling returns a list of FitnessLandscape objects with graph
    structures consistent with the posterior samples.
    """
    superscape = mock_superscape_with_posterior
    n_samples = 10
    ensemble = superscape.sample_latent_landscapes(n_samples=n_samples)

    assert isinstance(ensemble, list)
    assert len(ensemble) == n_samples
    assert all(isinstance(land, FitnessLandscape) for land in ensemble)

    # Check that the graph structures in the ensemble match our two posterior samples
    num_path_graphs = 0
    num_complete_graphs = 0
    for land in ensemble:
        # A path graph on 3 nodes has 2 edges
        if land.graph.number_of_edges() == 2:
            num_path_graphs += 1
        # A complete graph on 3 nodes has 3 edges
        elif land.graph.number_of_edges() == 3:
            num_complete_graphs += 1

    assert num_path_graphs + num_complete_graphs == n_samples
    # With enough samples, we should see both types of graphs
    if n_samples > 5:
        assert num_path_graphs > 0
        assert num_complete_graphs > 0

def test_sample_posterior_analysis_scalar_output(mock_superscape_for_posterior_analysis):
    """
    """
    superscape = mock_superscape_for_posterior_analysis
    analysis_fn = lambda L: L.graph.number_of_nodes()
    results = sample_posterior_graph_analysis(
        superscape,
        analysis_func=analysis_fn,
        n_samples=2 
    )

    assert "results" in results
    assert "summary" in results
    assert results["results"] == [4, 5]

    summary = results["summary"]
    assert summary["mean"] == pytest.approx(4.5)
    assert summary["std"] == pytest.approx(np.std([4, 5], ddof=1))

def test_sample_posterior_analysis_dict_output(mock_superscape_for_posterior_analysis):
    """
    Tests that the analysis function correctly summarizes dictionary outputs.
    """
    superscape = mock_superscape_for_posterior_analysis

    def graph_metrics(L):
        return {
            "node_count": L.graph.number_of_nodes(),
            "edge_count": L.graph.number_of_edges(),
        }

    results = sample_posterior_graph_analysis(
        superscape,
        analysis_func=graph_metrics,
        n_samples=2
    )

    assert "results" in results
    assert "per_key" in results
    
    node_summary = results["per_key"]["node_count"]
    assert node_summary["samples"] == [4, 5]
    assert node_summary["mean"] == pytest.approx(4.5)

    edge_summary = results["per_key"]["edge_count"]
    assert edge_summary["samples"] == [3, 10]
    assert edge_summary["mean"] == pytest.approx(6.5)