import pytest
import numpy as np
import networkx as nx
from unittest.mock import patch

from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import BaseNumpySequence
from fitness_landscape.core.superscape import FitnessSuperscape
from fitness_landscape.core.fitness import NumericFitness

from fitness_landscape.graph_matching.latent_alignment import (
    BernoulliBeta,
    RJMCMCAligner,
    auto_anchors_by_cosine
)
from fitness_landscape.graph_matching.hierarchical_alignment import HierarchicalRJMCMCAligner

from fitness_landscape.graph_matching.minimum_spanning_graph import reconstruct_latent_graph_with_steiner, reconstruct_latent_graph_midpoint
from fitness_landscape.utils import (
    make_latent_geometric_graph_connected,
    sample_observed_induced_connected
)


def test_bernoulli_beta_log_marginal():
    """
    Tests the BernoulliBeta log marginal likelihood calculation.
    """
    bb = BernoulliBeta(alpha1=5, alpha0=2)

    logp1 = bb.log_marginal_edges(o_success=10, o_fail=2)
    logp2 = bb.log_marginal_edges(o_success=2, o_fail=10)
    assert logp1 > logp2

def test_auto_anchors_by_cosine():
    """
    Tests that nodes with high cosine similarity are correctly anchored.
    """
    g1 = nx.Graph()
    g1.add_node(0, emb_arr=np.array([1.0, 0.0, 0.1]))
    g1.add_node(1, emb_arr=np.array([0.0, 1.0, 0.0]))

    g2 = nx.Graph()
    g2.add_node(0, emb_arr=np.array([0.99, 0.0, 0.1])) # Similar to g1-0
    g2.add_node(1, emb_arr=np.array([0.0, 0.2, 1.0])) # Different

    graphs = [g1, g2]
    auto_anchors_by_cosine(graphs, cos_threshold=0.95)

    assert graphs[0].nodes[0]['anchor'] is True
    assert graphs[1].nodes[0]['anchor'] is True
    assert graphs[0].nodes[0]['anchor_id'] == graphs[1].nodes[0]['anchor_id']

    assert 'anchor' not in graphs[0].nodes[1]
    assert 'anchor' not in graphs[1].nodes[1]


@pytest.fixture
def two_simple_graphs():
    """
    Provides two simple graphs for testing the RJMCMCAligner.
    """
    g1 = nx.Graph()
    g1.add_node(0, emb_arr=np.array([1.0, 0.0]))
    g1.add_node(1, emb_arr=np.array([0.0, 1.0]))
    g1.add_edge(0, 1)

    g2 = nx.Graph()
    g2.add_node(0, emb_arr=np.array([0.9, 0.1]))
    g2.add_node(1, emb_arr=np.array([0.1, 0.9]))
    g2.add_edge(0, 1)

    return [g1, g2]


def test_rjmcmc_aligner_initialization(two_simple_graphs):
    """
    Tests that the RJMCMCAligner initializes correctly without errors.
    """
    aligner = RJMCMCAligner(two_simple_graphs, burn_in=1, samples=1, thin=1, auto_anchor=False)

    assert aligner.K == 2
    assert aligner.NL == 2
    assert aligner.C_global.shape == (2, 2)
    assert aligner.C_global[0, 1] == 2
    assert aligner.C_global[1, 0] == 2

def test_rjmcmc_birth_move(two_simple_graphs):
    """
    Tests the _birth move to ensure it adds a latent node correctly.
    """
    aligner = RJMCMCAligner(two_simple_graphs, burn_in=1, samples=1, thin=1, auto_anchor=False)

    aligner.perm[0][1] = -1
    initial_nl = aligner.NL

    was_born = aligner._birth()
    assert was_born is True
    assert aligner.NL == initial_nl + 1

    assert aligner.perm[0][1] == initial_nl

def test_rjmcmc_death_move(two_simple_graphs):
    """
    Tests the _death move to ensure it removes a latent node correctly.
    """
    aligner = RJMCMCAligner(two_simple_graphs, auto_anchor=False)

    initial_nl = aligner.NL
    aligner.NL += 1
    aligner.L = np.pad(aligner.L, ((0, 1), (0, 1)))
    aligner.C_global = np.pad(aligner.C_global, ((0, 1), (0, 1)))

    for i in range(len(aligner.C_k)):
        aligner.C_k[i] = np.pad(aligner.C_k[i], ((0, 1), (0, 1)))

    was_deleted = aligner._death()

    assert was_deleted is True
    assert aligner.NL == initial_nl

def test_rjmcmc_sample_run(two_simple_graphs):
    """
    Tests that the main sample() method runs to completion without errors.
    """
    try:
        aligner = RJMCMCAligner(two_simple_graphs, burn_in=10, samples=5, thin=2, auto_anchor=False, seed=42)
        # Explicitly call with a set number of chains for the test
        aligner.sample(num_chains=2)
    except Exception as e:
        pytest.fail(f"RJMCMCAligner.sample() raised an exception: {e}")

    # Assert the correct total number of samples (num_chains * samples).
    expected_samples = 2 * 5
    assert len(aligner._stored_L) == expected_samples
    assert len(aligner._stored_pi[0]) == expected_samples

    latent_graph = aligner.latent_blueprint_graph()
    assert isinstance(latent_graph, nx.Graph)

    prob_map = aligner.get_node_to_latent_mapping()
    assert isinstance(prob_map, dict)
    assert prob_map[0].shape[0] == 2

AMINO_ACID_ALPHABET = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

@pytest.fixture
def simple_landscape_factory():
    """
    Creates a simple but valid fitness landscape.
    """
    def _factory():
        alphabet = AMINO_ACID_ALPHABET
        seq_length = 4
        seq1_list = np.random.choice(alphabet, seq_length).tolist()
        seq1 = BaseNumpySequence(seq1_list, alphabet=alphabet)
        seq2 = seq1.mutate(positions=[1])
        sequences = [seq1, seq2]
        fitnesses = [0.5, 0.8]
        fitness_layers = {
            'default': NumericFitness(name='default', values=[[f] for f in fitnesses])
        }

        landscape = FitnessLandscape.from_sequences(
            sequences=sequences,
            fitness_layers=fitness_layers,
            graph_type='hamming'
        )

        for i, node in enumerate(landscape.graph.nodes()):
            landscape.graph.nodes[node]['emb_arr'] = np.random.rand(10)
            landscape.graph.nodes[node]['ungapped_arr'] = np.random.rand(seq_length, len(alphabet))
            landscape.graph.nodes[node]['gapped_arr'] = np.random.rand(seq_length, len(alphabet) + 1)

        return landscape
    return _factory

@pytest.fixture
def two_landscapes(simple_landscape_factory):
    """
    Provides two distinct landscape instances for testing.
    """
    return [simple_landscape_factory(), simple_landscape_factory()]

def test_rjmcmc_aligner_initialization(two_simple_graphs):
    aligner = RJMCMCAligner(two_simple_graphs, burn_in=1, samples=1, thin=1, auto_anchor=False)
    assert aligner.K == 2
    assert aligner.NL == 2
    assert aligner.C_global.shape == (2, 2)

    assert aligner.C_global[0, 1] == 2
    assert aligner.C_global[1, 0] == 2

def test_steiner_latent_graph_reconstruction():
    """Tests the steiner graph reconstruction of an observed graph"""
    G = make_latent_geometric_graph_connected(n_latent = 20,
                                              d_target = 4,
                                              k_edges = 16,
                                              seed = 42)

    G_ind = sample_observed_induced_connected(G, node_keep=0.5, edge_keep=0.5, seed=42)

    G_recon, _, _ = reconstruct_latent_graph_with_steiner(G_ind)

    assert G_recon.number_of_nodes() >= G_ind.number_of_nodes()
    assert G_recon.number_of_edges() >= G_ind.number_of_edges()
    assert nx.is_connected(G_recon)

def test_steiner_midpoint_latent_graph_reconstruction():
    """Tests the steiner graph reconstruction of an observed graph"""
    G = make_latent_geometric_graph_connected(n_latent = 20,
                                              d_target = 4,
                                              k_edges = 16,
                                              seed = 42)

    G_ind = sample_observed_induced_connected(G, node_keep=0.5, edge_keep=0.5, seed=42)

    G_recon = reconstruct_latent_graph_midpoint(G_ind)

    assert G_recon.number_of_nodes() >= G_ind.number_of_nodes()
    assert G_recon.number_of_edges() >= G_ind.number_of_edges()
    assert nx.is_connected(G_recon)