import numpy as np
import pytest
from fitness_landscape.utils import cosine_similarity_matrix, get_landscape_dist_mat, make_latent_geometric_graph_connected, sample_observed_induced_connected
from fitness_landscape.core.landscape import FitnessLandscape
from fitness_landscape.core.sequence import generate_sequences
from fitness_landscape.core.fitness import NumericFitness
import networkx as nx


@pytest.fixture
def util_landscape():
    """Provides a basic FitnessLandscape for utility testing."""
    sequences = generate_sequences(length=3, alphabet=[0, 1])
    fitness_values = [[val] for val in np.random.rand(8)]
    fitness_layers = {
        'default': NumericFitness(name='default', values=fitness_values)
    }
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type='hamming'
    )

def test_cosine_similarity_matrix():
    """Tests the cosine similarity matrix computation."""
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 1], [1, -1]])
    sim_matrix = cosine_similarity_matrix(A, B)
    assert np.allclose(sim_matrix, np.array([[0.70710678, 0.70710678], [0.70710678, -0.70710678]]))

def test_get_landscape_dist_mat(util_landscape):
    """Tests getting the landscape distance matrix."""
    dist_mat = get_landscape_dist_mat(util_landscape)
    assert dist_mat.shape == (len(util_landscape.sequences), len(util_landscape.sequences))
    assert np.all(np.diag(dist_mat) == 0)

def test_construct_geometric_graph():
    """Test for synthetic graph construction"""
    # Small synthetic graph
    G = make_latent_geometric_graph_connected(n_latent = 20,
                                              d_target = 4,
                                              k_edges = 16,
                                              seed = 42)

    assert G.number_of_nodes() == 20
    assert np.isclose(np.mean(np.array(list(dict(G.degree()).values()))), 4, 0.2)
    assert nx.is_connected(G)
    assert not G.is_directed()

def test_geometric_graph_induction():
    """Test synthetic graph induction"""
    G = make_latent_geometric_graph_connected(n_latent = 20,
                                              d_target = 4,
                                              k_edges = 16,
                                              seed = 42)

    G_ind = sample_observed_induced_connected(G, node_keep=0.5, edge_keep=0.5, seed=42)

    assert np.isclose((G.number_of_nodes() * 0.5), G_ind.number_of_nodes(), 2)
    assert np.isclose((G.number_of_edges() * 0.5), G_ind.number_of_edges(), 5)
    assert nx.is_connected(G_ind)

