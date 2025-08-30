import numpy as np

from fitness_landscape.utils import make_latent_geometric_graph_connected, sample_observed_induced_connected


def test_sample_observed_induced_connected_graph_return_graph():
    G = make_latent_geometric_graph_connected(n_latent=15, d_target=3, k_edges=8, seed=1)
    G2 = sample_observed_induced_connected(G, node_keep=0.5, edge_keep=0.5, seed=2, return_graph=True)
    # Connected and subset of nodes
    assert len(G2) <= len(G)
    assert G2.number_of_nodes() >= 2


def test_sample_observed_induced_connected_from_landscape(binary_3bit_landscape):
    L = binary_3bit_landscape
    # ensure positions for weight inference if needed
    for i, (u, data) in enumerate(L.graph.nodes(data=True)):
        data['pos'] = np.array([i, 0.0])
    sub = sample_observed_induced_connected(L, node_keep=0.5, edge_keep=0.5, seed=3)
    # Returns a FitnessLandscape
    from fitness_landscape.core.landscape import FitnessLandscape
    assert isinstance(sub, FitnessLandscape)
    assert sub.graph.number_of_nodes() >= 2

