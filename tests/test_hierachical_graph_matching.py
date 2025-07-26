import pytest
import numpy as np
import networkx as nx
from typing import List, Dict
from fitness_landscape.hierarchical_aligner import HierarchicalRJMCMCAligner
from fitness_landscape.graph_matching.latent_alignment RJMCMCAligner

@pytest.fixture
def mock_graphs() -> List[nx.Graph]:
    """

    """
    graphs = [nx.Graph(), nx.Graph()]
    
    # Cluster A embeddings (high similarity within cluster)
    emb_a1 = np.array([0.9, 0.1, 0.0, 0.0])
    emb_a2 = np.array([0.85, 0.15, 0.0, 0.0])
    
    # Cluster B embeddings (high similarity within cluster)
    emb_b1 = np.array([0.0, 0.0, 0.9, 0.1])
    emb_b2 = np.array([0.0, 0.0, 0.85, 0.15])

    # Graph 0
    graphs[0].add_node(0, emb_arr=emb_a1)
    graphs[0].add_node(1, emb_arr=emb_a2)
    graphs[0].add_node(2, emb_arr=emb_b1)
    graphs[0].add_node(3, emb_arr=emb_b2)
    graphs[0].add_edges_from([(0, 1), (2, 3)])

    # Graph 1
    graphs[1].add_node(0, emb_arr=emb_a1)
    graphs[1].add_node(1, emb_arr=emb_a2)
    graphs[1].add_node(2, emb_arr=emb_b1)
    graphs[1].add_node(3, emb_arr=emb_b2)
    graphs[1].add_edges_from([(0, 1), (2, 3)])
    
    return graphs

@pytest.fixture
def aligner_params() -> Dict:
    """
    Provides a default set of parameters for the RJMCMCAligner.
    """
    return {
        "alpha": 0.5,
        "burn_in": 100,
        "samples": 100,
        "thin": 5
    }

@pytest.fixture
def mock_rjmcmc_sample(mocker):
    """
    Mocks the RJMCMCAligner.sample() and subsequent result methods.
    This is CRUCIAL for making tests fast and deterministic.
    """
    def mock_sample(self):
        # Simulate the creation of internal state after sampling
        self._stored_L = [np.array([[0, 1], [1, 0]])]  
        self._stored_pi = {
            k: [np.array([0, 1])] for k, g in enumerate(self.graphs) if g.number_of_nodes() > 0
        }

    mocker.patch(
        "your_package.latent_alignment.RJMCMCAligner.sample",
        mock_sample
    )

def test_hierarchical_aligner_initialization(mock_graphs,
                                             aligner_params):
    """
    Tests that the aligner initializes without errors.
    """
    aligner = HierarchicalRJMCMCAligner(
        graphs=mock_graphs,
        aligner_params=aligner_params
    )
    assert aligner.K == 2
    assert aligner.local_thresh == 0.85
    assert aligner.global_thresh == 0.5

def test_create_clusters(mock_graphs, aligner_params):
    """
    Tests that the clustering logic correctly separates the two distinct groups of nodes.
    """
    aligner = HierarchicalRJMCMCAligner(
        graphs=mock_graphs,
        aligner_params=aligner_params,
        local_cluster_threshold=0.8, 
    )
    clusters = aligner._create_clusters()

    assert len(clusters) == 2

    cluster_nodes = {frozenset(c['global_indices']) for c in clusters}
    
    # Global indices: G0(0,1,2,3), G1(0,1,2,3) : 0,1,2,3,4,5,6,7
    expected_cluster_a = frozenset([0, 1, 4, 5]) # Nodes 0,1 from G0 and 0,1 from G1
    expected_cluster_b = frozenset([2, 3, 6, 7]) # Nodes 2,3 from G0 and 2,3 from G1
    
    assert expected_cluster_a in cluster_nodes
    assert expected_cluster_b in cluster_nodes


def test_run_local_alignments(mock_graphs,
                              aligner_params,
                              mock_rjmcmc_sample):
    """
    Tests that the local alignment process is orchestrated correctly.
    """
    aligner = HierarchicalRJMCMCAligner(
        graphs=mock_graphs,
        aligner_params=aligner_params
    )
    local_results = aligner._run_local_alignments()

    assert isinstance(local_results, list)
    assert len(local_results) == 2 # Two clusters should be found
    
    # Check the structure of the results for one cluster
    result = local_results[0]
    assert 'blueprint' in result
    assert 'node_mapping' in result
    assert 'node_order' in result
    assert isinstance(result['blueprint'], nx.Graph)
    assert isinstance(result['node_mapping'], dict)


def test_full_run_alignment_end_to_end(mock_graphs,
                                       aligner_params,
                                       mock_rjmcmc_sample):
    """
    An end-to-end integration test for the entire hierarchical process.
    """
    aligner = HierarchicalRJMCMCAligner(
        graphs=mock_graphs,
        aligner_params=aligner_params,
        local_cluster_threshold=0.8,
        global_bridge_threshold=0.1 # Ensure bridge is found
    )

    final_graph, final_mappings = aligner.run_alignment()

    # Test the final graph structure
    assert isinstance(final_graph, nx.Graph)
    # With two local 2-node blueprints, we expect 4 nodes total
    assert final_graph.number_of_nodes() == 4
    # The stitching logic should add a "bridge" edge, making the graph connected
    assert nx.is_connected(final_graph)

    # Test the final mappings structure
    assert isinstance(final_mappings, dict)
    assert len(final_mappings) == len(mock_graphs) # One mapping per original graph

    # Test the dimensions and content of the probability matrices
    for graph_idx, prob_matrix in final_mappings.items():
        num_original_nodes = mock_graphs[graph_idx].number_of_nodes()
        num_final_latent_nodes = final_graph.number_of_nodes()
        
        assert prob_matrix.shape == (num_original_nodes, num_final_latent_nodes)
        
        assert np.allclose(prob_matrix.sum(axis=1), 1.0)


def test_edge_case_empty_graphs(aligner_params,
                                mock_rjmcmc_sample):
    """
    Tests that the aligner handles empty input graphs gracefully.
    """
    empty_graphs = [nx.Graph(), nx.Graph()]
    aligner = HierarchicalRJMCMCAligner(
        graphs=empty_graphs,
        aligner_params=aligner_params
    )
    
    final_graph, final_mappings = aligner.run_alignment()

    assert final_graph.number_of_nodes() == 0
    assert final_mappings[0].shape == (0, 0)
    assert final_mappings[1].shape == (0, 0)