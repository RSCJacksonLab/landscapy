import numpy as np
import networkx as nx
from typing import Hashable, List, Union, Dict
from ..core.landscape import FitnessLandscape

def calculate_ruggedness_dirichlet_energy(landscape: FitnessLandscape,
                                          edge_weight_bins: Union[np.ndarray, List] = None,
                                          weighted_laplacian : bool = False) -> Dict:
    """
    Function to determine the analytical dirichlet energy of a fitness
    landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    edge_weight_bins : array-like, default=`None`
        The edge weight bins to collect edges into. If `None`,
        Dirichlet energy is determined as the global energy without
        edge contribution pooling. 
    
    weighted_laplacian : bool, default=False
        Boolean to indicate if a weighted Laplacian should be used.
    
    Returns
    -------
    results : Dict
        The Dirichlet energy results dict.
    """

    node_order = list(landscape.graph.nodes())
    if weighted_laplacian:
        # Weighted Laplacian for kNN graphs.
        laplacian = nx.laplacian_matrix(
            G=landscape.graph, nodelist=node_order, weight='weight'
        ).toarray()
    else:
        laplacian = nx.laplacian_matrix(
            G=landscape.graph, nodelist=node_order
        ).toarray()

    results = {}

    signal = landscape.get_node_signal(node_order)
    fitness_by_node = dict(zip(node_order, signal))

    # The formula for total Dirichlet energy is f' * L * f
    total_de = signal @ laplacian @ signal
    
    # Normalize by the number of nodes
    total_de_per_node = total_de / nx.number_of_nodes(landscape.graph)
    
    results = {
        'total_dirichlet_energy': total_de_per_node,
        'weighted_laplacian': weighted_laplacian,
    }

    if edge_weight_bins is not None:
        results['edge_weight_bins'] = {}
            
        # Iterate over each bin.
        for bin_range in edge_weight_bins:
            
            # Collect edges in the bin range.
            edge_list = _collect_edges(landscape=landscape, weight=bin_range)
            edge_de = 0.0
            
            for u, v in edge_list:
                fitness1 = fitness_by_node[u]
                fitness2 = fitness_by_node[v]
                
                current_edge_weight = landscape.graph.edges[u, v].get('weight', 1.0)
                edge_de += _sum_dirichlet_energy(fitness1=fitness1,
                                                fitness2=fitness2,
                                                weighted_edge=weighted_laplacian,
                                                edge_weight=current_edge_weight)
                
            bin_key = f"{bin_range}"
            results['edge_weight_bins'][f'{bin_key}_dirichlet_energy'] = edge_de
            # Normalize contribution by total energy (not per-node energy)
            if total_de > 0:
                results['edge_weight_bins'][f'{bin_key}_contribution'] = edge_de / total_de
            else:
                results['edge_weight_bins'][f'{bin_key}_contribution'] = 0.0

    return results

def _sum_dirichlet_energy(fitness1: float,
                          fitness2: float,
                          weighted_edge: bool = False,
                          edge_weight: float = None) -> float:
    """
    Helper function to compute the edge summed form of the Dirichlet
    energy. 
    """
    squared_diffs = (fitness1 - fitness2)**2
    if weighted_edge:
        if edge_weight is None:
            raise ValueError('Edge weights must be provided if weighting by an edge.')
        squared_diffs = squared_diffs * edge_weight
    
    return squared_diffs / 2

def _collect_edges(landscape: Union[FitnessLandscape, nx.Graph],
                   weight: np.ndarray) -> List[tuple[int, int]]:
    """
    Helper function to collect edges into distance bins. 
    """
    if isinstance(landscape, FitnessLandscape):
        graph = landscape.graph
    elif isinstance(landscape, nx.Graph):
        graph = landscape
    else:
        raise TypeError("landscape must be a FitnessLandscape or nx.Graph")
    
    min_weight, max_weight = weight
    selected_edges = []
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        if min_weight <= w < max_weight:
            selected_edges.append((u, v))
    return selected_edges

def local_dirichlet_energy_contribution(
    landscape: FitnessLandscape,
) -> Dict[Hashable, float]:
    """Calculate each node's local contribution to Dirichlet energy.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape with an active scalar fitness layer and an undirected graph.

    Returns
    -------
    dict of hashable to float
        Half the sum of squared fitness differences over each node's incident
        edges. Summing these values counts each undirected edge once overall.

    Raises
    ------
    TypeError
        If ``landscape`` is invalid or has no graph.
    """
    if not isinstance(landscape, FitnessLandscape) or landscape.graph is None:
        raise TypeError("Input must be a FitnessLandscape with an initialized graph.")

    graph = landscape.graph
    fitness_values = landscape.get_signal()
    local_energies = {}

    for node in graph.nodes():
        sequence_index = landscape.sequence_index_for_node(node)
        local_sum = 0
        for neighbor in graph.neighbors(node):
            neighbor_index = landscape.sequence_index_for_node(neighbor)
            local_sum += (fitness_values[sequence_index] - fitness_values[neighbor_index])**2
        local_energies[node] = 0.5 * local_sum
        
    return local_energies
