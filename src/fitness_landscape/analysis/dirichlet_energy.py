import numpy as np
import networkx as nx
from typing import List, Union, Dict
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
    
    Returns
    -------
    results : Dict
        The Dirichlet energy results dict.
    """

    if weighted_laplacian: 
        # Weighted Laplacian for kNN graphs.
        laplacian = nx.laplacian_matrix(G = landscape.graph, weight='weight')
    else:
        laplacian = nx.laplacian_matrix(G = landscape.graph)

    results = {}

    signal = landscape.get_signal()

    total_de = signal @ laplacian @ signal
    total_de = total_de / nx.number_of_nodes(landscape.graph)
    
    results = {
        'total_dirichlet_energy': total_de,
        'weighted_laplacian': weighted_laplacian,
    }

    if edge_weight_bins is not None:
        results['edge_weight_bins'] = {}
            
        # Iterate over each bin.
        for bin_range in edge_weight_bins:
            
            # Collect edges in the bin range.
            edge_list = _collect_edges(landscape=landscape.graph, weight=bin_range)
            edge_de = 0.0
            
            for u, v in edge_list:
                fitness1 = landscape.graph.nodes[u].get('fitness', 0.0)
                fitness2 = landscape.graph.nodes[v].get('fitness', 0.0)
                
                # Use the edge weight if available (default to 1.0 if missing)
                current_edge_weight = landscape.graph.edges[u, v].get('weight', 1.0)
                edge_de += _sum_dirichlet_energy(fitness1=fitness1,
                                                fitness2=fitness2,
                                                weighted_edge=weighted_laplacian,
                                                edge_weight=current_edge_weight)
                
            # Save the edge Dirichlet energy and its contribution.
            bin_key = f"{bin_range}"
            results['edge_weight_bins'][f'{bin_key}_dirichlet_energy'] = edge_de
            results['edge_weight_bins'][f'{bin_key}_contribution'] = edge_de / total_de

    return results

def _sum_dirichlet_energy(fitness1: float,
                          fitness2: float,
                          weighted_edge: bool = False,
                          edge_weight: float = None) -> float:
    """
    Helper function to compute the edge summed form of the Dirichlet
    energy. 

    Parameters
    ----------
    fitness1 : float
        Fitness value of first incident node. 
    
    fitness2 : float
        Fitness value of second incident node.
    
    weighted_edge : bool, default=`False`
        Boolean to weight the summed dirichlet energy ny a weight.

    egde_weight : float, default=`None`
        The edge weight.
    
    Returns
    -------
    float
        The summed dirichlet energy.
    """
    squared_diffs = (fitness1 - fitness2)**2
    if weighted_edge:
        assert edge_weight is not None, \
        'Edge weights must be included if weighting by an edge.'

        squared_diffs = squared_diffs * edge_weight
    
    return squared_diffs / 2

def _collect_edges(landscape: Union[FitnessLandscape, nx.Graph],
                   weight: np.ndarray) -> np.ndarray:
    """
    Helper function to collect edges into distance bins. 

    Parameters
    ----------
    landscape : FitnessLandscape or nx.Graph
        The fitness landscape to analyze.
    
    bins : array-like
        The edge-weight bin to collect edges into. Shape must be (2,).
    
    Returns
    -------
    List[tuple[int, int]]
        The list of edges indexing the sequences.
    """

    if isinstance(landscape, FitnessLandscape):
        graph = landscape.graph
    elif isinstance(landscape, nx.Graph):
        graph = landscape
    else:
        raise TypeError("landscape must be a FitnessLandscape or nx.Graph")
    
    # Expect weight to be a two-element sequence: [min_weight, max_weight)
    min_weight, max_weight = weight
    selected_edges = []
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        if w >= min_weight and w < max_weight:
            selected_edges.append((u, v))
    return selected_edges

def local_dirichlet_energy(landscape: FitnessLandscape) -> Dict: #TODO: add indexing sequences.
    """
    Function to determine the local Dirichlet energy of a node.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    Returns
    -------
    Dict
        Dict of local energies.
    """

    landscape = landscape.graph

    nodes = list(landscape.nodes())
    values = np.array([landscape.nodes[node]['fitness'] for node in nodes])

    # Build adjacency matrix using nx.adjacency_matrix
    adj_matrix = nx.adjacency_matrix(landscape, nodelist=nodes, weight=None)
    adj_matrix = adj_matrix + adj_matrix.T  # Ensure symmetry
    adj_matrix[adj_matrix > 1] = 1  # Remove multiple edges

    results = {}

    for idx, node in enumerate(nodes):

        # Use getrow to retrieve the row as a 2D sparse matrix
        neighbor_indices = adj_matrix.getrow(idx).nonzero()[1]
        sub_indices = np.append(neighbor_indices, idx)
        sub_values = values[sub_indices]

        sub_adj = adj_matrix[sub_indices][:, sub_indices]
        degree_vector = np.array(sub_adj.sum(axis=1)).flatten()
        degree_matrix = np.diag(degree_vector)
        laplacian = degree_matrix - sub_adj.toarray()
        local_dirichlet = sub_values @ laplacian @ sub_values

        results[(f'{node}', idx)] = local_dirichlet

def local_dirichlet_energy_contribution(landscape: FitnessLandscape) -> Dict[int, float]:
    """
    Calculates the local contribution of each node to the total
    Dirichlet energy.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze, which must have a graph
        representation.

    Returns
    -------
    Dict[int, float]
        A dictionary mapping each node index to its local Dirichlet
        energy contribution.
    """
    if not isinstance(landscape, FitnessLandscape) or landscape.graph is None:
        raise TypeError("Input must be a FitnessLandscape with an initialized graph.")

    graph = landscape.graph
    local_energies = {}

    for node, node_data in graph.nodes(data=True):
        fitness_i = node_data.get('fitness')
        if fitness_i is None:
            continue 

        degree_i = graph.degree(node)
        
        # Calculate the sum of fitness values of all neighbors
        sum_neighbor_fitness = 0
        for neighbor in graph.neighbors(node):
            neighbor_fitness = graph.nodes[neighbor].get('fitness')
            if neighbor_fitness is not None:
                sum_neighbor_fitness += neighbor_fitness
        
        local_energy = (degree_i * (fitness_i ** 2)) - (fitness_i * sum_neighbor_fitness)
        
        local_energies[node] = local_energy
        
    return local_energies