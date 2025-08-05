import numpy as np
import networkx as nx
from ..core.landscape import FitnessLandscape
from ..transforms.eigenmode import eigenmode_decomposition, _eigenmode_analysis_numpy, _eigenmode_analysis_torch
from typing import Union, Dict, Literal

def graph_properties(graph: Union[FitnessLandscape, nx.Graph]) -> Dict:
    """
    Calculate graph properties relevant to fitness landscapes.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to analyze.
        
    Returns
    -------
    dict
        Dictionary of graph properties.
    """
    
    properties = ['degree', 'clustering', 'path_length', 'components', 'density']
    
    results = {}
    
    for prop in properties:
        if prop == 'degree':
            # Calculate degree statistics
            degrees = [d for _, d in graph.degree()]
            results['degree'] = {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': np.min(degrees),
                'max': np.max(degrees)
            }
        
        elif prop == 'clustering':
            # Calculate clustering coefficient
            results['clustering'] = nx.average_clustering(graph)
        
        elif prop == 'path_length':
            # Calculate average shortest path length
            if nx.is_connected(graph):
                results['path_length'] = nx.average_shortest_path_length(graph)
            else:
                # Calculate for largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                results['path_length'] = nx.average_shortest_path_length(subgraph)
                results['path_length_note'] = 'Calculated for largest connected component'
        
        elif prop == 'components':
            # Calculate connected components
            components = list(nx.connected_components(graph))
            results['components'] = {
                'count': len(components),
                'largest_size': len(max(components, key=len)),
                'sizes': [len(c) for c in components]
            }
        
        elif prop == 'density':
            # Calculate graph density
            results['density'] = nx.density(graph)
        
        else:
            raise ValueError(f"Unsupported property: {prop}")
    
    return results

def calculate_ruggedness_local_optima(landscape: FitnessLandscape,
                                      **kwargs) -> Dict:
    """
    Function to measure ruggedness as the number of local fitness
    optima / maxima. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    Returns
    -------
    Dict
        The results dictionary.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    # Find local optima
    local_optima = []
    
    for i, seq in enumerate(sequences):
        # Get fitness of current sequence
        fitness = landscape.get_fitness(seq)
        
        # Get neighbors
        neighbors = list(landscape.graph.neighbors(i))
        
        # Check if fitness is higher than all neighbors
        is_local_optimum = True
        for neighbor in neighbors:
            neighbor_fitness = landscape.get_fitness(sequences[neighbor])
            if neighbor_fitness > fitness:
                is_local_optimum = False
                break
        
        if is_local_optimum:
            local_optima.append(i)
    
    # Calculate density of local optima
    density = len(local_optima) / len(sequences)
    
    # Calculate fitness statistics of local optima
    local_optima_fitness = [landscape.get_fitness(sequences[i]) for i in local_optima]
    
    if local_optima_fitness:
        mean_fitness = np.mean(local_optima_fitness)
        std_fitness = np.std(local_optima_fitness)
        max_fitness = np.max(local_optima_fitness)
        min_fitness = np.min(local_optima_fitness)
    else:
        mean_fitness = std_fitness = max_fitness = min_fitness = None
    
    return {
        'local_optima_count': len(local_optima),
        'local_optima_density': density,
        'local_optima_indices': local_optima,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'max_fitness': max_fitness,
        'min_fitness': min_fitness,
        'method': 'local_optima'
    }


def graph_spectral_analysis(graph: Union[nx.Graph, FitnessLandscape],
                            k: int = None,
                            matrix: Literal['adjacency', 'laplacian'] = 'laplacian',
                            backend: Literal['numpy', 'torch'] = 'numpy') -> Dict:
    """
    Analyze the eigenmodes of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to analyze.
    k : int or None, optional
        Number of eigenmodes to analyze.
    matrix : str, default = `laplacian`
        The matrix to decompose.
    backend : str, default = `numpy`
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    dict
        Eigenspectral analysis results. 
    """
    # Compute eigenmode decomposition
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix=matrix, k=k, backend=backend)
    
    # Handle FitnessLandscape input
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    
    # Compute analysis metrics based on backend
    if backend == 'numpy':
        return _eigenmode_analysis_numpy(eigenvalues, eigenvectors)
    elif backend == 'torch':
        return _eigenmode_analysis_torch(eigenvalues, eigenvectors)
    else:
        raise ValueError(f"Unsupported backend: {backend}")