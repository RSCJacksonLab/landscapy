import numpy as np
import networkx as nx
from ..core.landscape import FitnessLandscape
from typing import Union, Dict

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