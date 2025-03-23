"""
Graph representations and operations for fitness landscapes.

This module provides functions for creating and manipulating graph representations
of fitness landscapes using NetworkX.
"""

import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from .sequence import Sequence, sequence_distance


def create_hamming_graph(sequences, fitness_values=None, weight_by_fitness=False, **kwargs):
    """
    Create a Hamming graph from sequences and fitness values.
    
    In a Hamming graph, nodes represent sequences and edges connect sequences
    that differ by exactly one position (Hamming distance = 1).
    
    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    fitness_values : array-like or None, optional
        Fitness values corresponding to sequences.
    weight_by_fitness : bool, optional
        Whether to weight edges by fitness differences.
    **kwargs
        Additional parameters for graph creation.
        
    Returns
    -------
    networkx.Graph
        Hamming graph.
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes with sequence and fitness attributes
    for i, seq in enumerate(sequences):
        if not isinstance(seq, Sequence):
            seq = Sequence(seq)
        
        # Add node with sequence attribute
        G.add_node(i, sequence=seq.to_array())
        
        # Add fitness attribute if provided
        if fitness_values is not None:
            G.nodes[i]['fitness'] = float(fitness_values[i])
    
    # Add edges between sequences with Hamming distance = 1
    for i in range(len(sequences)):
        seq_i = sequences[i]
        for j in range(i + 1, len(sequences)):
            seq_j = sequences[j]
            
            # Calculate Hamming distance
            dist = sequence_distance(seq_i, seq_j, metric='hamming')
            
            if dist == 1:
                # Add edge with weight
                if weight_by_fitness and fitness_values is not None:
                    weight = abs(float(fitness_values[i]) - float(fitness_values[j]))
                    G.add_edge(i, j, weight=weight, distance=dist)
                else:
                    G.add_edge(i, j, weight=1.0, distance=dist)
    
    return G


def create_knn_graph(sequences, fitness_values=None, k=5, metric='hamming', 
                    weight_by_distance=True, **kwargs):
    """
    Create a k-nearest neighbor graph.
    
    In a KNN graph, nodes represent sequences and edges connect each sequence
    to its k nearest neighbors according to the specified distance metric.
    
    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    fitness_values : array-like or None, optional
        Fitness values corresponding to sequences.
    k : int, optional
        Number of neighbors.
    metric : str, optional
        Distance metric ('hamming', 'euclidean', etc.)
    weight_by_distance : bool, optional
        Whether to weight edges by distance.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    networkx.Graph
        KNN graph.
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes with sequence and fitness attributes
    for i, seq in enumerate(sequences):
        if not isinstance(seq, Sequence):
            seq = Sequence(seq)
        
        # Add node with sequence attribute
        G.add_node(i, sequence=seq.to_array())
        
        # Add fitness attribute if provided
        if fitness_values is not None:
            G.nodes[i]['fitness'] = float(fitness_values[i])
    
    # Calculate all pairwise distances
    n_sequences = len(sequences)
    distances = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):
            dist = sequence_distance(sequences[i], sequences[j], metric=metric)
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Connect each sequence to its k nearest neighbors
    for i in range(n_sequences):
        # Get indices of k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances[i])
        nearest_indices = nearest_indices[1:k+1]  # Skip self (index 0)
        
        for j in nearest_indices:
            # Add edge with weight
            if weight_by_distance:
                weight = distances[i, j]
            else:
                weight = 1.0
            
            G.add_edge(i, j, weight=weight, distance=distances[i, j])
    
    return G


def graph_properties(graph, properties=None):
    """
    Calculate graph properties relevant to fitness landscapes.
    
    Parameters
    ----------
    graph : networkx.Graph
        Graph to analyze.
    properties : list or None, optional
        Properties to calculate. If None, calculate all properties.
        
    Returns
    -------
    dict
        Dictionary of graph properties.
    """
    if properties is None:
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
