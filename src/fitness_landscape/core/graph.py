import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from .sequence import Sequence, sequence_distance

# Simple graph constructors

def create_hamming_graph(sequences: List[Sequence],
                         fitness_values: Union[np.ndarray, List] = None,
                         weight_by_fitness: bool = False) -> nx.Graph:
    """
    Create a Hamming graph from sequences and fitness values. In a
    Hamming graph, nodes represent sequences and edges connect
    sequences that differ by exactly one position (Hamming
    distance = 1).
    
    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    fitness_values : array-like
        Fitness values corresponding to sequences.
    weight_by_fitness : bool, default = `False`
        Whether to weight edges by fitness differences.
        
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

def create_knn_graph(sequences: List[Sequence],
                     k: int,
                     fitness_values: Union[List, np.ndarray] = None,
                     metric: Literal['hamming'] = 'hamming', 
                     weight_by_distance: bool = True, **kwargs) -> nx.Graph:
    """
    Create a k-nearest neighbor graph. In a KNN graph, nodes represent
    sequences and edges connect each sequence to its k nearest
    neighbors according to the specified distance metric.
    
    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    k : int
        Number of neighbors.
    fitness_values : array-like or None
        Fitness values corresponding to sequences.
    metric : str, optional
        Distance metric ('hamming') // More to add
    weight_by_distance : bool, default=`True`
        Whether to weight edges by distance.
        
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