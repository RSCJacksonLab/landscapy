"""
Ruggedness analysis for fitness landscapes.

This module provides functions for analyzing the ruggedness of fitness landscapes,
which is a measure of how difficult it is to navigate the landscape.
"""

import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.graph import create_hamming_graph


def calculate_ruggedness(landscape, method='autocorrelation', **kwargs):
    """
    Calculate ruggedness of a fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    method : str, optional
        Method for calculating ruggedness:
        - 'autocorrelation': Use random walk autocorrelation
        - 'fdc': Use fitness-distance correlation
        - 'local_optima': Use density of local optima
        - 'roughness': Use roughness metric
    **kwargs
        Additional parameters for the method.
        
    Returns
    -------
    dict
        Ruggedness metrics.
    """
    if method == 'autocorrelation':
        return _calculate_ruggedness_autocorrelation(landscape, **kwargs)
    elif method == 'fdc':
        return _calculate_ruggedness_fdc(landscape, **kwargs)
    elif method == 'local_optima':
        return _calculate_ruggedness_local_optima(landscape, **kwargs)
    elif method == 'roughness':
        return _calculate_ruggedness_roughness(landscape, **kwargs)
    else:
        raise ValueError(f"Unsupported ruggedness calculation method: {method}")


def _calculate_ruggedness_autocorrelation(landscape, steps=1000, lag_max=10, **kwargs):
    """Calculate ruggedness using random walk autocorrelation."""
    # Extract sequences and fitness values
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Perform random walk
    walk_indices = _random_walk(landscape.graph, steps)
    
    # Get fitness values along walk
    fitness_values = np.array([landscape.get_fitness(sequences[i]) for i in walk_indices])
    
    # Calculate autocorrelation
    autocorr = _autocorrelation(fitness_values, lag_max)
    
    # Calculate correlation length
    correlation_length = _correlation_length(autocorr)
    
    return {
        'autocorrelation': autocorr,
        'correlation_length': correlation_length,
        'method': 'autocorrelation',
        'steps': steps,
        'lag_max': lag_max
    }


def _calculate_ruggedness_fdc(landscape, target_sequence=None, **kwargs):
    """Calculate ruggedness using fitness-distance correlation."""
    # Extract sequences and fitness values
    sequences = landscape.sequences
    fitness_values = np.array([landscape.get_fitness(seq) for seq in sequences])
    
    # Determine target sequence (global optimum by default)
    if target_sequence is None:
        # Find sequence with maximum fitness
        max_idx = np.argmax(fitness_values)
        target_sequence = sequences[max_idx]
    
    # Calculate distances to target
    from ..core.sequence import sequence_distance
    distances = np.array([sequence_distance(seq, target_sequence) for seq in sequences])
    
    # Calculate correlation
    correlation = np.corrcoef(distances, fitness_values)[0, 1]
    
    # Calculate additional statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    max_dist = np.max(distances)
    
    return {
        'fdc': correlation,
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'max_distance': max_dist,
        'method': 'fdc'
    }


def _calculate_ruggedness_local_optima(landscape, **kwargs):
    """Calculate ruggedness using density of local optima."""
    # Extract sequences
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
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


def _calculate_ruggedness_roughness(landscape, **kwargs):
    """Calculate ruggedness using roughness metric."""
    # Extract sequences
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Calculate roughness as average absolute fitness difference between neighbors
    total_diff = 0.0
    edge_count = 0
    
    for i, j in landscape.graph.edges():
        # Get fitness values
        fitness_i = landscape.get_fitness(sequences[i])
        fitness_j = landscape.get_fitness(sequences[j])
        
        # Add absolute difference
        total_diff += abs(fitness_i - fitness_j)
        edge_count += 1
    
    # Calculate average roughness
    if edge_count > 0:
        roughness = total_diff / edge_count
    else:
        roughness = 0.0
    
    # Calculate additional statistics
    fitness_values = np.array([landscape.get_fitness(seq) for seq in sequences])
    fitness_range = np.max(fitness_values) - np.min(fitness_values)
    normalized_roughness = roughness / fitness_range if fitness_range > 0 else 0.0
    
    return {
        'roughness': roughness,
        'normalized_roughness': normalized_roughness,
        'fitness_range': fitness_range,
        'edge_count': edge_count,
        'method': 'roughness'
    }


def _random_walk(graph, steps):
    """Perform random walk on graph."""
    # Start at random node
    current = np.random.choice(list(graph.nodes()))
    
    # Initialize walk
    walk = [current]
    
    # Perform walk
    for _ in range(steps - 1):
        # Get neighbors
        neighbors = list(graph.neighbors(current))
        
        if not neighbors:
            # No neighbors, stay at current node
            walk.append(current)
        else:
            # Move to random neighbor
            current = np.random.choice(neighbors)
            walk.append(current)
    
    return walk


def _autocorrelation(values, lag_max):
    """Calculate autocorrelation of values up to lag_max."""
    # Normalize values
    values = np.array(values)
    values = values - np.mean(values)
    values = values / np.std(values)
    
    # Calculate autocorrelation
    n = len(values)
    autocorr = np.zeros(lag_max + 1)
    
    for lag in range(lag_max + 1):
        # Calculate autocorrelation at lag
        autocorr[lag] = np.sum(values[:(n-lag)] * values[lag:]) / (n - lag)
    
    return autocorr


def _correlation_length(autocorr):
    """Calculate correlation length from autocorrelation."""
    # Find first lag where autocorrelation drops below 1/e
    threshold = 1.0 / np.e
    
    for lag, corr in enumerate(autocorr):
        if corr < threshold:
            return lag
    
    # If autocorrelation never drops below threshold, return max lag
    return len(autocorr) - 1


def adaptive_walk(landscape, start_sequence=None, max_steps=100, strategy='greedy'):
    """
    Perform adaptive walk on fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to walk on.
    start_sequence : Sequence or None, optional
        Starting sequence. If None, a random sequence is chosen.
    max_steps : int, optional
        Maximum number of steps to take.
    strategy : str, optional
        Walk strategy:
        - 'greedy': Always move to the neighbor with highest fitness
        - 'random_improvement': Move to a random neighbor with higher fitness
        
    Returns
    -------
    dict
        Walk results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Determine start sequence
    if start_sequence is None:
        # Choose random sequence
        start_idx = np.random.choice(len(sequences))
        start_sequence = sequences[start_idx]
    else:
        # Find index of start sequence
        for i, seq in enumerate(sequences):
            if seq == start_sequence:
                start_idx = i
                break
        else:
            raise ValueError("Start sequence not found in landscape")
    
    # Initialize walk
    current_idx = start_idx
    current_fitness = landscape.get_fitness(sequences[current_idx])
    
    walk_indices = [current_idx]
    walk_fitness = [current_fitness]
    
    # Perform walk
    for step in range(max_steps):
        # Get neighbors
        neighbors = list(landscape.graph.neighbors(current_idx))
        
        # Get fitness of neighbors
        neighbor_fitness = [landscape.get_fitness(sequences[i]) for i in neighbors]
        
        # Find neighbors with higher fitness
        better_indices = [i for i, fitness in enumerate(neighbor_fitness) 
                         if fitness > current_fitness]
        
        if not better_indices:
            # No better neighbors, end walk
            break
        
        # Choose next step based on strategy
        if strategy == 'greedy':
            # Choose neighbor with highest fitness
            best_idx = np.argmax(neighbor_fitness)
            next_idx = neighbors[best_idx]
        elif strategy == 'random_improvement':
            # Choose random neighbor with higher fitness
            better_neighbors = [neighbors[i] for i in better_indices]
            next_idx = np.random.choice(better_neighbors)
        else:
            raise ValueError(f"Unsupported walk strategy: {strategy}")
        
        # Update current position
        current_idx = next_idx
        current_fitness = landscape.get_fitness(sequences[current_idx])
        
        # Update walk
        walk_indices.append(current_idx)
        walk_fitness.append(current_fitness)
    
    # Calculate walk statistics
    steps_taken = len(walk_indices) - 1
    fitness_gain = walk_fitness[-1] - walk_fitness[0]
    
    return {
        'walk_indices': walk_indices,
        'walk_fitness': walk_fitness,
        'steps_taken': steps_taken,
        'fitness_gain': fitness_gain,
        'start_fitness': walk_fitness[0],
        'end_fitness': walk_fitness[-1],
        'reached_optimum': steps_taken < max_steps,
        'strategy': strategy
    }


def neutral_network_analysis(landscape, threshold=0.0):
    """
    Analyze neutral networks in the fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    threshold : float, optional
        Fitness difference threshold for considering two sequences neutral.
        
    Returns
    -------
    dict
        Neutral network analysis results.
    """
    # Extract sequences and fitness values
    sequences = landscape.sequences
    fitness_values = np.array([landscape.get_fitness(seq) for seq in sequences])
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, fitness_values)
    
    # Create neutral network graph
    neutral_graph = nx.Graph()
    
    # Add all nodes
    for i in range(len(sequences)):
        neutral_graph.add_node(i, fitness=fitness_values[i])
    
    # Add edges between neutral neighbors
    for i, j in landscape.graph.edges():
        # Check if fitness difference is within threshold
        if abs(fitness_values[i] - fitness_values[j]) <= threshold:
            neutral_graph.add_edge(i, j)
    
    # Find connected components (neutral networks)
    components = list(nx.connected_components(neutral_graph))
    
    # Calculate statistics for each neutral network
    networks = []
    
    for i, component in enumerate(components):
        # Convert to list for indexing
        component = list(component)
        
        # Calculate statistics
        network_fitness = [fitness_values[j] for j in component]
        mean_fitness = np.mean(network_fitness)
        std_fitness = np.std(network_fitness)
        size = len(component)
        
        # Calculate network diameter
        subgraph = neutral_graph.subgraph(component)
        try:
            diameter = nx.diameter(subgraph)
        except nx.NetworkXError:
            # Not connected or empty graph
            diameter = 0
        
        networks.append({
            'id': i,
            'size': size,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness,
            'diameter': diameter,
            'nodes': component
        })
    
    # Sort networks by size (largest first)
    networks.sort(key=lambda x: x['size'], reverse=True)
    
    # Calculate overall statistics
    total_nodes = sum(network['size'] for network in networks)
    
    return {
        'networks': networks,
        'network_count': len(networks),
        'largest_network_size': networks[0]['size'] if networks else 0,
        'largest_network_fraction': networks[0]['size'] / total_nodes if networks else 0,
        'threshold': threshold
    }


def landscape_correlation(landscape1, landscape2):
    """
    Calculate correlation between two fitness landscapes.
    
    Parameters
    ----------
    landscape1 : FitnessLandscape
        First fitness landscape.
    landscape2 : FitnessLandscape
        Second fitness landscape.
        
    Returns
    -------
    dict
        Correlation metrics.
    """
    # Check if landscapes have the same sequences
    if len(landscape1.sequences) != len(landscape2.sequences):
        raise ValueError("Landscapes must have the same number of sequences")
    
    # Extract fitness values
    fitness1 = np.array([landscape1.get_fitness(seq) for seq in landscape1.sequences])
    fitness2 = np.array([landscape2.get_fitness(seq) for seq in landscape2.sequences])
    
    # Calculate Pearson correlation
    pearson_corr = np.corrcoef(fitness1, fitness2)[0, 1]
    
    # Calculate Spearman rank correlation
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(fitness1, fitness2)
    
    # Calculate rank consistency
    # (fraction of pairs where relative fitness ordering is the same)
    n = len(fitness1)
    consistent_pairs = 0
    total_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if (fitness1[i] > fitness1[j] and fitness2[i] > fitness2[j]) or \
               (fitness1[i] < fitness1[j] and fitness2[i] < fitness2[j]) or \
               (fitness1[i] == fitness1[j] and fitness2[i] == fitness2[j]):
                consistent_pairs += 1
            total_pairs += 1
    
    rank_consistency = consistent_pairs / total_pairs if total_pairs > 0 else 0.0
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'rank_consistency': rank_consistency,
        'sequence_count': n
    }
