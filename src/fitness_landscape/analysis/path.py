"""
Path analysis for fitness landscapes.

This module provides functions for analyzing evolutionary paths through fitness landscapes,
including accessible paths, shortest paths, and path statistics.
"""

import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.graph import create_hamming_graph


def find_accessible_paths(landscape, start_sequence, end_sequence, **kwargs):
    """
    Find all accessible paths between two sequences.
    
    An accessible path is one where fitness increases at each step.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    start_sequence : Sequence
        Starting sequence.
    end_sequence : Sequence
        Ending sequence.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Path analysis results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Find indices of start and end sequences
    start_idx = None
    end_idx = None
    
    for i, seq in enumerate(sequences):
        if seq == start_sequence:
            start_idx = i
        if seq == end_sequence:
            end_idx = i
    
    if start_idx is None:
        raise ValueError("Start sequence not found in landscape")
    if end_idx is None:
        raise ValueError("End sequence not found in landscape")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Create directed graph for accessible paths
    directed_graph = nx.DiGraph()
    
    # Add nodes with fitness values
    for i, seq in enumerate(sequences):
        directed_graph.add_node(i, fitness=landscape.get_fitness(seq))
    
    # Add directed edges for fitness increases
    for i, j in landscape.graph.edges():
        fitness_i = landscape.get_fitness(sequences[i])
        fitness_j = landscape.get_fitness(sequences[j])
        
        if fitness_j > fitness_i:
            directed_graph.add_edge(i, j)
        elif fitness_i > fitness_j:
            directed_graph.add_edge(j, i)
    
    # Find all simple paths from start to end
    try:
        all_paths = list(nx.all_simple_paths(directed_graph, start_idx, end_idx))
    except nx.NetworkXNoPath:
        all_paths = []
    
    # Convert path indices to sequences and fitness values
    paths = []
    
    for path in all_paths:
        path_sequences = [sequences[i] for i in path]
        path_fitness = [landscape.get_fitness(seq) for seq in path_sequences]
        
        paths.append({
            'indices': path,
            'sequences': path_sequences,
            'fitness': path_fitness
        })
    
    # Calculate path statistics
    if paths:
        path_lengths = [len(path['indices']) - 1 for path in paths]
        mean_length = np.mean(path_lengths)
        min_length = np.min(path_lengths)
        max_length = np.max(path_lengths)
    else:
        mean_length = min_length = max_length = None
    
    return {
        'paths': paths,
        'path_count': len(paths),
        'mean_path_length': mean_length,
        'min_path_length': min_length,
        'max_path_length': max_length,
        'start_sequence': start_sequence,
        'end_sequence': end_sequence,
        'start_fitness': landscape.get_fitness(start_sequence),
        'end_fitness': landscape.get_fitness(end_sequence)
    }


def find_shortest_paths(landscape, start_sequence, end_sequence, **kwargs):
    """
    Find shortest paths between two sequences.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    start_sequence : Sequence
        Starting sequence.
    end_sequence : Sequence
        Ending sequence.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Path analysis results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Find indices of start and end sequences
    start_idx = None
    end_idx = None
    
    for i, seq in enumerate(sequences):
        if seq == start_sequence:
            start_idx = i
        if seq == end_sequence:
            end_idx = i
    
    if start_idx is None:
        raise ValueError("Start sequence not found in landscape")
    if end_idx is None:
        raise ValueError("End sequence not found in landscape")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Find shortest paths
    try:
        shortest_paths = list(nx.all_shortest_paths(landscape.graph, start_idx, end_idx))
    except nx.NetworkXNoPath:
        shortest_paths = []
    
    # Convert path indices to sequences and fitness values
    paths = []
    
    for path in shortest_paths:
        path_sequences = [sequences[i] for i in path]
        path_fitness = [landscape.get_fitness(seq) for seq in path_sequences]
        
        # Check if path is accessible (fitness increases at each step)
        is_accessible = True
        for i in range(len(path_fitness) - 1):
            if path_fitness[i + 1] <= path_fitness[i]:
                is_accessible = False
                break
        
        paths.append({
            'indices': path,
            'sequences': path_sequences,
            'fitness': path_fitness,
            'is_accessible': is_accessible
        })
    
    # Calculate path statistics
    if paths:
        path_lengths = [len(path['indices']) - 1 for path in paths]
        accessible_paths = [path for path in paths if path['is_accessible']]
        accessible_count = len(accessible_paths)
    else:
        path_lengths = []
        accessible_count = 0
    
    return {
        'paths': paths,
        'path_count': len(paths),
        'path_length': path_lengths[0] if path_lengths else None,
        'accessible_count': accessible_count,
        'accessible_fraction': accessible_count / len(paths) if paths else 0.0,
        'start_sequence': start_sequence,
        'end_sequence': end_sequence,
        'start_fitness': landscape.get_fitness(start_sequence),
        'end_fitness': landscape.get_fitness(end_sequence)
    }


def analyze_path_accessibility(landscape, **kwargs):
    """
    Analyze accessibility of paths between local minima and maxima.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Path accessibility analysis results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Find local minima and maxima
    local_minima = []
    local_maxima = []
    
    for i, seq in enumerate(sequences):
        # Get fitness of current sequence
        fitness = landscape.get_fitness(seq)
        
        # Get neighbors
        neighbors = list(landscape.graph.neighbors(i))
        
        # Check if local minimum
        is_minimum = True
        for neighbor in neighbors:
            neighbor_fitness = landscape.get_fitness(sequences[neighbor])
            if neighbor_fitness < fitness:
                is_minimum = False
                break
        
        if is_minimum:
            local_minima.append(i)
        
        # Check if local maximum
        is_maximum = True
        for neighbor in neighbors:
            neighbor_fitness = landscape.get_fitness(sequences[neighbor])
            if neighbor_fitness > fitness:
                is_maximum = False
                break
        
        if is_maximum:
            local_maxima.append(i)
    
    # Create directed graph for accessible paths
    directed_graph = nx.DiGraph()
    
    # Add nodes with fitness values
    for i, seq in enumerate(sequences):
        directed_graph.add_node(i, fitness=landscape.get_fitness(seq))
    
    # Add directed edges for fitness increases
    for i, j in landscape.graph.edges():
        fitness_i = landscape.get_fitness(sequences[i])
        fitness_j = landscape.get_fitness(sequences[j])
        
        if fitness_j > fitness_i:
            directed_graph.add_edge(i, j)
        elif fitness_i > fitness_j:
            directed_graph.add_edge(j, i)
    
    # Analyze paths from each local minimum to each local maximum
    paths_to_maxima = {}
    
    for min_idx in local_minima:
        paths_to_maxima[min_idx] = {}
        
        for max_idx in local_maxima:
            # Skip if minimum and maximum are the same
            if min_idx == max_idx:
                continue
            
            # Find all simple paths from minimum to maximum
            try:
                all_paths = list(nx.all_simple_paths(directed_graph, min_idx, max_idx))
                paths_to_maxima[min_idx][max_idx] = len(all_paths)
            except nx.NetworkXNoPath:
                paths_to_maxima[min_idx][max_idx] = 0
    
    # Calculate accessibility statistics
    total_pairs = len(local_minima) * len(local_maxima) - len(set(local_minima) & set(local_maxima))
    accessible_pairs = sum(1 for min_idx in paths_to_maxima 
                          for max_idx in paths_to_maxima[min_idx] 
                          if paths_to_maxima[min_idx][max_idx] > 0)
    
    accessibility = accessible_pairs / total_pairs if total_pairs > 0 else 0.0
    
    return {
        'local_minima': local_minima,
        'local_maxima': local_maxima,
        'minima_count': len(local_minima),
        'maxima_count': len(local_maxima),
        'paths_to_maxima': paths_to_maxima,
        'accessibility': accessibility,
        'accessible_pairs': accessible_pairs,
        'total_pairs': total_pairs
    }


def calculate_path_metrics(landscape, sample_size=100, **kwargs):
    """
    Calculate metrics for random paths through the landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    sample_size : int, optional
        Number of random paths to sample.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Path metrics.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Sample random pairs of sequences
    n_sequences = len(sequences)
    start_indices = np.random.choice(n_sequences, sample_size)
    end_indices = np.random.choice(n_sequences, sample_size)
    
    # Calculate path metrics
    path_lengths = []
    accessible_paths = []
    accessible_fractions = []
    
    for i in range(sample_size):
        start_idx = start_indices[i]
        end_idx = end_indices[i]
        
        # Skip if start and end are the same
        if start_idx == end_idx:
            continue
        
        # Find shortest paths
        try:
            shortest_paths = list(nx.all_shortest_paths(landscape.graph, start_idx, end_idx))
        except nx.NetworkXNoPath:
            continue
        
        # Calculate path length
        path_length = len(shortest_paths[0]) - 1
        path_lengths.append(path_length)
        
        # Check accessibility of each path
        accessible_count = 0
        
        for path in shortest_paths:
            path_fitness = [landscape.get_fitness(sequences[i]) for i in path]
            
            # Check if path is accessible (fitness increases at each step)
            is_accessible = True
            for j in range(len(path_fitness) - 1):
                if path_fitness[j + 1] <= path_fitness[j]:
                    is_accessible = False
                    break
            
            if is_accessible:
                accessible_count += 1
        
        accessible_paths.append(accessible_count)
        accessible_fractions.append(accessible_count / len(shortest_paths))
    
    # Calculate statistics
    if path_lengths:
        mean_length = np.mean(path_lengths)
        std_length = np.std(path_lengths)
        mean_accessible = np.mean(accessible_fractions)
        std_accessible = np.std(accessible_fractions)
    else:
        mean_length = std_length = mean_accessible = std_accessible = None
    
    return {
        'mean_path_length': mean_length,
        'std_path_length': std_length,
        'mean_accessible_fraction': mean_accessible,
        'std_accessible_fraction': std_accessible,
        'sample_size': sample_size,
        'valid_samples': len(path_lengths)
    }


def find_evolutionary_trajectories(landscape, start_sequence, max_steps=100, n_trajectories=10, **kwargs):
    """
    Find evolutionary trajectories from a starting sequence.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    start_sequence : Sequence
        Starting sequence.
    max_steps : int, optional
        Maximum number of steps per trajectory.
    n_trajectories : int, optional
        Number of trajectories to simulate.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Trajectory analysis results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Find index of start sequence
    start_idx = None
    
    for i, seq in enumerate(sequences):
        if seq == start_sequence:
            start_idx = i
            break
    
    if start_idx is None:
        raise ValueError("Start sequence not found in landscape")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Simulate trajectories
    trajectories = []
    
    for _ in range(n_trajectories):
        # Initialize trajectory
        current_idx = start_idx
        current_fitness = landscape.get_fitness(sequences[current_idx])
        
        trajectory = {
            'indices': [current_idx],
            'fitness': [current_fitness]
        }
        
        # Simulate steps
        for step in range(max_steps):
            # Get neighbors
            neighbors = list(landscape.graph.neighbors(current_idx))
            
            # Get fitness of neighbors
            neighbor_fitness = [landscape.get_fitness(sequences[i]) for i in neighbors]
            
            # Find neighbors with higher fitness
            better_indices = [i for i, fitness in enumerate(neighbor_fitness) 
                             if fitness > current_fitness]
            
            if not better_indices:
                # No better neighbors, end trajectory
                break
            
            # Choose random neighbor with higher fitness
            better_neighbors = [neighbors[i] for i in better_indices]
            next_idx = np.random.choice(better_neighbors)
            
            # Update current position
            current_idx = next_idx
            current_fitness = landscape.get_fitness(sequences[current_idx])
            
            # Update trajectory
            trajectory['indices'].append(current_idx)
            trajectory['fitness'].append(current_fitness)
        
        # Add sequences to trajectory
        trajectory['sequences'] = [sequences[i] for i in trajectory['indices']]
        
        # Calculate trajectory statistics
        trajectory['length'] = len(trajectory['indices']) - 1
        trajectory['fitness_gain'] = trajectory['fitness'][-1] - trajectory['fitness'][0]
        trajectory['reached_optimum'] = trajectory['length'] < max_steps
        
        trajectories.append(trajectory)
    
    # Calculate overall statistics
    lengths = [traj['length'] for traj in trajectories]
    fitness_gains = [traj['fitness_gain'] for traj in trajectories]
    reached_optima = [traj['reached_optimum'] for traj in trajectories]
    
    return {
        'trajectories': trajectories,
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'mean_fitness_gain': np.mean(fitness_gains),
        'std_fitness_gain': np.std(fitness_gains),
        'optimum_fraction': np.mean(reached_optima),
        'start_sequence': start_sequence,
        'start_fitness': landscape.get_fitness(start_sequence),
        'n_trajectories': n_trajectories,
        'max_steps': max_steps
    }


def calculate_basin_of_attraction(landscape, local_optimum, **kwargs):
    """
    Calculate basin of attraction for a local optimum.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    local_optimum : Sequence
        Local optimum sequence.
    **kwargs
        Additional parameters.
        
    Returns
    -------
    dict
        Basin of attraction analysis results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Find index of local optimum
    optimum_idx = None
    
    for i, seq in enumerate(sequences):
        if seq == local_optimum:
            optimum_idx = i
            break
    
    if optimum_idx is None:
        raise ValueError("Local optimum not found in landscape")
    
    # Create Hamming graph if not already present
    if landscape.graph is None:
        landscape.graph = create_hamming_graph(sequences, 
                                              [landscape.get_fitness(seq) for seq in sequences])
    
    # Verify that the sequence is a local optimum
    optimum_fitness = landscape.get_fitness(local_optimum)
    neighbors = list(landscape.graph.neighbors(optimum_idx))
    
    for neighbor in neighbors:
        neighbor_fitness = landscape.get_fitness(sequences[neighbor])
        if neighbor_fitness > optimum_fitness:
            raise ValueError("Specified sequence is not a local optimum")
    
    # Calculate basin of attraction
    basin = set()
    
    # For each sequence, check if adaptive walk leads to the optimum
    for i, seq in enumerate(sequences):
        # Skip the optimum itself
        if i == optimum_idx:
            basin.add(i)
            continue
        
        # Simulate adaptive walk
        current_idx = i
        current_fitness = landscape.get_fitness(sequences[current_idx])
        
        visited = set([current_idx])
        reached_optimum = False
        
        while True:
            # Get neighbors
            neighbors = list(landscape.graph.neighbors(current_idx))
            
            # Get fitness of neighbors
            neighbor_fitness = [landscape.get_fitness(sequences[j]) for j in neighbors]
            
            # Find best neighbor
            best_idx = np.argmax(neighbor_fitness)
            best_neighbor = neighbors[best_idx]
            best_fitness = neighbor_fitness[best_idx]
            
            # Check if we've reached a local optimum
            if best_fitness <= current_fitness:
                break
            
            # Check if we've reached the target optimum
            if best_neighbor == optimum_idx:
                reached_optimum = True
                break
            
            # Update current position
            current_idx = best_neighbor
            current_fitness = best_fitness
            
            # Check for cycles
            if current_idx in visited:
                break
            
            visited.add(current_idx)
        
        if reached_optimum:
            basin.add(i)
    
    # Calculate basin statistics
    basin_size = len(basin)
    basin_fraction = basin_size / len(sequences)
    
    return {
        'basin': list(basin),
        'basin_size': basin_size,
        'basin_fraction': basin_fraction,
        'optimum': local_optimum,
        'optimum_fitness': optimum_fitness
    }
