import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.sequence import Sequence
from ..core.graph import create_hamming_graph
from logging import Logger

def find_greedy_accessible_paths(landscape: FitnessLandscape, 
                                 start_sequence: Sequence,
                                 end_sequence: Sequence,
                                 **kwargs) -> Dict:
    """
    Function to find all fitness greedy paths between two sequences.
    Assumes a Hamming graph structure and does not weight paths by
    the evolutionary distance. 
        
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    start_sequence : Sequence
        Starting sequence.
    end_sequence : Sequence
        Ending sequence.        
    
    Returns
    -------
    Dict
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
    
    # Assert graph structure exists in landscape and warn if not Hamming graph.
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    if landscape.graph_type is not 'hamming':
        Logger.warning(msg="Landscape graph type is not `Hamming`. Path analysis relies on Hamming structure for valid interpretation.")

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

def analyze_path_accessibility(landscape: FitnessLandscape,
                               **kwargs) -> Dict:
    """
    Analyze accessibility of paths between local minima and maxima
    on a Hamming graph.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    
    Returns
    -------
    Dict
        Path accessibility analysis results.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    # Assert graph structure exists in landscape and warn if not Hamming graph.
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    if landscape.graph_type is not 'hamming':
        Logger.warning(msg="Landscape graph type is not `Hamming`. Path analysis relies on Hamming structure for valid interpretation.")
    
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

def calculate_basin_of_attraction(landscape: FitnessLandscape,
                                  local_optimum: Sequence,
                                  **kwargs) -> Dict:
    """
    Calculate the characteristics of a basin of attraction around a
    local optimum. Assumes a Hamming graph structure.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    local_optimum : Sequence
        Local optimum sequence.
        
    Returns
    -------
    Dict
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
    
    # Assert graph structure exists in landscape and warn if not Hamming graph.
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    if landscape.graph_type is not 'hamming':
        Logger.warning(msg="Landscape graph type is not `Hamming`. Path analysis relies on Hamming structure for valid interpretation.")
    
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

def adaptive_walk_stochastic(landscape: FitnessLandscape,
                             start_sequence: Sequence=None,
                             max_steps: int=100,
                             strategy: Literal['greedy', 'random_improvement']='greedy') -> Dict:
    """
    Perform adaptive walk on fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to walk on.
    start_sequence : Sequence, default=`None`
        Starting sequence. If None, a random sequence is chosen.
    max_steps : int, default=`100`
        Maximum number of steps to take.
    strategy : str, default=`greedy`
        Walk strategy. 'greedy': Always move to the neighbor with
        highest fitness. 'random_improvement': Move to a random
        neighbor with higher fitness.
        
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

def neutral_network_analysis(landscape: FitnessLandscape,
                             threshold: float = 0.0): 
    """
    Analyze neutral networks, where sequences can diverge and not
    improve in fitness, in the fitness landscape.
    
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