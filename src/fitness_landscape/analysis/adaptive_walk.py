import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence


def _sequence_index(
    landscape: FitnessLandscape,
    target: BaseNumpySequence,
    *,
    label: str = "Sequence",
) -> int:
    """Resolve a sequence argument, preferring object identity for duplicates."""
    for index, sequence in enumerate(landscape.sequences):
        if sequence is target:
            return index
    matches = [
        index for index, sequence in enumerate(landscape.sequences) if sequence == target
    ]
    if not matches:
        raise ValueError(f"{label} not found in landscape")
    if len(matches) > 1:
        raise ValueError(
            "Sequence value matches multiple landscape rows; pass the exact sequence object."
        )
    return matches[0]


def _node_fitness(landscape: FitnessLandscape) -> dict[Any, float]:
    """Return active scalar fitness keyed by graph-node label."""
    signal = landscape.get_signal()
    return {
        node: float(signal[landscape.sequence_index_for_node(node)])
        for node in landscape.graph.nodes()
    }


def _random_node(nodes: list[Any]) -> Any:
    """Choose from arbitrary hashable labels without NumPy coercing tuples."""
    return nodes[int(np.random.randint(len(nodes)))]

def find_greedy_accessible_paths(landscape: FitnessLandscape, 
                                 start_sequence: BaseNumpySequence,
                                 end_sequence: BaseNumpySequence,
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
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.
    
    Returns
    -------
    Dict
        Path analysis results.
    """
    sequences = landscape.sequences
    start_idx = _sequence_index(landscape, start_sequence, label="Start sequence")
    end_idx = _sequence_index(landscape, end_sequence, label="End sequence")
    
    # Assert graph structure exists in landscape and warn if not Hamming graph.
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    

    start_node = landscape.node_for_sequence_index(start_idx)
    end_node = landscape.node_for_sequence_index(end_idx)
    fitness_by_node = _node_fitness(landscape)

    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(
        (node, {"fitness": fitness}) for node, fitness in fitness_by_node.items()
    )
    for source, target in landscape.graph.edges():
        if fitness_by_node[target] > fitness_by_node[source]:
            directed_graph.add_edge(source, target)
        elif fitness_by_node[source] > fitness_by_node[target]:
            directed_graph.add_edge(target, source)
    
    # Find all simple paths from start to end
    try:
        all_paths = list(nx.all_simple_paths(directed_graph, start_node, end_node))
    except nx.NetworkXNoPath:
        all_paths = []
    
    # Convert path indices to sequences and fitness values
    paths = []
    
    for node_path in all_paths:
        path_indices = [landscape.sequence_index_for_node(node) for node in node_path]
        path_sequences = [sequences[index] for index in path_indices]
        paths.append({
            'nodes': node_path,
            'indices': path_indices,
            'sequences': path_sequences,
            'fitness': [fitness_by_node[node] for node in node_path],
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
        'start_node': start_node,
        'end_node': end_node,
        'start_fitness': fitness_by_node[start_node],
        'end_fitness': fitness_by_node[end_node],
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
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.
    
    Returns
    -------
    Dict
        Path accessibility analysis results.
    """
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    fitness_by_node = _node_fitness(landscape)

    local_minima = []
    local_maxima = []

    for node in landscape.graph.nodes():
        neighbors = list(landscape.graph.neighbors(node))
        fitness = fitness_by_node[node]
        if all(fitness_by_node[neighbor] >= fitness for neighbor in neighbors):
            local_minima.append(node)
        if all(fitness_by_node[neighbor] <= fitness for neighbor in neighbors):
            local_maxima.append(node)

    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(
        (node, {"fitness": fitness}) for node, fitness in fitness_by_node.items()
    )
    for source, target in landscape.graph.edges():
        if fitness_by_node[target] > fitness_by_node[source]:
            directed_graph.add_edge(source, target)
        elif fitness_by_node[source] > fitness_by_node[target]:
            directed_graph.add_edge(target, source)
    
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
        'local_minima_indices': [
            landscape.sequence_index_for_node(node) for node in local_minima
        ],
        'local_maxima_indices': [
            landscape.sequence_index_for_node(node) for node in local_maxima
        ],
        'minima_count': len(local_minima),
        'maxima_count': len(local_maxima),
        'paths_to_maxima': paths_to_maxima,
        'accessibility': accessibility,
        'accessible_pairs': accessible_pairs,
        'total_pairs': total_pairs
    }

def calculate_basin_of_attraction_greedy(landscape: FitnessLandscape,
                                  local_optimum: BaseNumpySequence,
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
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.
        
    Returns
    -------
    Dict
        Basin of attraction analysis results.
    """
    sequences = landscape.sequences
    optimum_idx = _sequence_index(landscape, local_optimum, label="Local optimum")
    
    # Assert graph structure exists in landscape and warn if not Hamming graph.
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    optimum_node = landscape.node_for_sequence_index(optimum_idx)
    fitness_by_node = _node_fitness(landscape)
    optimum_fitness = fitness_by_node[optimum_node]
    neighbors = list(landscape.graph.neighbors(optimum_node))
    
    for neighbor in neighbors:
        if fitness_by_node[neighbor] > optimum_fitness:
            raise ValueError("Specified sequence is not a local optimum")
    
    # Calculate basin of attraction
    basin = set()
    
    # For each sequence, check if adaptive walk leads to the optimum
    for node in landscape.graph.nodes():
        # Skip the optimum itself
        if node == optimum_node:
            basin.add(node)
            continue
        
        # Simulate adaptive walk
        current_node = node
        current_fitness = fitness_by_node[current_node]
        visited = {current_node}
        reached_optimum = False
        
        while True:
            # Get neighbors
            neighbors = list(landscape.graph.neighbors(current_node))
            if not neighbors:
                break
            
            # Get fitness of neighbors
            neighbor_fitness = [fitness_by_node[neighbor] for neighbor in neighbors]
            
            # Find best neighbor
            best_idx = np.argmax(neighbor_fitness)
            best_neighbor = neighbors[best_idx]
            best_fitness = neighbor_fitness[best_idx]
            
            # Check if we've reached a local optimum
            if best_fitness <= current_fitness:
                break
            
            # Check if we've reached the target optimum
            if best_neighbor == optimum_node:
                reached_optimum = True
                break
            
            # Update current position
            current_node = best_neighbor
            current_fitness = best_fitness
            
            # Check for cycles
            if current_node in visited:
                break
            visited.add(current_node)
        
        if reached_optimum:
            basin.add(node)
    
    # Calculate basin statistics
    basin_size = len(basin)
    basin_fraction = basin_size / len(sequences)
    
    return {
        'basin': list(basin),
        'basin_indices': [landscape.sequence_index_for_node(node) for node in basin],
        'basin_size': basin_size,
        'basin_fraction': basin_fraction,
        'optimum': local_optimum,
        'optimum_node': optimum_node,
        'optimum_index': optimum_idx,
        'optimum_fitness': optimum_fitness
    }

def calculate_basin_of_attraction_stochastic(landscape: FitnessLandscape,
                                             local_optimum: BaseNumpySequence,
                                             n_simulations: int = 100,
                                             max_steps: int = 1000,
                                             beta: float = 1.0,
                                             acceptance_threshold: float = 0.5,
                                             **kwargs) -> Dict:
    """
    Calculates the basin of attraction for a local optimum using a
    stochastic Metropolis-style adaptive walk.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    local_optimum : BaseNumpySequence
        The local optimum sequence defining the basin.
    n_simulations : int, optional
        Number of stochastic walks to simulate from each starting
        sequence.
    max_steps : int, optional
        Maximum number of steps for each walk to prevent infinite
        loops.
    beta : float, optional
        Inverse temperature parameter for the Metropolis criterion.
        Controls selection strength.
    acceptance_threshold : float, optional
        The minimum probability required for a sequence to be included
        in the basin.
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.

    Returns
    -------
    Dict
        A dictionary containing the basin analysis results.
    """
    sequences = landscape.sequences
    if not sequences or landscape.graph is None:
        raise ValueError("Landscape must contain sequences and an initialized graph.")

    optimum_idx = _sequence_index(landscape, local_optimum, label="Local optimum")
    optimum_node = landscape.node_for_sequence_index(optimum_idx)
    fitness_by_node = _node_fitness(landscape)
    optimum_fitness = fitness_by_node[optimum_node]
    basin_probabilities = {}
    basin_sequences = set()

    # For each sequence in the landscape, estimate its probability of reaching the optimum
    for start_node in landscape.graph.nodes():
        if start_node == optimum_node:
            basin_probabilities[start_node] = 1.0
            continue

        successful_walks = 0
        for _ in range(n_simulations):
            current_node = start_node
            reached = False
            
            for _step in range(max_steps):
                if current_node == optimum_node:
                    reached = True
                    break

                current_fitness = fitness_by_node[current_node]
                neighbors = list(landscape.graph.neighbors(current_node))
                
                if not neighbors:
                    # Trapped at a node with no neighbors
                    break 

                # Propose a random move to a neighbor
                proposed_neighbor = _random_node(neighbors)
                proposed_fitness = fitness_by_node[proposed_neighbor]
                
                # Calculate the acceptance probability
                delta_fitness = proposed_fitness - current_fitness
                if delta_fitness > 0:
                    acceptance_prob = 1.0
                else:
                    acceptance_prob = np.exp(beta * delta_fitness)
                
                # Accept or reject the move
                if np.random.rand() < acceptance_prob:
                    current_node = proposed_neighbor
            if current_node == optimum_node:
                reached = True
            if reached:
                successful_walks += 1

        # Calculate the probability of belonging to the basin
        basin_probabilities[start_node] = successful_walks / n_simulations

    # A sequence is in the basin if its probability exceeds the threshold
    for idx, prob in basin_probabilities.items():
        if prob >= acceptance_threshold:
            basin_sequences.add(idx)

    return {
        'basin': list(basin_sequences),
        'basin_indices': [
            landscape.sequence_index_for_node(node) for node in basin_sequences
        ],
        'basin_size': len(basin_sequences),
        'basin_fraction': len(basin_sequences) / len(sequences),
        'optimum': local_optimum,
        'optimum_node': optimum_node,
        'optimum_index': optimum_idx,
        'optimum_fitness': optimum_fitness,
        'basin_probabilities': basin_probabilities,
        'basin_probabilities_by_index': {
            landscape.sequence_index_for_node(node): probability
            for node, probability in basin_probabilities.items()
        },
        'parameters': {
            'n_simulations': n_simulations,
            'max_steps': max_steps,
            'beta': beta,
            'acceptance_threshold': acceptance_threshold
        }
    }


def adaptive_walk_stochastic(landscape: FitnessLandscape,
                             start_sequence: BaseNumpySequence=None,
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
    
    if landscape.graph is None:
        raise ValueError("Landscape graph must be initialized before an adaptive walk.")

    fitness_by_node = _node_fitness(landscape)
    
    # Determine start sequence
    if start_sequence is None:
        start_idx = np.random.choice(len(sequences))
        start_sequence = sequences[start_idx]
    else:
        start_idx = _sequence_index(landscape, start_sequence, label="Start sequence")

    start_node = landscape.node_for_sequence_index(int(start_idx))
    
    # Initialize walk
    current_node = start_node
    current_fitness = fitness_by_node[current_node]

    walk_nodes = [current_node]
    walk_fitness = [current_fitness]
    
    # Perform walk
    for step in range(max_steps):
        # Get neighbors
        neighbors = list(landscape.graph.neighbors(current_node))
        
        # Get fitness of neighbors
        neighbor_fitness = [fitness_by_node[node] for node in neighbors]
        
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
            next_node = neighbors[best_idx]
        elif strategy == 'random_improvement':
            # Choose random neighbor with higher fitness
            better_neighbors = [neighbors[i] for i in better_indices]
            next_node = _random_node(better_neighbors)
        else:
            raise ValueError(f"Unsupported walk strategy: {strategy}")
        
        # Update current position
        current_node = next_node
        current_fitness = fitness_by_node[current_node]
        
        # Update walk
        walk_nodes.append(current_node)
        walk_fitness.append(current_fitness)
    
    # Calculate walk statistics
    walk_indices = [landscape.sequence_index_for_node(node) for node in walk_nodes]
    steps_taken = len(walk_nodes) - 1
    fitness_gain = walk_fitness[-1] - walk_fitness[0]
    
    return {
        'walk_indices': walk_indices,
        'walk_nodes': walk_nodes,
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
    if landscape.graph is None:
        raise ValueError("Landscape graph must be initialized before neutral-network analysis.")
    fitness_by_node = _node_fitness(landscape)
    
    # Create neutral network graph
    neutral_graph = nx.Graph()
    
    for node in landscape.graph.nodes():
        neutral_graph.add_node(node, fitness=fitness_by_node[node])
    
    # Add edges between neutral neighbors
    for source, target in landscape.graph.edges():
        if abs(fitness_by_node[source] - fitness_by_node[target]) <= threshold:
            neutral_graph.add_edge(source, target)
    
    # Find connected components (neutral networks)
    components = list(nx.connected_components(neutral_graph))
    
    # Calculate statistics for each neutral network
    networks = []
    
    for i, component in enumerate(components):
        # Convert to list for indexing
        component = list(component)
        
        # Calculate statistics
        network_fitness = [fitness_by_node[node] for node in component]
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
            'nodes': component,
            'sequence_indices': [
                landscape.sequence_index_for_node(node) for node in component
            ],
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
