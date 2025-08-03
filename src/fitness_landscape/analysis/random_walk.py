import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.graph import create_hamming_graph
from ..core.sequence import sequence_distance
from ..transforms import graph_fourier_transform, eigenmode_decomposition, inverse_graph_fourier_transform

def calculate_ruggedness_autocorrelation_analytical(landscape: FitnessLandscape,
                                                      lag_max: int = None) -> Dict:
    """
    Function to determine the analytical autocorrelation of a fitness
    landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    lag_max : int, default=`None`
        The maximum lag to include in the autocorrelation calculation.
    
    Returns
    -------
    results : Dict
        The analytical autocorrelation results.
    """
 
    # Center signal on 0
    raw_signal = landscape.get_signal()
    mean_centered_signal = raw_signal - np.mean(raw_signal)

    # Perform computation in fourier basis
    eigenvectors, _, gft_coefficients = graph_fourier_transform(landscape, signal=mean_centered_signal)
    power_spectrum = np.abs(gft_coefficients)**2
    autocov_matrix = eigenvectors @ np.diag(power_spectrum) @ eigenvectors.T
    
    # Average over distances
    dist_matrix = nx.floyd_warshall_numpy(landscape.graph)
    max_dist = int(np.max(dist_matrix))
    autocorr_by_lag = [[] for _ in range(max_dist + 1)]
    num_nodes = landscape.graph.number_of_nodes()
    
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            k = int(dist_matrix[i, j])
            autocorr_by_lag[k].append(autocov_matrix[i, j])
            
    r_k = np.array([np.mean(vals) if vals else 0 for vals in autocorr_by_lag])
    
    # Normalize the final autocorrelation function by r(0).
    if r_k[0] != 0:
        r_k /= r_k[0]
    
    # Calculate the correlation length.
    if len(r_k) > 1 and 0 < np.abs(r_k[1]) < 1:
        correlation_length = -1 / np.log(np.abs(r_k[1]))
    else:
        correlation_length = np.inf

    if lag_max is not None:
        r_k = r_k[:lag_max + 1]
    
    return {
        'autocorrelation': r_k,
        'correlation_length': correlation_length
    }

def calculate_ruggedness_autocorrelation_stochastic(landscape: FitnessLandscape,
                                                      n_walks: int = 100,
                                                      steps: int = 100,
                                                      lag_max: int = None,
                                                      seed: Optional[int] = None) -> Dict:
    """
    Function to determine the stochastic autocorrelation of a fitness
    landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    steps : int, default=1000
        The number of random walk steps.
    
    lag_max : int, default=`None`
        The maximum lag to include in the autocorrelation calculation.
    
    seed : int, default=`None`
        The seed for the RNG.
    
    Returns
    -------
    results : Dict
        The stochastic autocorrelation results.
    """
    rng = np.random.default_rng(seed)
    
    if lag_max is None:
        lag_max = steps // 2
        
    all_autocorrs = []
    
    for _ in range(n_walks):
        # Start each walk at a random node
        current_node = rng.choice(list(landscape.graph.nodes()))
        
        fitness_trajectory = []
        for _ in range(steps):
            fitness_trajectory.append(landscape.get_fitness(landscape.sequences[current_node]))
            neighbors = list(landscape.graph.neighbors(current_node))
            if not neighbors:
                # Handle nodes with no neighbors 
                break
            current_node = rng.choice(neighbors)
        
        if len(fitness_trajectory) < 2:
            continue # Skip walks that are too short
            
        # De-mean the signal for this trajectory
        fitness_trajectory = np.array(fitness_trajectory)
        mean_fitness = np.mean(fitness_trajectory)
        
        # Calculate autocorrelation for this single walk
        autocorr = np.correlate(fitness_trajectory - mean_fitness, fitness_trajectory - mean_fitness, mode='full')
        
        # Normalize and slice
        autocorr = autocorr[len(fitness_trajectory)-1 : len(fitness_trajectory) -1 + lag_max]
        if autocorr[0] != 0:
            autocorr /= autocorr[0]
            all_autocorrs.append(autocorr)
            
    if not all_autocorrs:
        return {
            'autocorrelation': np.array([1.0] + [0.0] * (lag_max - 1)),
            'correlation_length': np.inf
        }

    # Average the autocorrelation functions from all the walks
    avg_autocorr = np.mean(all_autocorrs, axis=0)
    
    # Calculate correlation length from the averaged autocorrelation
    if len(avg_autocorr) > 1 and 0 < np.abs(avg_autocorr[1]) < 1:
        correlation_length = -1 / np.log(np.abs(avg_autocorr[1]))
    else:
        correlation_length = np.inf
    
    return {
        'autocorrelation': avg_autocorr,
        'correlation_length': correlation_length
    }