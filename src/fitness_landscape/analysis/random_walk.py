import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.graph import create_hamming_graph
from ..core.sequence import sequence_distance
from .eigenmode import eigenmode_decomposition

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
    eigenvalues, _ = eigenmode_decomposition(graph=landscape,
                                             matrix='adjacency')
    
    power_spectrum = np.abs(np.fft.fft(eigenvalues))**2
    autocorr = np.fft.ifft(power_spectrum).real
    autocorr /= autocorr[0]  # Normalize
    
    # Calculate correlation length
    correlation_length = -1 / np.log(np.abs(autocorr[1])) if autocorr[1] != 0 else np.inf
    
    if lag_max is not None:
        autocorr = autocorr[:lag_max + 1]
    
    return {
        'autocorrelation': autocorr,
        'correlation_length': correlation_length
    }

def calculate_ruggedness_autocorrelation_stochastic(landscape: FitnessLandscape,
                                                      steps: int = 1000,
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
    
    # Start at a random node
    current_node = rng.choice(list(landscape.graph.nodes()))
    
    fitness_trajectory = []
    for _ in range(steps):
        fitness_trajectory.append(landscape.get_fitness(landscape.sequences[current_node]))
        neighbors = list(landscape.graph.neighbors(current_node))
        current_node = rng.choice(neighbors)
        
    fitness_trajectory = np.array(fitness_trajectory)
    mean_fitness = np.mean(fitness_trajectory)
    
    if lag_max is None:
        lag_max = len(fitness_trajectory) // 2
        
    # Calculate autocorrelation using numpy's correlate function
    autocorr = np.correlate(fitness_trajectory - mean_fitness, fitness_trajectory - mean_fitness, mode='full')
    autocorr = autocorr[len(fitness_trajectory)-1 : len(fitness_trajectory) + lag_max]
    autocorr /= autocorr[0]
    
    correlation_length = -1 / np.log(np.abs(autocorr[1])) if len(autocorr) > 1 and autocorr[1] != 0 else np.inf
    
    return {
        'autocorrelation': autocorr,
        'correlation_length': correlation_length
    }