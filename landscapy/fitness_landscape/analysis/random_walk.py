import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.graph import create_hamming_graph
from ..core.sequence import sequence_distance
from logging import Logger
from .eigenmode import eigenmode_decomposition

def calculate_ruggedness_autocorrelation_analytical(landscape: FitnessLandscape,
                                                    lag_max: int = 10) -> Dict:
    """
    Calculate landscape ruggedness as the lagged autocorrelation of a 
    random walk using the eigenvalues of the Markov transition matrix.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    lag_max : int, default=`10`
        The lag size.
    
    Returns
    -------
    Dict
        The results dict of the analyitcal random walk.
    """
    eigenvalues, _ = eigenmode_decomposition(graph=landscape,
                                             matrix='transition',
                                             backend='numpy')
    eigenvalues = np.real(eigenvalues)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    autocorr = np.zeros(lag_max)
    for t in range(lag_max):
        autocorr[t] = np.sum(eigenvalues[1:]**t)
    
    return {
        'autocorrelation': autocorr,
        'eigenvalues': eigenvalues,
        'lag_max': lag_max
        }
    

def calculate_ruggedness_autocorrelation_stochastic(landscape: FitnessLandscape,
                                                    steps: int = 1000,
                                                    lag_max: int = 10,
                                                    **kwargs) -> Dict:
    """
    Calculate ruggedness using random walk autocorrelation with
    reference to the underlying stochastic process. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The Fitness landscape to analyze. 
    
    steps : int
        The number of steps to allow the random walk to proceed.
    
    lag_max : int
        The lag size.
    
    Returns
    -------
    Dict
        The result dict of the stochastic random walk.
    """
    # Extract sequences and fitness values
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    if landscape.graph_type is not 'hamming':
        Logger.warning(msg="Landscape graph type is not `Hamming`. Path analysis relies on Hamming structure for valid interpretation.")
    
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


def _random_walk(graph: nx.Graph,
                 steps: int) -> List:
    """
    Helper function for stochastic autocorrelation fo a random walk. 

    Parameters
    ----------
    graph : nx.graph
        The fitness landscape network graph. 
    
    steps : int
        The number of steps to run the random walk for.

    Returns
    -------
    walk : list
        The sites visited on the random walk. 
    """
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


def _autocorrelation(values: List,
                     lag_max: int) -> np.ndarray:
    """
    Helper function to compute the autocorrelation between values in an
    input list. 

    Parameter
    ---------
    values : List
        List of fitness values visited on random walk. 
    
    lag_max : int
        The maximum lag to determine the autocorrelation over.
    
    Returns
    -------
    autocorrelation : np.ndarray
        The autocorrelation values of the random walk. 
    """
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


def _correlation_length(autocorr: np.ndarray) -> int:
    """
    Helper function to compute the lebgth of a stochastic walk from the
    autocorrelation array. 

    Parameters
    ----------
    autocorr : np.ndarray
        The autocorrelation values. 
    
    Returns
    -------
    int
        The length of the correlation. 
    """
    # Find first lag where autocorrelation drops below 1/e
    threshold = 1.0 / np.e
    
    for lag, corr in enumerate(autocorr):
        if corr < threshold:
            return lag
    
    # If autocorrelation never drops below threshold, return max lag
    return len(autocorr) - 1