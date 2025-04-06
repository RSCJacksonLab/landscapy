import numpy as np
from typing import Tuple, Optional
from ..core.landscape import FitnessLandscape

def generate_RMF_landscape(N: int,
                           slope: float,
                           sigma: float,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an RMF (Rough Mount Fuji) fitness landscape for binary
    sequences.
    
    Parameters
    ----------
    N : int
        Length of the binary sequences.
    slope : float
        The slope parameter (alpha) that scales the smooth contribution.
    sigma : float
        The standard deviation of the Gaussian noise (ruggedness).
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    sequences : np.ndarray
        Array of all binary sequences of length N (each row is a sequence).
    fitness_values : np.ndarray
        Array of fitness values corresponding to each sequence.
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_sequences = 2 ** N
    sequences = []
    fitness_values = []
    
    # Define the optimum sequence.
    optimum = np.zeros(N, dtype=int)
    
    # Enumerate all binary sequences (for small N)
    for i in range(num_sequences):
        # Convert integer to binary string padded to length N
        seq_str = np.binary_repr(i, width=N)
        seq = np.array(list(seq_str), dtype=int)
        sequences.append(seq)
        
        # Compute Hamming distance from the optimum
        distance = np.sum(seq != optimum)
        
        smooth_component = slope * (N - distance)
        rugged_component = np.random.normal(loc=0, scale=sigma)
        fitness = smooth_component + rugged_component
        
        fitness_values.append(fitness)
    
    return np.array(sequences), np.array(fitness_values)

class NKFitnessLandscape(FitnessLandscape):

    """
    Rough-mount-fuji (RMF) Landscape FitnessLanscape subclass.

    Attributes
    ----------
    
    N : int
        Length of the binary sequences.
    slope : float
        The slope of the smooth component of the RMF landscape.
    sigma : float
        The variance used in the Gaussian random noise for the rugged
        component of the RMF landscape.
    seed : int, optional
        Random seed for reproducibility.
    graph_type : str, default=`Hamming`
        Graph type for creating the network representation ('hamming' or 'knn').
    """

    def __init__(self,
                 N: int,
                 slope : float,
                 sigma : float,
                 seed: Optional[int] = None,
                 graph_type: str = 'hamming',
                 **kwargs):
        
        sequences, fitness_values = generate_RMF_landscape(N=N,
                                                           slope=slope,
                                                           sigma=sigma,
                                                           seed=seed)
        self.N = N
        self.slope = slope
        self.sigma = sigma
        self.seed = seed

        super().__init__(sequences=sequences,
                         fitness_values=fitness_values,
                         graph_type=graph_type,
                         **kwargs)