import numpy as np
from typing import Optional, List
from ..core.landscape import FitnessLandscape
from ..core.sequence import generate_sequences, BaseNumpySequence
from ..core.fitness import NumericFitness

def create_rmf_landscape(N: int,
                         slope: float,
                         sigma: float,
                         seed: Optional[int] = None,
                         optimum: Optional[List[int]] = None,
                         **kwargs) -> FitnessLandscape:
    """
    Factory function to create a Rough Mount Fuji (RMF) fitness
    landscape.

    Parameters
    ----------
    N : int
        Length of the sequences.
    slope : float
        Slope of the RMF landscape.
    sigma : float
        Standard deviation of the stochastic noise.
    optimum : list of int, optional
        The sequence representing the optimum. If None, defaults to
        a sequence of zeros.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class representing the RMF
        landscape.
    """
    sequences_np = generate_sequences(length=N, alphabet=[0, 1])

    sequences = [BaseNumpySequence(seq) for seq in sequences_np]
    
    if optimum is None:
        optimum_seq = BaseNumpySequence(np.zeros(N, dtype=int))
    else:
        optimum_seq = BaseNumpySequence(np.array(optimum, dtype=int))

    distances = np.array([optimum_seq.distance(seq) for seq in sequences])
    
    # Deterministic part
    fitness_values = -slope * distances
    
    # Stochastic part
    if sigma > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, size=len(sequences))
        fitness_values += noise
        
    # Wrap the fitness values for the NumericFitness layer
    replicates = [[val] for val in fitness_values]
    
    fitness_layers = {
        f'rmf_sigma={sigma}_slope={slope}': NumericFitness(name=f'rmf_sigma={sigma}_slope={slope}',
                                                           values=replicates,
                                                           metadata={
                                                               'slope' : slope,
                                                               'sigma' : sigma,
                                                               'optimum_seq' : optimum_seq
                                                               })
    }

    return FitnessLandscape.build(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph='hamming',
        **kwargs
    )
