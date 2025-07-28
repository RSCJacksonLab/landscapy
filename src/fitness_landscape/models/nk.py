import numpy as np
import networkx as nx
from typing import Optional, Tuple
from ..core.landscape import FitnessLandscape
from ..core.fitness import NumericFitness
from ..core.sequence import BaseNumpySequence, BinarySequence, MultialleleSequence
from itertools import product

def generate_NK_states(N: int,
                       K: int,
                       alphabet_size: int = 2,
                       seed: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all possible sequences and fitness values for an NK
    landscape.

    Parameters
    ----------
    N : int
        Number of sites in each sequence.
    K : int
        Number of interacting neighbors for each gene (epistatic
        interactions).
    alphabet_size : int, default=`2`
        Number of possible states per site (default is 2 for binary
        sequences).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    sequences : np.ndarray
        Array of sequences (each sequence is an array of integers).
    fitness_values : np.ndarray
        Array of fitness values corresponding to each sequence.

    """
    if seed is not None:
        np.random.seed(seed)

    # Generate all possible sequences
    alleles = range(alphabet_size)
    sequences = np.array(list(product(alleles, repeat=N)))
    num_sequences = len(sequences)
    fitness_values = np.zeros(num_sequences)

    # Create fitness contribution tables
    fitness_contrib = []
    for _ in range(N):
        table_size = alphabet_size ** (K + 1)
        fitness_contrib.append(np.random.rand(table_size))

    # Calculate fitness for each sequence
    for i, seq in enumerate(sequences):
        total_fit = 0.0
        for j in range(N):
            
            # Define a circular neighborhood
            indices = [(j + offset) % N for offset in range(K + 1)]
            config = seq[indices]

            # Calculate index for the fitness contribution table
            index = 0
            for allele_idx, allele in enumerate(config):
                index += allele * (alphabet_size ** (K - allele_idx))

            total_fit += fitness_contrib[j][index]

        fitness_values[i] = total_fit / N

    return sequences, fitness_values


#TODO: fix GNK for generic sequence type.
def create_gnk_landscape(N: int,
                        K: int,
                        alphabet_size: int = 2,
                        seed: Optional[int] = None,
                        **kwargs) -> FitnessLandscape:
    """
    Factory function to create an NK fitness landscape.

    Parameters
    ----------
    N : int
        Number of sites in each sequence.
    K : int
        Number of interacting neighbors for each gene (epistatic
        interactions).
    alphabet_size : int, default=`2`
        Number of possible states per site (default is 2 for binary
        sequences).
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the FitnessLandscape
        constructor.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class representing the NK
        landscape.
    """
    sequences_np, fitness_values = generate_NK_states(N, K, alphabet_size=alphabet_size, seed=seed)
    
    sequences = [BaseNumpySequence(seq) for seq in sequences_np]

    # Wrap the single fitness array into a list of lists for the NumericFitness layer
    replicates = [[val] for val in fitness_values]
    
    # Create the fitness layer
    fitness_layers = {
        f'nk_k={K}': NumericFitness(name=f'nk_k={K}',
                                    values=replicates,
                                    metadata={'N' : N,
                                              'K' : K,
                                              'alphabet_size' : alphabet_size})
    }
    
    return FitnessLandscape(
        sequences=sequences,
        fitness_layers=fitness_layers,
        **kwargs
    )


def create_nk_binary_landscape(N: int,
                               K: int,
                               seed: Optional[int] = None,
                               **kwargs) -> FitnessLandscape:
    """
    Factory function to create a binary NK fitness landscape.
    Sequence types are `BinarySequence`.

    Parameters
    ----------
    N : int
        Number of sites in each sequence.
    K : int
        Number of interacting neighbors for each gene (epistatic
        interactions).
    alphabet_size : int, default=`2`
        Number of possible states per site (default is 2 for binary
        sequences).
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the FitnessLandscape
        constructor.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class representing the NK
        landscape.
    """
    sequences_np, fitness_values = generate_NK_states(N, K, alphabet_size=2, seed=seed)
    
    sequences = [BinarySequence(seq) for seq in sequences_np]

    # Wrap the single fitness array into a list of lists for the NumericFitness layer
    replicates = [[val] for val in fitness_values]
    
    # Create the fitness layer
    fitness_layers = {
        f'nk_k={K}': NumericFitness(name=f'nk_k={K}',
                                    values=replicates,
                                    metadata={'N' : N,
                                              'K' : K,
                                              'alphabet_size' : 2,
                                              'type': 'binary'})
    }
    
    return FitnessLandscape(
        sequences=sequences,
        fitness_layers=fitness_layers,
        **kwargs
    )

def create_nk_multi_landscape(N: int,
                               K: int,
                               alphabet_size: int,
                               seed: Optional[int] = None,
                               **kwargs) -> FitnessLandscape:
    """
    Factory function to create a binary NK fitness landscape.
    Sequence types are `BinarySequence`.

    Parameters
    ----------
    N : int
        Number of sites in each sequence.
    K : int
        Number of interacting neighbors for each gene (epistatic
        interactions).
    alphabet_size : int, default=`2`
        Number of possible states per site (default is 2 for binary
        sequences).
    seed : int, optional
        Random seed for reproducibility.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the FitnessLandscape
        constructor.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class representing the NK
        landscape.
    """
    sequences_np, fitness_values = generate_NK_states(N, K, alphabet_size=alphabet_size, seed=seed)
    
    sequences = [MultialleleSequence(seq) for seq in sequences_np]

    # Wrap the single fitness array into a list of lists for the NumericFitness layer
    replicates = [[val] for val in fitness_values]
    
    # Create the fitness layer
    fitness_layers = {
        f'nk_k={K}': NumericFitness(name=f'nk_k={K}',
                                    values=replicates,
                                    metadata={'N' : N,
                                              'K' : K,
                                              'alphabet_size' : alphabet_size,
                                              'type' : 'multi-allele'})
    }
    
    return FitnessLandscape(
        sequences=sequences,
        fitness_layers=fitness_layers,
        **kwargs
    )


