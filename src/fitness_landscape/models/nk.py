import numpy as np
import networkx as nx
from typing import Optional, Tuple
from ..core.landscape import FitnessLandscape

def generate_NK_landscape(N: int,
                          K: int,
                          alphabet_size: int = 2,
                          seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate all possible sequences and fitness values for an NK landscape.

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
    
    num_sequences = alphabet_size ** N
    sequences = []
    fitness_values = []
    
    fitness_contrib = []
    for i in range(N):
        table_size = alphabet_size ** (K + 1)
        fitness_contrib.append(np.random.rand(table_size))
    
    # Iterate over all possible sequences
    for i in range(num_sequences):
        # Generate a sequence as a numpy array; for binary sequences, use binary representation.
        seq = np.array(list(np.binary_repr(i, width=N)), dtype=int)
        sequences.append(seq)
        
        total_fit = 0.0
        # Sum the contributions from each gene
        for j in range(N):
            # Define a circular neighborhood: gene j and the next K genes (modulo N)
            indices = [(j + offset) % N for offset in range(K + 1)]
            config = seq[indices]

            index = int("".join(config.astype(str)), base=alphabet_size)
            total_fit += fitness_contrib[j][index]
        
        # Average the contributions to obtain the overall fitness.
        fitness_values.append(total_fit / N)
    
    return np.array(sequences), np.array(fitness_values)


class NKFitnessLandscape(FitnessLandscape):

    def __init__(self, N: int,
                 K: int,
                 alphabet_size: int,
                 seed: Optional[int] = None,
                 graph_type: str = 'hamming',
                 **kwargs):
        """
        NK Landscape FitnessLanscape subclass.

        Attributes
        ----------
        N : int
            Number of genes in each sequence.
        K : int
            Number of interactions per gene.
        alleles : int, optional
            Number of states per gene (default is 2 for binary sequences).
        seed : int, optional
            Random seed for reproducibility.
        graph_type : str, default=`Hamming`
            Graph type for creating the network representation ('hamming' or 'knn').
        """
        sequences, fitness_values = generate_NK_landscape(N,
                                                          K,
                                                          alphabet_size=alphabet_size,
                                                          seed=seed)
        self.K = K
        self.N = N
        self.alphabet_size = alphabet_size
        self.seed = seed

        super().__init__(sequences=sequences,
                         fitness_values=fitness_values,
                         graph_type=graph_type,
                         **kwargs)
        
