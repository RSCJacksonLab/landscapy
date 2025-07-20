import numpy as np
import networkx as nx
from typing import Optional, Tuple
from ..core.landscape import FitnessLandscape
from itertools import product

def generate_NK_landscape(N: int,
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

class NKFitnessLandscape(FitnessLandscape):
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
        Graph type for creating the network representation ('hamming'
        or 'knn').
    """

    def __init__(self, N: int,
                 K: int,
                 alphabet_size: int,
                 seed: Optional[int] = None,
                 graph_type: str = 'hamming',
                 **kwargs):

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
        
