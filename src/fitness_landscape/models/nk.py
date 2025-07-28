import numpy as np
import networkx as nx
from typing import Optional, List
from ..core.landscape import FitnessLandscape
from ..core.fitness import NumericFitness
from ..core.sequence import BaseNumpySequence, BinarySequence, MultialleleSequence
from itertools import product

def generate_NK_states(N: int,
                       K: int,
                       alphabet: List = [0,1],
                       seed: int = None,
                       adj_mat: Optional[np.ndarray] = None,
                       base_sequence: Optional[List] = None,
                       variable_sites: Optional[List[int]] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences and fitness values for a generalized NK landscape.

    Parameters
    ----------
    N : int
        Number of sites in each sequence.
    K : int
        Number of interacting neighbors for each site.
    alphabet : list
        The alphabet of characters or symbols to use for the sequences.
    seed : int, optional
        Random seed for reproducibility.
    adj_mat : np.ndarray, optional
        Adjacency matrix defining epistatic interactions. If None, a
        circular neighborhood is used.
    base_sequence : list, optional
        A template sequence. If provided, only the sites specified in
        `variable_sites` will be varied.
    variable_sites : list of int, optional
        Indices of the sites to be varied in the `base_sequence`.

    Returns
    -------
    sequences : np.ndarray
        Array of generated sequences.
    fitness_values : np.ndarray
        Array of corresponding fitness values.
    """
    if seed is not None:
        np.random.seed(seed)

    alphabet_size = len(alphabet)
    allele_map = {allele: i for i, allele in enumerate(alphabet)}

    if base_sequence is not None and variable_sites is not None:
        num_variable_sites = len(variable_sites)
        variant_combinations = list(product(alphabet, repeat=num_variable_sites))
        
        sequences = []
        for combo in variant_combinations:
            new_sequence = list(base_sequence)
            for i, site_idx in enumerate(variable_sites):
                new_sequence[site_idx] = combo[i]
            sequences.append(new_sequence)
        sequences = np.array(sequences)
    else:
        sequences = np.array(list(product(alphabet, repeat=N)))

    num_sequences = len(sequences)
    fitness_values = np.zeros(num_sequences)

    fitness_contrib = []
    for _ in range(N):
        table_size = alphabet_size ** (K + 1)
        fitness_contrib.append(np.random.rand(table_size))

    for i, seq in enumerate(sequences):
        total_fit = 0.0
        for j in range(N):
            if adj_mat is not None:
                neighbors = np.where(adj_mat[j] == 1)[0]
                if len(neighbors) > K:
                    neighbors = np.random.choice(neighbors, K, replace=False)
                indices = np.sort(np.append(neighbors, j))
            else:
                indices = [(j + offset) % N for offset in range(K + 1)]
            
            config = seq[indices]
            
            index = 0
            for allele_idx, allele in enumerate(config):
                numeric_allele = allele_map[allele]
                index += numeric_allele * (alphabet_size ** (len(config) - 1 - allele_idx))
            
            total_fit += fitness_contrib[j][index]

        fitness_values[i] = total_fit / N

    return sequences, fitness_values

def create_gnk_landscape(N: int,
                         K: int,
                         alphabet: List = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                         seed: Optional[int] = None,
                         adj_mat: Optional[np.ndarray] = None,
                         base_sequence: Optional[List] = None,
                         variable_sites: Optional[List[int]] = None,
                         **kwargs) -> FitnessLandscape:
    """
    Factory function to create a generalized NK fitness landscape.

    Parameters
    ----------
    N : int
        Number of sites in each sequence.
    K : int
        Number of interacting neighbors for each site.
    alphabet : list
        The alphabet of characters or symbols to use for the sequences.
    seed : int, optional
        Random seed for reproducibility.
    adj_mat : np.ndarray, optional
        Adjacency matrix defining epistatic interactions.
    base_sequence : list, optional
        A template sequence.
    variable_sites : list of int, optional
        Indices of the sites to be varied.
    **kwargs : dict, optional
        Additional keyword arguments for the FitnessLandscape constructor.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class.
    """
    alphabet_size = len(alphabet)
    sequences_np, fitness_values = generate_NK_states(
        N, K, alphabet, seed, adj_mat, base_sequence, variable_sites
    )
    
    if alphabet_size == 2:
        sequences = [BinarySequence(seq) for seq in sequences_np]
    else:
        sequences = [MultialleleSequence(seq, alphabet=alphabet) for seq in sequences_np]

    replicates = [[val] for val in fitness_values]
    
    fitness_layers = {
        f'nk_k={K}': NumericFitness(
            name=f'nk_k={K}',
            values=replicates,
            metadata={'N': N, 'K': K, 'alphabet_size': alphabet_size}
        )
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


