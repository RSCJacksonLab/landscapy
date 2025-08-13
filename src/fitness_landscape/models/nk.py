import numpy as np
import random

from itertools import product
from typing import List, Optional, Union

from torch import ne

from ..core.landscape import FitnessLandscape
from ..core.fitness import NumericFitness
from ..core.sequence import BaseNumpySequence, BinarySequence, MultialleleSequence


def generate_NK_states(N: int,
                       K: Optional[int] = None,
                       alphabet: List = [0,1],
                       seed: Optional[int] = None,
                       adj_mat: Optional[np.ndarray] = None,
                       base_sequence: Optional[Union[List, str]] = None,
                       variable_sites: Optional[List[int]] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences and fitness values for a generalized NK landscape.

    Parameters
    ----------
    N : int
        Number of variable sites in each sequence. If variable sites is
        not specified but a base sequence is, the first N sites will be
        varied.
    K : int
        Number of interacting neighbors for each site. Not required if an
        adjacency matrix is provided.
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
        Indices of the sites to be varied in the `base_sequence`. The
        sites are assumed to be pre-zero indexed.

    Returns
    -------
    sequences : np.ndarray
        Array of generated sequences.
    fitness_values : np.ndarray
        Array of corresponding fitness values.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    alphabet_size = len(alphabet)
    allele_map = {allele: i for i, allele in enumerate(alphabet)}

    # If no variable sites assume all sites should vary
    if variable_sites is None:
        variable_sites = list(range(N))
    elif len(variable_sites) != N:
        raise ValueError("Length of variable_sites must equal to N.")
    
    # Check for either K or adjacency matrix
    if K is None and adj_mat is None:
        raise ValueError("Either K or adj_mat must be provided.")
    if K is not None and adj_mat is not None:
        print(
            "Warning: Both K and adj_mat provided. Using adj_mat for "
            "interactions."
        )

    if base_sequence is not None:
        if len(base_sequence) < N:
            raise ValueError(
                "Length of base_sequence must longer than or equal to N."
            )
        if any(i >= (len(base_sequence)) for i in variable_sites):
            raise IndexError(
                "All indices in variable_sites must correspond to an index "
                "in the provided base sequence."
            )
        # Check that all variable sites in the base sequence are in the alphabet
        for idx in variable_sites:
            if base_sequence[idx] not in alphabet:
                raise ValueError(
                    f"Character '{base_sequence[idx]}' at position {idx} of base_sequence "
                    f"is not in the provided alphabet."
                )

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

    # make fitness contributions for each interaction
    fitness_contrib = []
    for j in range(N):
        if adj_mat is not None:
            neighbours = np.where(adj_mat[j] == 1)[0]
            neighbours = neighbours[neighbours != j]
            n_interactions = 1 + len(neighbours)
        elif K is not None:
            n_interactions = K + 1
        else:
            raise ValueError("Either K or adj_mat must be provided.")
        fitness_contrib.append(np.random.rand(alphabet_size ** n_interactions))

    # get fitness contributions for each sequence
    for seq_idx, seq in enumerate(sequences):
        total_fit = 0.0
        for site_idx, site in enumerate(variable_sites):
            if adj_mat is not None:
                neighbors = np.where(adj_mat[site_idx] == 1)[0]
                indices = [site] + [variable_sites[x] for x in neighbors
                                    if x != site_idx]
            elif K is not None:
                interaction_options = [x for x in variable_sites
                                       if x != site]
                indices = [site] + random.sample(interaction_options, K)
            else:
                raise ValueError("Either K or adj_mat must be provided.")
            config = [seq[pos] for pos in indices]
            
            index = 0
            for allele_idx, allele in enumerate(config):
                numeric_allele = allele_map[allele]
                index += numeric_allele * (alphabet_size ** (len(config) - 1 - allele_idx))
            
            total_fit += fitness_contrib[site_idx][index]

        fitness_values[seq_idx] = total_fit / N

    return sequences, fitness_values

def create_gnk_landscape(N: int,
                         K: Optional[int] = None,
                         alphabet: List = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                         seed: Optional[int] = None,
                         adj_mat: Optional[np.ndarray] = None,
                         base_sequence: Optional[Union[List, str]] = None,
                         variable_sites: Optional[List[int]] = None,
                         **kwargs) -> FitnessLandscape:
    """
    Factory function to create a generalized NK fitness landscape.

    Parameters
    ----------
    N : int
        Number of variable sites in each sequence. If variable sites is
        not specified but a base sequence is, the first N sites will be
        varied.
    K : int
        Number of interacting neighbors for each site. If not specified,
        will be inferred from the adjacency matrix.
    alphabet : list
        The alphabet of characters or symbols to use for the sequences.
    seed : int, optional
        Random seed for reproducibility.
    adj_mat : np.ndarray, optional
        Adjacency matrix defining epistatic interactions.
    base_sequence : list, optional
        A template sequence.
    variable_sites : list of int, optional
        Indices of the sites to be varied in the `base_sequence`. The
        sites are assumed to be pre-zero indexed.
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

    sequences = [BaseNumpySequence(seq, alphabet=alphabet) for seq in sequences_np]

    replicates = [[val] for val in fitness_values]
    
    fitness_layers = {
        f'nk_k={K}': NumericFitness(
            name=f'nk_k={K}',
            values=replicates,
            metadata={'N': N, 'K': K, 'alphabet_size': alphabet_size}
        )
    }
    
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type='hamming',
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
    sequences_np, fitness_values = generate_NK_states(N, K, alphabet=[0,1], seed=seed)
    
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
    
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type='hamming',
        **kwargs
    )

def create_nk_multi_landscape(N: int,
                               K: int,
                               alphabet: List,
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
    sequences_np, fitness_values = generate_NK_states(N, K, alphabet=alphabet, seed=seed)
    
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
    
    return FitnessLandscape.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph_type='hamming',
        **kwargs
    )


