import numpy as np
import random

from itertools import product
from typing import List, Optional, Union
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
    rng = np.random.default_rng(seed)
    
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

    # If K == 0 special logic.
    if adj_mat is None and (K is not None and K == 0):
        # Binary: fitness = Hamming weight / N  → mean ~ 0.5 over the full cube
        if alphabet_size == 2 and set(alphabet) == {0, 1}:
            fitness_values = sequences.astype(float).mean(axis=1)
        else:
            # Multi-allele: map alleles to 0..A-1 and normalize each site by (A-1)
            inv_map = {allele: i for i, allele in enumerate(alphabet)}  # 0..A-1
            seq_num = np.vectorize(inv_map.__getitem__)(sequences)       # shape (2^N, N)
            fitness_values = (seq_num / float(alphabet_size - 1)).mean(axis=1)
        return sequences, fitness_values

    num_sequences = len(sequences)
    fitness_values = np.zeros(num_sequences)

    # Build neighbor sets in GLOBAL indices (matching the sequence positions)
    neighbor_sets_global: list[list[int]] = []
    if adj_mat is not None:
        if adj_mat.shape != (N, N):
            raise ValueError(f"adj_mat must be shape ({N},{N}), got {adj_mat.shape}")
        for si, site in enumerate(variable_sites):
            
            # neighbors are given in local [0..N-1] indexing of variable_sites
            nbr_local = np.where(adj_mat[si] == 1)[0]
            nbr_local = [j for j in nbr_local if j != si]  # exclude self
            
            # convert to global indices
            neigh_global = [variable_sites[j] for j in nbr_local]
            
            # sort to make the order deterministic
            idxs_global = [site] + sorted(neigh_global)
            neighbor_sets_global.append(idxs_global)

    else:
        # sample K neighbors ONCE per site from the other variable sites
        for si, site in enumerate(variable_sites):
            choices = [v for v in variable_sites if v != site]
            if K > len(choices):
                raise ValueError(f"K={K} exceeds available neighbor choices={len(choices)} for site {site}")
            neigh_global = rng.choice(choices, size=K, replace=False).tolist()
            idxs_global = [site] + sorted(neigh_global)  # sorted for stable ordering
            neighbor_sets_global.append(idxs_global)
                
    # Build one lookup table per site using fixed neighbors
    # Zero-center each table to reduce intercept bias
    fitness_contrib = []
    for idxs_global in neighbor_sets_global:
        arity = len(idxs_global)  # typically K+1
        table = rng.random(alphabet_size ** arity)
        table -= table.mean()     # zero-center subfunction (optional but helpful)
        fitness_contrib.append(table)

    # Evaluate sequences using the SAME order for indexing
    # Mixed-radix index consistent with the order used to build the table
    for seq_idx, seq in enumerate(sequences):
        total_fit = 0.0
        for si, idxs_global in enumerate(neighbor_sets_global):
            
            # extract config in the fixed, deterministic order
            config = [seq[pos] for pos in idxs_global]
            
            # compute base-(alphabet_size) index
            index = 0
            for allele in config:
                index = index * alphabet_size + allele_map[allele]
            total_fit += fitness_contrib[si][index]
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
                               K: Optional[int] = None,
                               seed: Optional[int] = None,
                               adj_mat: Optional[np.ndarray] = None,
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
    adj_mat : np.ndarray, optional
        Adjacency matrix defining epistatic interactions.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the FitnessLandscape
        constructor.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class representing the NK
        landscape.
    """
    sequences_np, fitness_values = generate_NK_states(N, 
                                                      K, 
                                                      alphabet=[0,1], 
                                                      seed=seed,
                                                      adj_mat=adj_mat)
    
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


