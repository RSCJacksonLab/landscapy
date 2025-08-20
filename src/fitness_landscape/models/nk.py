import numpy as np
import random

from itertools import product
from typing import List, Optional, Union, Dict, Tuple
from ..core.landscape import FitnessLandscape
from ..core.fitness import NumericFitness
from ..core.sequence import BaseNumpySequence, BinarySequence, MultialleleSequence


def generate_NK_states(N: int,
                       K: Optional[int] = None,
                       alphabet: Union[List, Dict[int, List]] = (0, 1),
                       seed: Optional[int] = None,
                       adj_mat: Optional[np.ndarray] = None,
                       base_sequence: Optional[Union[List, str]] = None,
                       variable_sites: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to generate sequences and fitness values for a generalized
    NK landscape, supporting per-site alphabets and custom interaction
    matrices on a base sequence indexed by `n`.

    Parameters
    ----------
    N : int
        Number of variable sites.
    K : int, optional
        Number of interacting neighbors for each variable site. Not
        required if an adjacency matrix is provided.
    alphabet : list OR dict[int, list], default=[0,1]
        - If list: uniform alphabet for all variable sites.
        - If dict: per-site alphabet for specific global indices.
          Example: {0:['A','B'], 1:['A','B','C'], 5:['B','C']}
    seed : int, optional
        Random seed for reproducibility.
    adj_mat : np.ndarray, optional
        (N x N) adjacency matrix over the variable_sites order.
        If None, K-neighborhood is randomly constructed per variable site.
    base_sequence : list or str, optional
        Full template sequence (global coordinates). If provided, only
        `variable_sites` are varied; otherwise sequences are length N
        over the variable sites alone.
    variable_sites : list[int], optional
        Global indices (0-based) of the sites varied. If None and
        base_sequence is None, defaults to range(N). If base_sequence is given
        and variable_sites is None, defaults to the first N indices.

    Returns
    -------
    sequences : np.ndarray
        Generated sequences. If base_sequence is None, shape is
        (num_states, N). If base_sequence is provided, shape is
        (num_states, len(base_sequence)).
    fitness_values : np.ndarray
        Array of corresponding fitness values (shape: num_states,).
    """
    rng = np.random.default_rng(seed)

    # Validity check base sequence if base_sequence is not none.
    if base_sequence is not None:
        if isinstance(base_sequence, str):
            base_sequence = list(base_sequence)
        L_total = len(base_sequence)
        if variable_sites is None:
            variable_sites = list(range(N))  # “first N sites varied”
        if len(variable_sites) != N:
            raise ValueError("Length of variable_sites must equal N when base_sequence is provided.")
        if any(i >= L_total for i in variable_sites):
            raise IndexError("All variable_sites must be valid indices in base_sequence.")
    else:
        # Treat the NK problem as an N-dimensional space; no base sequence.
        L_total = N
        if variable_sites is None:
            variable_sites = list(range(N))
        if len(variable_sites) != N:
            raise ValueError("Length of variable_sites must equal N.")

    # Build per-site alphabets for ALL global indices.
    if isinstance(alphabet, dict):
        site_alpha: Dict[int, List] = {int(k): list(v) for k, v in alphabet.items()}
        
        # If alphabet is missing for a site in the alphanet Dict, throw error.
        missing = [i for i in variable_sites if i not in site_alpha]
        if missing:
            raise ValueError(
                f"Per-site alphabet missing for variable_sites {missing}. "
                "Add them to the `alphabet` dict."
            )
        uniform_alpha = None
    else:
        # Static alphabet case
        uniform_alpha = list(alphabet)
        site_alpha = {i: uniform_alpha for i in variable_sites}

    # Validate base_sequence characters belong to their site alphabets
    # The base sequence must align to the alphabet dict.
    if base_sequence is not None:
        for s in variable_sites:
            if base_sequence[s] not in site_alpha[s]:
                raise ValueError(
                    f"base_sequence[{s}]={base_sequence[s]!r} not in per-site alphabet {site_alpha[s]}."
                )

    # Build per-site allele maps for fast radix indexing.
    # Throw error if any sites are empty.
    allele_map: Dict[int, Dict] = {s: {a: idx for idx, a in enumerate(site_alpha[s])}
                                   for s in variable_sites}
    alpha_sizes = {s: len(site_alpha[s]) for s in variable_sites}
    if any(sz < 1 for sz in alpha_sizes.values()):
        raise ValueError("Each per-site alphabet must be non-empty.")

    # Build the cartesian set of sequences in NK landscape.
    # Order the variable sites as given (this is the “local” 0..N-1 order).
    var_order = list(variable_sites)
    var_alphs = [site_alpha[s] for s in var_order]

    if base_sequence is not None:
        # Full-length sequences where only variable_sites vary
        combos = product(*var_alphs)
        seqs = []
        for combo in combos:
            new_seq = list(base_sequence)  # copy
            for v, site in zip(combo, var_order):
                new_seq[site] = v
            seqs.append(new_seq)
        sequences = np.array(seqs, dtype=object if any(not isinstance(x, (int, np.integer)) for x in seqs[0]) else None)
    else:
        # Sequences are only the variable part; length N of variable sites.
        combos = product(*var_alphs)
        sequences = np.array([list(c) for c in combos], dtype=object if any(not isinstance(x, (int, np.integer)) for x in next(iter(var_alphs))) else None)

    num_sequences = len(sequences)
    if num_sequences == 0:
        return sequences, np.zeros(0, dtype=float)

    # K == 0 special case (additive)
    if adj_mat is None and (K is not None and K == 0):
        # Normalize per-site by its alphabet range (Ai-1).
        # If Ai==1, contribution is zero for that site.
        fitness_values = np.zeros(num_sequences, dtype=float)

        if base_sequence is not None:
            # evaluate at variable_sites only and average over N
            for r, seq in enumerate(sequences):
                contrib = 0.0
                for s in var_order:
                    Ai = alpha_sizes[s]
                    if Ai <= 1:
                        continue
                    contrib += allele_map[s][seq[s]] / float(Ai - 1)
                fitness_values[r] = contrib / float(N)
        else:
            # sequences are length N in the order var_order == [0..N-1] by default
            for r, seq_var in enumerate(sequences):
                contrib = 0.0
                for j, s in enumerate(var_order):  # s = global index, j = local position
                    Ai = alpha_sizes[s]
                    if Ai <= 1:
                        continue
                    contrib += allele_map[s][seq_var[j]] / float(Ai - 1)
                fitness_values[r] = contrib / float(N)

        return sequences, fitness_values

    # Build neighbor sets in GLOBAL coordinates (matching full sequence positions) 
    if adj_mat is not None:
        if adj_mat.shape != (N, N):
            raise ValueError(f"adj_mat must be shape ({N},{N}), got {adj_mat.shape}.")
        neighbor_sets_global: List[List[int]] = []
        for si, site in enumerate(var_order):
            nbr_local = np.where(adj_mat[si] == 1)[0]
            nbr_local = [j for j in nbr_local if j != si]
            neigh_global = [var_order[j] for j in nbr_local]
            idxs_global = [site] + sorted(neigh_global)
            neighbor_sets_global.append(idxs_global)
    else:
        if K is None:
            raise ValueError("Either `K` or `adj_mat` must be provided.")
        neighbor_sets_global = []
        for site in var_order:
            choices = [v for v in var_order if v != site]
            if K > len(choices):
                raise ValueError(f"K={K} exceeds available neighbor choices={len(choices)} for site {site}")
            neigh_global = rng.choice(choices, size=K, replace=False).tolist()
            idxs_global = [site] + sorted(neigh_global)
            neighbor_sets_global.append(idxs_global)

    # Build per-site NK tables with mixed-radix (per-site) bases.
    # Zero-centered.
    fitness_contrib = []
    for idxs_global in neighbor_sets_global:
        # total states = product over bases of participating sites
        bases = [alpha_sizes[s] for s in idxs_global]
        total_states = int(np.prod(bases))
        table = rng.random(total_states)
        table -= table.mean()
        fitness_contrib.append((idxs_global, bases, table))

    # Evaluate sequences using mixed-radix indexing
    fitness_values = np.zeros(num_sequences, dtype=float)

    def _mixed_radix_index(vals,
                           bases):
        """
        Helper function to compute index with left-to-right radix.
        """
        idx = 0
        for v, b in zip(vals, bases):
            idx = idx * b + v
        return idx

    if base_sequence is not None:
        # sequences are full length, read alleles directly by global index
        for r, seq in enumerate(sequences):
            total = 0.0
            for idxs_global, bases, table in fitness_contrib:

                digits = [allele_map[s][seq[s]] for s in idxs_global]
                total += table[_mixed_radix_index(digits, bases)]
            fitness_values[r] = total / float(N)
    else:
        # sequences are only the variable part in the order var_order
        # Build a map: global site to local position j in sequences row
        global_to_local = {s: j for j, s in enumerate(var_order)}
        for r, seq_var in enumerate(sequences):
            total = 0.0
            for idxs_global, bases, table in fitness_contrib:
                digits = [allele_map[s][seq_var[global_to_local[s]]] for s in idxs_global]
                total += table[_mixed_radix_index(digits, bases)]
            fitness_values[r] = total / float(N)

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
