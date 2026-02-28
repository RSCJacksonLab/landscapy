import math
import warnings
from itertools import combinations, product
from typing import Optional, List

import numpy as np

from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence
from ..core.fitness import NumericFitness


def _estimate_sequence_count(n_mutable_sites: int,
                             n_mutation_layers: int,
                             alphabet_size: int) -> int:
    total = 1
    for d in range(1, n_mutation_layers + 1):
        total += math.comb(n_mutable_sites, d) * (alphabet_size - 1) ** d
    return total


def _generate_ranked_effect_matrix(L: int,
                                   A: int,
                                   n_components: int,
                                   noise_scale: float,
                                   rng: np.random.Generator) -> np.ndarray:
    M = np.zeros((L, A), dtype=float)
    for _ in range(n_components):
        position_weights = rng.normal(0.0, 1.0, size=L)
        aa_preferences = rng.normal(0.0, 1.0, size=A)
        M += np.outer(position_weights, aa_preferences)

    if noise_scale > 0.0:
        signal_std = float(M.std())
        if signal_std > 0.0:
            M += rng.normal(0.0, noise_scale * signal_std, size=M.shape)

    return M


def _enumerate_ranked_dms_sequences(wildtype: np.ndarray,
                                    mutable_positions: List[int],
                                    alphabet_size: int,
                                    n_mutation_layers: int) -> np.ndarray:
    wt = np.asarray(wildtype, dtype=int)
    sequences = [wt.copy()]

    allele_options = {
        pos: [aa for aa in range(alphabet_size) if aa != wt[pos]]
        for pos in mutable_positions
    }

    for d in range(1, n_mutation_layers + 1):
        for pos_combo in combinations(mutable_positions, d):
            choices = [allele_options[pos] for pos in pos_combo]
            for muts in product(*choices):
                seq = wt.copy()
                for pos, aa in zip(pos_combo, muts):
                    seq[pos] = aa
                sequences.append(seq)

    return np.asarray(sequences, dtype=int)


def create_ranked_dms_landscape(
    L: int,
    A: int = 20,
    n_components: int = 1,
    n_mutation_layers: int = 1,
    n_mutable_sites: Optional[int] = None,
    noise_scale: float = 0.01,
    seed: Optional[int] = None,
    wildtype: Optional[List[int]] = None,
    **kwargs
) -> FitnessLandscape:
    """
    Factory function to create a rank-controlled DMS fitness landscape.

    Parameters
    ----------
    L : int
        Number of positions in the sequence.
    A : int, default=20
        Alphabet size.
    n_components : int, default=1
        Number of rank-1 components in the fitness matrix.
    n_mutation_layers : int, default=1
        Maximum number of simultaneous mutations from the wild type.
    n_mutable_sites : int, optional
        Number of mutable positions. If None, all sites are mutable.
    noise_scale : float, default=0.01
        Additive Gaussian noise as a fraction of signal std.
    seed : int, optional
        Random seed for reproducibility.
    wildtype : list of int, optional
        Wild-type sequence as 0-indexed amino acid indices.

    Returns
    -------
    FitnessLandscape
        An instance of the FitnessLandscape class.
    """
    if L < 1:
        raise ValueError("L must be >= 1.")
    if A < 2:
        raise ValueError("A must be >= 2.")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if n_mutation_layers < 1:
        raise ValueError("n_mutation_layers must be >= 1.")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be >= 0.")

    rng = np.random.default_rng(seed)

    if wildtype is None:
        wildtype_arr = rng.integers(0, A, size=L).astype(int, copy=False)
    else:
        wildtype_arr = np.asarray(wildtype, dtype=int).ravel()
        if wildtype_arr.shape[0] != L:
            raise ValueError("wildtype must be length L.")
        if np.any(wildtype_arr < 0) or np.any(wildtype_arr >= A):
            raise ValueError("wildtype entries must be in [0, A).")

    if n_mutable_sites is None:
        n_mutable_sites_value = L
        mutable_positions = list(range(L))
    else:
        if n_mutable_sites < 1 or n_mutable_sites > L:
            raise ValueError("n_mutable_sites must be in [1, L].")
        n_mutable_sites_value = n_mutable_sites
        mutable_positions = rng.choice(L, size=n_mutable_sites, replace=False).tolist()
        mutable_positions.sort()

    if n_mutation_layers > n_mutable_sites_value:
        raise ValueError("n_mutation_layers cannot exceed n_mutable_sites.")

    est_count = _estimate_sequence_count(n_mutable_sites_value, n_mutation_layers, A)
    if est_count > 5_000_000:
        warnings.warn(
            f"Estimated sequence count {est_count} is large; consider reducing "
            "n_mutation_layers or n_mutable_sites.",
            RuntimeWarning,
        )

    effect_matrix = _generate_ranked_effect_matrix(
        L=L,
        A=A,
        n_components=n_components,
        noise_scale=noise_scale,
        rng=rng,
    )

    sequences_np = _enumerate_ranked_dms_sequences(
        wildtype=wildtype_arr,
        mutable_positions=mutable_positions,
        alphabet_size=A,
        n_mutation_layers=n_mutation_layers,
    )

    fitness_values = effect_matrix[np.arange(L), sequences_np].sum(axis=1)

    alphabet = list(range(A))
    sequences = [BaseNumpySequence(seq, alphabet=alphabet) for seq in sequences_np]
    replicates = [[float(val)] for val in fitness_values]

    layer_name = f"ranked_dms_n={n_components}_L={L}_layers={n_mutation_layers}"
    metadata = {
        "n_components": n_components,
        "noise_scale": noise_scale,
        "L": L,
        "A": A,
        "n_mutation_layers": n_mutation_layers,
        "n_mutable_sites": n_mutable_sites,
        "mutable_positions": mutable_positions,
        "seed": seed,
        "wildtype": wildtype_arr.tolist(),
    }

    fitness_layers = {
        layer_name: NumericFitness(
            name=layer_name,
            values=replicates,
            metadata=metadata,
        )
    }

    return FitnessLandscape.build(
        sequences=sequences,
        fitness_layers=fitness_layers,
        graph="hamming",
        **kwargs,
    )
