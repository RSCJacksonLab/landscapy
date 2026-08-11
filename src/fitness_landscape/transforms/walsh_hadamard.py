import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence, BinarySequence, generate_sequences
from ..utils import check_full_hamming

def _fwht_inplace(x: np.ndarray) -> None:
    n = x.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h << 1):
            j2 = i + h
            for j in range(i, j2):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h <<= 1

def _fwht(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=float).copy()
    _fwht_inplace(y)
    return y

def _popcount_order_masks(L: int):
    n = 1 << L
    orders = np.array([int(i).bit_count() for i in range(n)], dtype=int)
    return orders

def walsh_transform(landscape: FitnessLandscape,
                    order: int = None) -> np.ndarray:
    """
    Compute Walsh-Hadamard transform of a fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to transform.
    order : int, default=`None`
        Maximum order of coefficients to compute.
        
    Returns
    -------
    array-like
        Walsh coefficients.
    """
    # Confirm validity.
    res = check_full_hamming(landscape, check_graph=False, return_info=True)
    if not res.is_full_hamming:
        raise ValueError("Walsh-Hadamard transform requires a full binary Hamming cube.")
    # Collect lexigraphic ordering.
    L = res.L
    p = res.lex_perm
    if p is None:
        # fall back to code sort if needed / should never happen!
        p = np.argsort(res.codes)

    # Active signal (scalar) and reorder to lexicographic genotype order
    x = np.asarray(landscape.get_signal(), dtype=float)
    x = x[p]  # length 2^L

    # FWHT and orthonormal normalization
    y = _fwht(x) / np.sqrt(1 << L)   # y is the Walsh spectrum (modes 0..2^L-1)

    # Optional order truncation (zero out higher-order modes)
    if order is not None:
        orders = _popcount_order_masks(L)
        y = y * (orders <= int(order))

    return y

def walsh_coefficients(landscape: FitnessLandscape,
                       order: int = None) -> Dict:
    """
    Extract Walsh coefficients up to specified order.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.

    order : int, default=`None`
        Maximum order of coefficients to compute.
    
    Returns
    -------
    dict
        Dictionary mapping interaction terms to coefficients.
    """
    coeff = walsh_transform(landscape, order=None)  # compute full spectrum once
    L = int(np.log2(len(coeff)))

    result: Dict[str, float] = {}
    for mask in range(len(coeff)):
        r = int(mask).bit_count()
        if order is not None and r > order:
            continue
        term = "intercept" if r == 0 else ",".join(str(j) for j in range(L) if (mask >> j) & 1)
        result[term] = float(coeff[mask])
    return result
