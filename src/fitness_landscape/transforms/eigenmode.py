from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import warnings
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape


def eigenmode_decomposition(graph: Union[nx.Graph, FitnessLandscape],
                            k: int = None,
                            matrix: Literal['adjacency', 'laplacian', 'transition', 'norm_laplacian'] = 'laplacian',
                            weight_key: str = 'weight',
                            dense_threshold: int = 5000) -> Tuple:
    """
    Compute eigenmode decomposition of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to decompose.
    k : int or None, optional
        Number of eigenmodes to compute. If None, compute all eigenpairs
        using a dense decomposition (may be expensive for large graphs).
    matrix : str, default = `laplacian`
        The graph matrix to decompose. Either Laplacian matrix or the
        adjacency matrix.
    weight_key : str, default='weight'
        Edge attribute used when constructing weighted graph matrices.
    dense_threshold : int, default=5000
        The node threshold count to compute sparse / dense matrices.
        
    Returns
    -------
    tuple
        (eigenvalues, eigenvectors)
    """
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph or FitnessLandscape")

    # Build the requested matrix (prefer sparse)
    if matrix == 'laplacian':
        M = nx.laplacian_matrix(graph, weight=weight_key).astype(float).tocsr()
        symmetric_psd = True
    
    elif matrix == 'norm_laplacian':
        M = nx.normalized_laplacian_matrix(graph, weight=weight_key).astype(float).tocsr()
        symmetric_psd = True
    
    elif matrix == 'adjacency':
        M = nx.adjacency_matrix(graph, weight=weight_key).astype(float).tocsr()
        symmetric_psd = True  # symmetric for undirected graphs (not PSD)
    
    elif matrix == 'transition':
        # Row-stochastic: A D^{-1}
        A = nx.adjacency_matrix(graph, weight=weight_key).astype(float).tocsr()
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d == 0.0] = 1.0
        Dinv = sp.diags(1.0 / d)
        M = sp.eye(A.shape[0], format='csr') - Dinv @ A
        symmetric_psd = False
    
    else:
        raise ValueError(f"Unsupported matrix: {matrix}")

    n = M.shape[0]

    def _dense_full(mat: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to perform dense matrix eigendecomposition.
        """
        A = mat.toarray() if sp.issparse(mat) else np.asarray(mat, dtype=float)
        # Use dense matrix decomposition.
        w, U = np.linalg.eigh(A)
        idx = np.argsort(w)
        return w[idx], U[:, idx]

    # Route based on size / k / symmetry
    if not sp.issparse(M):
        # Convert to CSR
        M = sp.csr_matrix(M)
    
    # If k is None, compute the full eigendecomposition (dense).
    if k is None:
        if n > dense_threshold:
            warnings.warn(
                "Computing all eigenpairs for a large graph; this may be slow or memory-intensive. "
                "Pass k to compute a truncated basis instead.",
                RuntimeWarning,
            )
        return _dense_full(M)

    # If small graph use dense.
    if n <= dense_threshold and k >= n:
        return _dense_full(M)

    k = int(k)
    if k <= 0 or k >= n:
        # Degenerate asks fall back to dense
        return _dense_full(M)

    # Sparse path
    if symmetric_psd and matrix in ('laplacian', 'norm_laplacian'):
        # Use shift–invert at sigma=0 to get smallest eigenpairs efficiently
        try:
            
            # Collect k is None type errors
            w, U = spla.eigsh(M, k=k, which='LM', sigma=0.0)
        
        except Exception:

            # Fallback to plain SM (no shift) if factorization fails
            w, U = spla.eigsh(M, k=k, which='SM')
    else:
        # Adjacency or others: get largest magnitude by default
        w, U = spla.eigsh(M, k=k, which='LM')

    # Sort by eigenvalues
    idx = np.argsort(w)
    return w[idx], U[:, idx]
