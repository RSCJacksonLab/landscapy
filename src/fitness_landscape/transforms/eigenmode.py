"""Decompose graph operators into spectral eigenmodes."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx
import warnings
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.edge_schema import AUTO_EDGE_KEY, resolve_edge_attribute


def eigenmode_decomposition(graph: Union[nx.Graph, FitnessLandscape],
                            k: int = None,
                            matrix: Literal['adjacency', 'laplacian', 'transition', 'norm_laplacian'] = 'laplacian',
                            weight_key: str | None = AUTO_EDGE_KEY,
                            dense_threshold: int = 5000) -> Tuple:
    """
    Compute a real eigenmode decomposition of an undirected graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to decompose.
    k : int or None, optional
        Positive number of eigenmodes to compute. If None, compute all eigenpairs
        using a dense decomposition (may be expensive for large graphs).
    matrix : str, default = `laplacian`
        Matrix to decompose: adjacency, combinatorial Laplacian, symmetric
        normalized Laplacian, or ``transition``. The latter denotes the
        random-walk Laplacian ``I - D^-1 A``.
    weight_key : str or None, default="auto"
        Conductance attribute used to construct graph matrices. ``"auto"``
        resolves constructor metadata; ``None`` requests unweighted matrices.
    dense_threshold : int, default=5000
        The node threshold count to compute sparse / dense matrices.
        
    Returns
    -------
    tuple
        ``(eigenvalues, eigenvectors)`` ordered by ascending eigenvalue. For
        ``matrix='transition'``, columns are real right eigenvectors of the
        random-walk Laplacian and are orthonormal under node measure
        ``degree`` (or unit measure for isolates).

    Notes
    -----
    The random-walk Laplacian is generally nonsymmetric. Landscapy solves the
    similar symmetric normalized operator and maps modes back with
    ``D^-1/2``. Isolated nodes are assigned zero rows in both operators and
    therefore contribute stationary zero modes.
    """
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph or FitnessLandscape")
    if graph.is_directed():
        raise TypeError("eigenmode_decomposition requires an undirected graph")
    if k is not None:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise TypeError("k must be a positive integer or None")
        if k <= 0:
            raise ValueError("k must be a positive integer or None")
        k = int(k)
    if isinstance(dense_threshold, (bool, np.bool_)) or not isinstance(
        dense_threshold, (int, np.integer)
    ):
        raise TypeError("dense_threshold must be a non-negative integer")
    if dense_threshold < 0:
        raise ValueError("dense_threshold must be a non-negative integer")
    resolved_weight_key = resolve_edge_attribute(
        graph,
        "conductance",
        weight_key,
        required=False,
    )

    # Build the requested matrix (prefer sparse)
    mode_scale = None
    if matrix == 'laplacian':
        M = nx.laplacian_matrix(graph, weight=resolved_weight_key).astype(float).tocsr()
        symmetric_psd = True
    
    elif matrix == 'norm_laplacian':
        M = nx.normalized_laplacian_matrix(
            graph,
            weight=resolved_weight_key,
        ).astype(float).tocsr()
        symmetric_psd = True
    
    elif matrix == 'adjacency':
        M = nx.adjacency_matrix(graph, weight=resolved_weight_key).astype(float).tocsr()
        symmetric_psd = False  # symmetric for undirected graphs, but not PSD
    
    elif matrix == 'transition':
        # L_rw = I - D^-1 A is similar to the symmetric normalized
        # L_sym = I - D^-1/2 A D^-1/2 on positive-degree nodes.
        A = nx.adjacency_matrix(graph, weight=resolved_weight_key).astype(float).tocsr()
        d = np.asarray(A.sum(axis=1)).ravel()
        positive = d > 0.0
        inverse_root = np.zeros_like(d, dtype=float)
        inverse_root[positive] = 1.0 / np.sqrt(d[positive])
        Dinv_root = sp.diags(inverse_root)
        M = sp.diags(positive.astype(float)) - Dinv_root @ A @ Dinv_root
        M = M.astype(float).tocsr()
        # Unit measure for isolates retains their canonical basis modes;
        # positive-degree nodes use the standard degree measure.
        mode_scale = np.ones_like(d, dtype=float)
        mode_scale[positive] = inverse_root[positive]
        symmetric_psd = True
    
    else:
        raise ValueError(f"Unsupported matrix: {matrix}")

    n = M.shape[0]

    def _finalize(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sort, validate real semantics, and map transition modes back."""
        values = np.real_if_close(eigenvalues, tol=1000)
        vectors = np.real_if_close(eigenvectors, tol=1000)
        if np.iscomplexobj(values) or np.iscomplexobj(vectors):
            raise RuntimeError("Undirected graph eigendecomposition returned complex modes")
        values = np.asarray(values, dtype=float)
        vectors = np.asarray(vectors, dtype=float)
        order = np.argsort(values, kind="stable")
        values = values[order]
        vectors = vectors[:, order]
        if symmetric_psd:
            tolerance = np.finfo(float).eps * max(1, n) * 100.0
            values[np.abs(values) <= tolerance] = 0.0
        if mode_scale is not None:
            vectors = mode_scale[:, None] * vectors
        return values, vectors

    def _dense_full(mat: sp.spmatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to perform dense matrix eigendecomposition.
        """
        A = mat.toarray() if sp.issparse(mat) else np.asarray(mat, dtype=float)
        # Use dense matrix decomposition.
        w, U = np.linalg.eigh(A)
        return _finalize(w, U)

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
    if k >= n:
        return _dense_full(M)

    # Sparse path
    if symmetric_psd:
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

    return _finalize(w, U)
