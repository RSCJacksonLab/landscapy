import numpy as np
import networkx as nx
from typing import Optional, Union
from ..utils import cosine_similarity_matrix as _cosine_sim


def normalize_adj_matrix(G: Union[nx.Graph, nx.DiGraph]) -> np.ndarray:
    """
    Construct a row-stochastic adjacency/transition matrix for ``G``.
    For sink rows (no outgoing mass), replace with uniform 1/n.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph. For undirected graphs, adjacency is treated as
        symmetric and row-normalized.

    Returns
    -------
    A : np.ndarray
        Row-stochastic matrix of shape (n, n).
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph)):
        raise TypeError("G must be a NetworkX Graph or DiGraph")
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return np.zeros((0, 0), dtype=float)
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    row_sum = A.sum(axis=1, keepdims=True)
    n = A.shape[0]
    # Avoid division by zero; normalize or replace with uniform
    mask = row_sum > 0
    A_norm = np.zeros_like(A)
    if np.any(mask):
        A_norm[mask[:, 0]] = A[mask[:, 0]] / row_sum[mask]
    if np.any(~mask):
        A_norm[~mask[:, 0]] = 1.0 / n
    return A_norm


def cosine_similarity_matrix(F1: np.ndarray, F2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between feature matrices (rows=nodes).
    Delegates to utils.cosine_similarity_matrix for numerical stability.
    """
    return _cosine_sim(F1, F2)


def isorank_with_features(G1: Union[nx.Graph, nx.DiGraph],
                          G2: Union[nx.Graph, nx.DiGraph],
                          F1: np.ndarray,
                          F2: np.ndarray,
                          *,
                          alpha: float = 0.85,
                          max_iter: int = 100,
                          tol: float = 1e-6) -> np.ndarray:
    """
    IsoRank with feature prior: S = alpha A1^T S A2 + (1-alpha) S0,
    where S0 is the cosine similarity of node features.

    Parameters
    ----------
    G1, G2 : Graph/DiGraph
        Input graphs for alignment.
    F1, F2 : np.ndarray
        Feature matrices of shapes (n1, d) and (n2, d).
    alpha : float, default=0.85
        Damping factor.
    max_iter : int, default=100
        Maximum iterations.
    tol : float, default=1e-6
        Frobenius norm tolerance for convergence.

    Returns
    -------
    S : np.ndarray
        Similarity matrix of shape (n1, n2).
    """
    n1 = len(G1)
    n2 = len(G2)
    if F1.shape[0] != n1 or F2.shape[0] != n2:
        raise ValueError("Feature rows must match number of nodes in corresponding graph")

    A1 = normalize_adj_matrix(G1)
    A2 = normalize_adj_matrix(G2)

    S0 = cosine_similarity_matrix(F1, F2)
    denom = float(S0.sum())
    S0 = S0 / denom if denom > 0 else np.full_like(S0, 1.0 / (n1 * n2))

    S = S0.copy()
    for _ in range(max_iter):
        S_prev = S
        S = alpha * (A1.T @ (S @ A2)) + (1.0 - alpha) * S0
        if np.linalg.norm(S - S_prev, ord='fro') < tol:
            break
    return S
