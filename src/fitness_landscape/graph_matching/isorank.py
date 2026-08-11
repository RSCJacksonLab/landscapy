from __future__ import annotations

import networkx as nx
import numpy as np

from ..utils import cosine_similarity_matrix as _cosine_sim


def normalize_adj_matrix(G: nx.Graph) -> np.ndarray:
    """
    Construct a row-stochastic adjacency/transition matrix for ``G``.
    For sink rows (no outgoing mass), replace with uniform 1/n.
    """
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise TypeError("G must be an undirected NetworkX Graph")
    nodes = list(G.nodes())
    if not nodes:
        return np.zeros((0, 0), dtype=float)
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    row_sum = A.sum(axis=1, keepdims=True)
    n = A.shape[0]
    mask = row_sum > 0
    A_norm = np.zeros_like(A)
    if np.any(mask):
        A_norm[mask[:, 0]] = A[mask[:, 0]] / row_sum[mask[:, 0]]
    if np.any(~mask):
        A_norm[~mask[:, 0]] = 1.0 / n
    return A_norm


def cosine_similarity_matrix(F1: np.ndarray, F2: np.ndarray) -> np.ndarray:
    """Cosine similarity between feature matrices (rows = nodes)."""
    return _cosine_sim(F1, F2)


def isorank_with_features(  # noqa: D401
    G1: nx.Graph,
    G2: nx.Graph,
    F1: np.ndarray,
    F2: np.ndarray,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    IsoRank with feature prior: S = alpha A1^T S A2 + (1-alpha) S0,
    where S0 is the cosine similarity of node features.
    """
    n1 = len(G1)
    n2 = len(G2)
    if F1.shape[0] != n1 or F2.shape[0] != n2:
        raise ValueError("Feature rows must match number of nodes in corresponding graph")

    A1 = normalize_adj_matrix(G1)
    A2 = normalize_adj_matrix(G2)

    S0 = cosine_similarity_matrix(F1, F2)
    denom = float(S0.sum())
    S0 = S0 / denom if denom > 0 else np.full_like(S0, 1.0 / max(1, n1 * n2))

    S = S0.copy()
    for _ in range(max_iter):
        S_prev = S
        S = alpha * (A1.T @ (S @ A2)) + (1.0 - alpha) * S0
        if np.linalg.norm(S - S_prev, ord="fro") < tol:
            break
    return S


__all__ = ["normalize_adj_matrix", "cosine_similarity_matrix", "isorank_with_features"]
