import networkx as nx
import numpy as np

def normalize_adj_matrix(G: nx.DiGraph) -> np.ndarray:
    """
    Helper function to construct a normalised adjacency matrix of G.
    Outgoing edges from terminal nodes are replaced with a uniform
    distribution over all nodes. 

    Parameters
    ----------
    G : nx.DiGraph
        The directed (acyclic) graph. 
    
    Returns
    -------
    A : np.ndarray
        The normalised adjacency matrix.
    """
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    for i in range(n):
        row_sum = np.sum(A[i, :])
        if row_sum == 0:
            A[i, :] = 1.0 / n
        else:
            A[i, :] /= row_sum
    return A

def cosine_similarity_matrix(F1: np.ndarray,
                             F2: np.ndarray) -> np.ndarray:
    """
    Helper function to compute the cosine similarity of between two
    feature matrices. 

    Parameters
    ----------
    F1 : np.ndarray
        Feature matrix 1.
    
    F2 : np.ndarray
        Feature matrix 2.
    
    Returns
    -------
    np.ndarray
        Cosine similarity matrix of shape (n,n).
    """

    F1_norm = F1 / np.linalg.norm(F1, axis=1, keepdims=True)
    F2_norm = F2 / np.linalg.norm(F2, axis=1, keepdims=True)
    return np.dot(F1_norm, F2_norm.T)

def isorank_with_features(G1: nx.DiGraph, 
                          G2: nx.DiGraph, 
                          F1: np.ndarray, 
                          F2: np.ndarray,
                          alpha: float = 0.85,
                          max_iter: int = 100,
                          tol: float = 1e-6) -> np.ndarray:
    """
    Computes the IsoRank similarity matrix between two matrices with
    features.
    
    Parameters
    ----------
    G1 : nx.DiGraph
        DAG 1.
    
    G2 : nx.DiGraph
        DAG 2.
    
    F1 : np.ndarray
        Feature matrix 1.
    
    F2 : np.ndarray
        Feature matrix 2.

    alpha : float, default=`0.85`
        Damping factor.
    
    max_iter: int, default=`100`
        Maximum number of iterations.
    
    tol : float, default=`1e-6`
        Tolerance for convergence.
    
    Returns
    -------
    S : np.ndarray
        A similarity matrix S of shape (n1, n2), incorporating both
        topology and node features.
    """
    
    A1 = normalize_adj_matrix(G1)
    A2 = normalize_adj_matrix(G2)
    
    # Initialise the S0 matrix as the cosine similarity matrix.
    S0 = cosine_similarity_matrix(F1, F2)
    S0 = S0 / np.sum(S0)
    
    S = S0.copy()
    
    for it in range(max_iter):
        S_prev = S.copy()
        S = alpha * np.dot(A1.T, np.dot(S, A2)) + (1 - alpha) * S0
        
        if np.linalg.norm(S - S_prev, ord='fro') < tol:
            break
    
    return S