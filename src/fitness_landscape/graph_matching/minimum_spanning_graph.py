import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from typing import List, Literal


def landmark_mds(sp_dist: np.ndarray,
                 p: int = 200,
                 dim: int = 2) -> np.ndarray:
    """
    Landmark multidimensional scaling to embed p landmark points with
    classical MDS. Non-lanmark point are interpolatd by triangulation
    in distance space. Converts edge-weights to Euclidean embedding
    space.

    Parameters
    ----------
    sp_dist : np.ndarray
        The scipy sparse distance matrix. 

    p : int, default=200
        The number of landmark points.
    
    dim : int, default=2
        The Euclidean embedding dimension.
    
    Returns
    -------
    X : np.ndarray
        The embedded array.
    """
    n = sp_dist.shape[0]
    landmarks = np.linspace(0, n - 1, p, dtype=int)
    D_land = sp_dist[np.ix_(landmarks, landmarks)]
    
    # Classical MDS on landmarks
    H = np.eye(p) - np.ones((p, p)) / p
    B = -0.5 * H @ (D_land ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:dim]
    L = np.diag(np.sqrt(np.maximum(eigvals[idx], 0)))
    X_land = eigvecs[:, idx] @ L
    
    
    # Interpolate non-landmark points.
    X = np.zeros((n, dim))
    X[landmarks] = X_land
    for i in range(n):
        if i in landmarks:
            continue
        
        # Triangulate position from distances to landmarks
        dists = sp_dist[i, landmarks]
        
        # Least squares fit in Euclidean space
        A = 2 * (X_land - X_land[0])
        b = (np.linalg.norm(X_land[0]) ** 2 - np.linalg.norm(X_land, axis=1) ** 2
             + dists ** 2 - dists[0] ** 2)
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        X[i] = pos
    
    return X


def detect_gap_pairs_kdtree(positions: np.ndarray,
                            sp_dist: np.ndarray,
                            gap_threshold: float,
                            k: int = 10) -> List:
    """
    Function to check k nearest neighbours in embedding for metric gaps
    and defining the Steiner point candidates. 

    Parameters
    ----------
    positions : np.ndarray
        The embedded node positions
    
    sp_dist : np.ndarray
        The scipy sparse distance matrix. 
    
    gap_threshold : float
        The Eucliean distance threshold to define a "gap" (i.e., 
        where a node is expected but not observed).
    
    k : int, default=10
        The number of neighbors for clustering and subsequent pairwise
        comparison.

    Returns
    -------
    gap_pairs : List
        The list of pairs of nodes where a gap exists according to the
        gap distance threshold.
    """
    tree = cKDTree(positions)
    gap_pairs = []
    for i in range(len(positions)):
        dists, idxs = tree.query(positions[i], k=k+1)  # +1 for self
        for j in idxs[1:]:
            euclid_dist = np.linalg.norm(positions[i] - positions[j])
            if sp_dist[i, j] - euclid_dist > gap_threshold:
                gap_pairs.append((i, j))
    return gap_pairs


def heuristic_regular_supergraph_sparse(positions: np.ndarray,
                                        observed_edges: List,
                                        observed_weights: List,
                                        d_max: int,
                                        k: int = 20) -> nx.Graph:
    """
    Sparse candidate set via kNN search.

    Parameters
    ----------
    positions : np.ndarray
        The embedded node positions
    observed_edges : List[Tuple]
        The list of edges observed in the induced graph. 
    observed_weights : List[float]
        The list of edge weights observed in the induced graph.
    d_max : int
        The regular degree of the latent graph.
    
    Returns
    -------
    G : nx.Graph
        The latent graph
    """
    n = len(positions)
    tree = cKDTree(positions)
    candidates = []
    for i in range(n):
        dists, idxs = tree.query(positions[i], k=k+1)
        for dist, j in zip(dists[1:], idxs[1:]):
            if i < j:
                candidates.append((dist, i, j))
    candidates.sort()

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=positions[i])
    for (u, v), w in zip(observed_edges, observed_weights):
        G.add_edge(u, v, weight=w)
    degrees = dict(G.degree())

    for w, u, v in candidates:
        if degrees[u] < d_max and degrees[v] < d_max and not G.has_edge(u, v):
            G.add_edge(u, v, weight=w)
            degrees[u] += 1
            degrees[v] += 1
        if all(deg == d_max for deg in degrees.values()):
            break
    return G


def reconstruct_latent_graph(G_obs: nx.Graph,
                             gap_threshold: float = 0.5,
                             p_landmarks: int = 200,
                             k_gap: int = 10,
                             k_edges: int = 20) -> nx.Graph:
    """
    Function to reconstruct the latent unobserved graph that an
    observed graph is induced from. Approximates Steiner positions for
    latent nodes based on multi-dimensional scaling of edge weights to
    a Euclidean embedding space. Scales in approximately linear time
    at constant values of `p_landmarks` and `k_*`.

    Parameters
    ----------
    G_obs : nx.Graph
        The observed, induced graph. 
    
    gap_threshold : float, default=0.5
        The threshold to consider adding a Steiner node.
    
    p_landmarks : int, float = 200
        The number of explicit nodes to embed with MDS (scales in
        quadratic time.)
    
    k_gap : int, default=10,
        The k value for pairwise embedding comparisons. 
    
    k_edge : int, default=20,
        The k value to use for heuristic search pairwise comparisons.

    Returns
    -------
    G_lat : nx.Graph
        The latent graph.
    """
    observed_edges = list(G_obs.edges())
    observed_weights = [G_obs[u][v]['weight'] for u, v in observed_edges]
    d_max = max(dict(G_obs.degree()).values())

    # Sparse shortest paths
    sp_dist = shortest_path(nx.to_scipy_sparse_array(G_obs, weight='weight'), directed=False, unweighted=False)

    # Landmark MDS embedding
    positions_obs = landmark_mds(sp_dist, p=p_landmarks, dim=2)

    # Gap detection with KDTree
    gap_pairs = detect_gap_pairs_kdtree(positions_obs, sp_dist, gap_threshold, k=k_gap)

    # Add Steiner points
    # TODO: move away from midpoint Steiner points.
    steiner_positions = 0.5 * (positions_obs[[i for i, _ in gap_pairs]] + positions_obs[[j for _, j in gap_pairs]])
    positions_all = np.vstack([positions_obs, steiner_positions]) if steiner_positions.size else positions_obs

    # Complete graph with sparse kNN candidates
    G_lat = heuristic_regular_supergraph_sparse(positions_all, observed_edges, observed_weights, d_max, k=k_edges)
    return G_lat
