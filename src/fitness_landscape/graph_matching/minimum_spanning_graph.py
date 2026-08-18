"""Reconstruct latent graph structure from sparse observations."""

import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import List, Literal, Sequence, Tuple, Optional, Dict


# Geodesics must be additive - weights need to be converted to log-weigths.
def graph_to_length_matrix(G: nx.Graph,
                           weight_key: str = "weight",
                           transform: Literal["neglog", "reciprocal"] = "neglog",
                           eps: float = 1e-12) -> np.ndarray:
    """
    Convert a weighted similarity graph to an all-pairs length matrix
    for geodesics.

    Parameters
    ----------
    G : nx.Graph
        Symmetric weighted graph (edge attribute `weight_key` is a
        similarity).
    weight_key : str, default='weight'
        Edge attribute holding similarity weights.
    transform : {'neglog', 'reciprocal'}, default='neglog'
        Map similarity w -> length l:
            - 'neglog': l = -log(w + eps)  (natural for probabilities/kernels)
            - 'reciprocal': l = 1 / max(w, eps)
    eps : float, default=1e-12
        Numerical floor to avoid division by zero / log(0).

    Returns
    -------
    D : np.ndarray of shape (n, n)
        Dense all-pairs geodesic length matrix (shortest-path distances on l).
    """
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        # Failing a weight key, retrieve `1.0`
        w = float(d.get(weight_key, 1.0))
        rows += [i, j]; cols += [j, i]; data += [w, w]
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if transform == "neglog":
        data = np.asarray(W.data, dtype=float)
        if np.any(data < 0):
            raise ValueError(
                "neglog transform requires non-negative edge weights. "
                "Check that the graph uses the correct weight_key and does not contain negative similarities."
            )
        max_w = float(data.max()) if data.size else 0.0
        if max_w > 1.0 + 1e-9:
            raise ValueError(
                "neglog transform expects similarities in (0, 1]; "
                f"observed max weight {max_w:.6g}. If these are distances, use length_transform='reciprocal' "
                "or normalise your weights before calling graph_to_length_matrix."
            )
        L = W.copy().astype(float)
        L.data = -np.log(np.clip(data, eps, None))
    elif transform == "reciprocal":
        data = np.asarray(W.data, dtype=float)
        if np.any(data < 0):
            raise ValueError(
                "reciprocal transform requires non-negative edge weights. "
                "Check that the graph uses the correct weight_key."
            )
        L = W.copy().astype(float)
        L.data = 1.0 / np.maximum(data, eps)
    else:
        raise ValueError("transform must be 'neglog' or 'reciprocal'")

    D = shortest_path(L, directed=False, unweighted=False)
    if not np.all(np.isfinite(D)):
        raise ValueError(
            "Input graph appears to be disconnected under the provided weight_key/transform; "
            "shortest path distances contain inf. Reconnect the graph (e.g. lower the diffusion "
            "threshold) or run the analysis on each connected component separately."
        )
    return D



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
    p = min(p, n)
    landmarks = np.linspace(0, n - 1, p, dtype=int)
    D_land = sp_dist[np.ix_(landmarks, landmarks)]

    H = np.eye(p) - np.ones((p, p)) / p
    B = -0.5 * H @ (D_land ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:dim]
    L = np.diag(np.sqrt(np.maximum(eigvals[idx], 0)))
    X_land = eigvecs[:, idx] @ L

    X = np.zeros((n, dim))
    X[landmarks] = X_land
    base = X_land[0]
    A = 2 * (X_land - base)
    base_norm2 = float(np.dot(base, base))

    for i in range(n):
        if i in landmarks:
            continue
        dists = sp_dist[i, landmarks]
        b = (base_norm2 - np.einsum("ij,ij->i", X_land, X_land) + dists**2 - dists[0]**2)
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
        dists, idxs = tree.query(positions[i], k=min(k+1, len(positions)))
        for j in idxs[1:]:
            euclid = np.linalg.norm(positions[i] - positions[j])
            if sp_dist[i, j] - euclid > gap_threshold:
                gap_pairs.append(tuple(sorted((i, j))))
    return sorted(set(gap_pairs))

# When deficit max degree is not appropriate.
def self_tuned_graph(positions: np.ndarray,
                    k0: int = 10,
                    tau: float = 1.0) -> nx.Graph:
    """
    Build a variable-degree graph using a self-tuned (ckNN) rule.

    Parameters
    ----------
    positions : np.ndarray, shape (n, d)
        Node coordinates (e.g., from MDS/diffusion).
    k0 : int, default=10
        Local scale via distance to k0-th neighbor.
    tau : float, default=1.0
        Connect i<->j if ||xi-xj||^2 / (sigma_i * sigma_j) <= tau.
        Weight = exp(- ||xi-xj||^2 / (sigma_i * sigma_j)).

    Returns
    -------
    G : nx.Graph
        Symmetric weighted graph with variable degrees.
    """
    n = len(positions)
    tree = cKDTree(positions)
    sig = np.zeros(n)
    for i in range(n):
        dists, _ = tree.query(positions[i], k=min(k0+1, n))
        sig[i] = dists[-1] if len(dists) > 1 else 1.0
        if sig[i] <= 1e-12:
            sig[i] = np.median(dists[1:]) if len(dists) > 2 else 1.0

    G = nx.Graph()
    G.add_nodes_from(range(n))
    k_cand = max(4*k0, min(n-1, 40))
    for i in range(n):
        dists, idxs = tree.query(positions[i], k=min(k_cand+1, n))
        for d, j in zip(dists[1:], idxs[1:]):
            s = (d*d) / (sig[i]*sig[j] + 1e-12)
            if s <= tau:
                w = float(np.exp(-s))
                if G.has_edge(i, j):
                    G[i][j]['weight'] = max(G[i][j]['weight'], w)
                else:
                    G.add_edge(i, j, weight=w)
    return G


def smacof_sparse(pairs: Sequence[Tuple[int, int]],
                  deltas: np.ndarray,
                  weights: np.ndarray,
                  n: int,
                  X0: np.ndarray,
                  fixed_mask: Optional[np.ndarray] = None,
                  max_iter: int = 200,
                  tol: float = 1e-5) -> np.ndarray:
    """
    Sparse SMACOF stress majorization with optional fixed coordinates.

    Parameters
    ----------
    pairs : sequence of tuple (i, j)
        Index pairs with target dissimilarities e_ij.
    deltas : np.ndarray, shape (m,)
        Target e_ij for each pair.
    weights : np.ndarray, shape (m,)
        Non-negative weights w_ij for each pair.
    n : int
        Total number of points (observed + Steiner).
    X0 : np.ndarray, shape (n, d)
        Initial coordinates.
    fixed_mask : np.ndarray of bool, shape (n,), optional
        If provided, rows with True stay fixed at X0 during updates.
    max_iter : int, default=200
        Maximum iterations.
    tol : float, default=1e-5
        Relative tolerance on stress decrease for convergence.

    Returns
    -------
    X : np.ndarray, shape (n, d)
        Optimized coordinates.
    """
    X = X0.copy()
    d = X.shape[1]
    m = len(pairs)
    if fixed_mask is None:
        fixed_mask = np.zeros(n, dtype=bool)

    # Build sparse weight matrix W and an index.
    I = np.array([i for i, j in pairs], dtype=int)
    J = np.array([j for i, j in pairs], dtype=int)
    W = sp.coo_matrix((weights, (I, J)), shape=(n, n))
    W = W + W.T
    v = np.asarray(W.sum(axis=1)).ravel()
    V = sp.diags(v)

    def stress(X):
        diffs = X[I] - X[J]
        dists = np.linalg.norm(diffs, axis=1) + 1e-12
        return float(np.sum(weights * (dists - deltas)**2))

    prev = stress(X)
    for _ in range(max_iter):
        # Current distances
        diffs = X[I] - X[J]
        dists = np.linalg.norm(diffs, axis=1) + 1e-12
        # Build B matrix entries
        coeff = weights * (deltas / dists)
        data = -coeff
        B = sp.coo_matrix((data, (I, J)), shape=(n, n))
        B = B + B.T
        # Diagonal
        bdiag = -np.asarray(B.sum(axis=1)).ravel()
        B = B + sp.diags(bdiag)

        # SMACOF update: X_new = V^{-1} B X
        # Handle zero rows in V by safe inverse
        inv_v = np.zeros_like(v)
        mask = v > 0
        inv_v[mask] = 1.0 / v[mask]
        Vinv = sp.diags(inv_v)
        X_new = Vinv @ (B @ X)

        # Re-impose fixed rows
        X_new[fixed_mask] = X[fixed_mask]

        cur = stress(X_new)
        if (prev - cur) / max(prev, 1e-12) < tol:
            X = X_new
            break
        X = X_new
        prev = cur
    return X

def build_sparse_stress_system(D: np.ndarray,
                               base_pairs: int = 2000,
                               k_nbr: int = 6) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Build a sparse set of observed-observed stress pairs to stabilize
    layout.

    Parameters
    ----------
    D : np.ndarray
        Geodesic distances between observed nodes.
    base_pairs : int, default=2000
        Target number of random landmark pairs to include.
    k_nbr : int, default=6
        Add nearest-neighbor pairs per node (by D) to enforce local geometry.

    Returns
    -------
    pairs : list of tuple
        Pairs (i, j) included in stress.
    deltas : np.ndarray
        Target distances for pairs.
    weights : np.ndarray
        Weights (default 1.0 here).
    """
    n = D.shape[0]
    rng = np.random.default_rng(0)

    # Random landmark pairs
    idxs = rng.choice(n*n, size=min(base_pairs, n*n), replace=False)
    I = (idxs // n).astype(int)
    J = (idxs % n).astype(int)
    mask = I != J
    I, J = I[mask], J[mask]
    pairs = list(zip(I.tolist(), J.tolist()))
    deltas = D[I, J].astype(float)

    # Local neighbor pairs
    for i in range(n):
        nbrs = np.argsort(D[i])[:k_nbr+1]  # include self
        for j in nbrs[1:]:
            pairs.append((i, j))
    pairs = list(dict.fromkeys(tuple(sorted(p)) for p in pairs if D[p] > 0))

    I = np.array([i for i, j in pairs], dtype=int)
    J = np.array([j for i, j in pairs], dtype=int)
    deltas = D[I, J].astype(float)
    weights = np.ones_like(deltas)
    return pairs, deltas, weights


def add_steiner_chains(pairs: List[Tuple[int, int]],
                       deltas: np.ndarray,
                       weights: np.ndarray,
                       gap_pairs: List[Tuple[int, int]],
                       n_obs: int,
                       D: np.ndarray,
                       seg_per_pair: int = 1,
                       chain_weight: float = 5.0) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """
    Add Steiner "chains" to the sparse stress system for high-gap pairs.

    For each gap pair (i,j), add `seg_per_pair` Steiner nodes s1..sK and
    chain constraints:
        (i,s1), (s1,s2), ..., (sK,j)
    with target segment length ≈ D(i,j)/(K+1). This lets the optimization
    bend the path between i and j without moving i,j themselves (much).

    Parameters
    ----------
    pairs : list of tuple
        Observed-node pairs in the sparse stress system.
    deltas : numpy.ndarray
        Target distance for each pair.
    weights : numpy.ndarray
        Stress weight for each pair.
    gap_pairs : list of tuple
        High-gap pairs (i,j) where geodesic >> Euclidean.
    n_obs : int
        Number of observed nodes (existing indices are 0..n_obs-1).
    D : np.ndarray
        The geodesic distance matrix.
    seg_per_pair : int, default=1
        Number of Steiner segments per gap pair (K)
    chain_weight : float, default=5.0
        Weight multiplier for chain constraints (stronger than baseline).

    Returns
    -------
    pairs2, deltas2, weights2
        Augmented sparse stress system including Steiner chains.
    steiner_index : dict
        Mapping gap_pair_index -> list of Steiner node indices created.
    """
    pairs2 = list(pairs)
    deltas2 = deltas.tolist()
    weights2 = weights.tolist()
    steiner_index = {}
    next_idx = n_obs

    for gp_idx, (i, j) in enumerate(gap_pairs):
        Dij = float(D[i, j])
        if not np.isfinite(Dij) or Dij <= 0:
            # can't place a meaningful chain if endpoints are disconnected or zero-length
            continue

        K = max(1, int(seg_per_pair))
        seg_len = Dij / (K + 1)
        steiner_nodes = list(range(next_idx, next_idx + K))
        next_idx += K
        steiner_index[gp_idx] = steiner_nodes

        chain = [i] + steiner_nodes + [j]
        for a, b in zip(chain[:-1], chain[1:]):
            pairs2.append((a, b))
            deltas2.append(seg_len)
            weights2.append(chain_weight)

    return pairs2, np.array(deltas2, float), np.array(weights2, float), steiner_index

def update_chain_deltas_for_D(pairs: List[Tuple[int, int]],
                              deltas: np.ndarray,
                              weights: np.ndarray,
                              gap_pairs: List[Tuple[int, int]],
                              steiner_index: Dict[int, List[int]],
                              D: np.ndarray) -> np.ndarray:
    """
    After Steiner chains are added, set the *actual* target segment lengths from D.

    For gap pair (i,j) with K Steiner nodes, set each chain edge to δ = D[i,j]/(K+1).

    Parameters
    ----------
    pairs : list of tuple
        Pair indices including Steiner chain edges appended at the end.
    deltas : np.ndarray
        Current δ array; will be updated in-place copy.
    weights : np.ndarray
        Weights array (not modified).
    gap_pairs : list of tuple
        Gap pairs (i,j).
    steiner_index : dict
        gap_pair_index -> list of Steiner node indices.
    D : np.ndarray
        Geodesic distances between observed nodes.

    Returns
    -------
    deltas_new : np.ndarray
        Updated δ array with proper segment lengths for all chain edges.
    """
    deltas_new = deltas.copy()
    ptr = len(deltas) - 1
    desired = []
    for (i, j), nodes in zip(gap_pairs, steiner_index.values()):
        K = len(nodes)
        L = float(D[i, j]) / (K + 1) if (K + 1) > 0 else float(D[i, j])
        chain = [i] + nodes + [j]
        desired.append([(a, b, L) for a, b in zip(chain[:-1], chain[1:])])

    # Build a map (a,b) -> desired L (undirected)
    want = {}
    for segs in desired:
        for a, b, L in segs:
            key = tuple(sorted((a, b)))
            want[key] = L

    for k, (a, b) in enumerate(pairs):
        key = tuple(sorted((a, b)))
        if key in want:
            deltas_new[k] = want[key]
    return deltas_new

def reconstruct_latent_graph_with_steiner(G_obs: nx.Graph,
                                          gap_threshold: float = 0.5,
                                          p_landmarks: int = 200,
                                          base_pairs: int = 2000,
                                          k_local: int = 6,
                                          seg_per_pair: int = 1,
                                          k0_scale: int = 10,
                                          tau_cknn: float = 1.0,
                                          weight_key: str = "weight",
                                          length_transform: Literal["neglog", "reciprocal"] = "neglog",
                                          _keep_steiner_in_graph: bool = True,
                                          max_iter_smacof: int = 150) -> Tuple[nx.Graph, np.ndarray, List[int]]:
    """
    Reconstruct a latent/envelope graph using Steiner points optimized
    by sparse MDS stress and a variable-degree (self-tuned)
    completion—no fixed-degree cap.

    Parameters
    ----------
    G_obs : nx.Graph
        Observed symmetric weighted graph.
    gap_threshold : float, default=0.5
        Threshold for (geodesic - Euclidean) to flag gaps.
    p_landmarks : int, default=200
        Landmarks for initial MDS.
    base_pairs : int, default=2000
        Random observed-observed pairs in sparse stress.
    k_local : int, default=6
        Number of local neighbors per node added to sparse stress.
    seg_per_pair : int, default=1
        Steiner segments per gap pair (1-2 recommended).
    k0_scale : int, default=10
        Local scale for self-tuned graph.
    tau_cknn : float, default=1.0
        Threshold parameter for self-tuned graph.
    weight_key : str, default='weight'
        Edge attribute name for weights.
    length_transform : {'neglog','reciprocal'}, default='neglog'
        Similarity→length transform for geodesics.
    _keep_steiner_in_graph : bool, default=`True`
        Boolean to keep the steiner points in the latent graph.
    max_iter_smacof : int, default=150
        Max iterations for sparse SMACOF.

    Returns
    -------
    G_lat : nx.Graph
        Latent/envelope graph (variable degree; no fixed cap).
    X_opt : np.ndarray, shape (n_obs + n_steiner, 2)
        Optimized coordinates (observed first, then Steiner).
    steiner_nodes : list of int
        Indices (in X_opt) of Steiner nodes (range(n_obs,
        n_obs+n_steiner)).
    """
    params = {
        "gap_threshold": gap_threshold,
        "p_landmarks": p_landmarks,
        "base_pairs": base_pairs,
        "k_local": k_local,
        "seg_per_pair": seg_per_pair,
        "k0_scale": k0_scale,
        "tau_cknn": tau_cknn,
        "weight_key": weight_key,
        "length_transform": length_transform,
        "_keep_steiner_in_graph": _keep_steiner_in_graph,
        "max_iter_smacof": max_iter_smacof,
    }

    if G_obs.number_of_nodes() == 0:
        G_empty = nx.Graph()
        G_empty.graph["latent_reconstruction_params"] = params
        return G_empty, np.zeros((0, 2)), []

    nodes_obs = list(G_obs.nodes())
    idx_obs = {u: i for i, u in enumerate(nodes_obs)}
    n_obs = len(nodes_obs)

    # Geodesics.
    D = graph_to_length_matrix(G_obs, weight_key=weight_key, transform=length_transform)

    # Initial embedding and gaps.
    X0_obs = landmark_mds(D, p=p_landmarks, dim=2)
    gap_pairs = detect_gap_pairs_kdtree(X0_obs, D, gap_threshold, k=10)

    # Sparse stress system (observed-observed).
    pairs, deltas, weights = build_sparse_stress_system(D, base_pairs=base_pairs, k_nbr=k_local)

    # Add Steiner chains for gap pairs
    pairs2, deltas2, weights2, steiner_index = add_steiner_chains(
        pairs, deltas, weights, gap_pairs, D=D, n_obs=n_obs, seg_per_pair=seg_per_pair, chain_weight=5.0
    )
    # Set proper segment targets from D
    deltas2 = update_chain_deltas_for_D(pairs2, deltas2, weights2, gap_pairs, steiner_index, D)

    # Initial positions for (obs + steiner): put Steiner near the midpoint(s) of i,j
    n_steiner = sum(len(v) for v in steiner_index.values())
    X0 = np.zeros((n_obs + n_steiner, 2))
    X0[:n_obs] = X0_obs
    cur = n_obs
    for (i, j), nodes in zip(gap_pairs, steiner_index.values()):
        if not nodes:
            continue
        # simple linear interpolation along the segment i->j
        start = X0_obs[i]; end = X0_obs[j]
        for k, s in enumerate(nodes, start=1):
            t = k / (len(nodes) + 1)
            X0[cur] = (1 - t) * start + t * end
            cur += 1

    #  Optimize coordinates via sparse SMACOF
    X_opt = smacof_sparse(
        pairs2, deltas2, weights2,
        n=n_obs + n_steiner, X0=X0,
        fixed_mask=None,  # set to np.r_[np.ones(n_obs,dtype=bool), np.zeros(n_steiner,dtype=bool)] to fix observed
        max_iter=max_iter_smacof, tol=1e-5
    )

    # Build variable-degree envelope via self-tuned graph
    if _keep_steiner_in_graph:
        G_env = self_tuned_graph(X_opt, k0=k0_scale, tau=tau_cknn)
        # Map observed node labels into 0..n_obs-1; Steiner keep integer indices
        G_lat = nx.Graph()
        # Observed nodes keep original labels
        label_map = {i: nodes_obs[i] for i in range(n_obs)}
        # Steiner nodes get synthetic labels
        label_map.update({i: f"steiner_{i-n_obs}" for i in range(n_obs, n_obs + n_steiner)})
        for i in range(n_obs + n_steiner):
            G_lat.add_node(label_map[i])
        for u, v, d in G_env.edges(data=True):
            G_lat.add_edge(label_map[u], label_map[v], weight=float(d.get('weight', 1.0)))
        # union with observed edges (preserve strong observed links)
        for u, v, d in G_obs.edges(data=True):
            if not G_lat.has_edge(u, v):
                G_lat.add_edge(u, v, weight=float(d.get('weight', 1.0)))
            else:
                G_lat[u][v]['weight'] = max(G_lat[u][v]['weight'], float(d.get('weight', 1.0)))
    else:
        # Only observed nodes in the final latent envelope
        G_env_all = self_tuned_graph(X_opt, k0=k0_scale, tau=tau_cknn)
        G_env_obs = nx.Graph()
        G_env_obs.add_nodes_from(nodes_obs)
        # keep only edges among observed indices
        for u, v, d in G_env_all.edges(data=True):
            if u < n_obs and v < n_obs:
                G_env_obs.add_edge(nodes_obs[u], nodes_obs[v], weight=float(d.get('weight', 1.0)))
        # union with observed edges
        G_lat = nx.Graph()
        G_lat.add_nodes_from(nodes_obs)
        for u, v, d in G_env_obs.edges(data=True):
            G_lat.add_edge(u, v, weight=float(d.get('weight', 1.0)))
        for u, v, d in G_obs.edges(data=True):
            if not G_lat.has_edge(u, v):
                G_lat.add_edge(u, v, weight=float(d.get('weight', 1.0)))
            else:
                G_lat[u][v]['weight'] = max(G_lat[u][v]['weight'], float(d.get('weight', 1.0)))

    steiner_nodes = list(range(n_obs, n_obs + n_steiner))
    G_lat.graph["latent_reconstruction_params"] = params
    return G_lat, X_opt, steiner_nodes

# Theoretically weaker - more accurate???
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
    k : int, default=20
        Nearest neighbours considered per node.
    
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

# Theoretically weaker - more accurate???
def reconstruct_latent_graph_midpoint(G_obs: nx.Graph,
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
    
    k_edges : int, default=20,
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
