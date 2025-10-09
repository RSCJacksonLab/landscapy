import numpy as np
import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from numpy.linalg import svd
from typing import List, Union, Tuple, Dict, Optional
from ..transforms.eigenmode import eigenmode_decomposition
from ..core.landscape import FitnessLandscape
import statistics as stats
from scipy.optimize import linear_sum_assignment
from ..graph_matching import isorank_with_features
from ..utils import _compute_embeddings_from_sequences


def _edge_set(G: nx.Graph) -> set:
    """
    Helper function to get the edge set of a network graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph. 
    
    Returns
    -------
    set
        The set of edges in the graph
    """
    return set(tuple(sorted(e)) for e in G.edges())

def _common_edges(Ga: nx.Graph,
                  Gb: nx.Graph) -> set:
    """
    Helper function to find the common edhes between two network
    graphs.

    Parameters
    ----------
    Ga : nx.Graph
        An input graph.

    Gb : nx.Graph
        An input graph. 

    Returns
    -------
    set
        The set of common edges between `Ga` and `Gb`
    """
    Ea, Eb = _edge_set(Ga), _edge_set(Gb)
    return Ea & Eb

def _total_weight(G: nx.Graph,
                  weight_key: str = 'weight'):
    """
    Helper function to compute the total weight of a network graph.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    
    weight_key : str, default=`weight`
        The key that edge weights are stored under. 
    """
    return float(sum(d.get(weight_key, 1.0) for _, _, d in G.edges(data=True)))

def procrustes(emb_a: np.ndarray,
               emb_b: np.ndarray) -> Union[np.ndarray, float]:
    """
    Procrustes align Y to X (both n x d), return aligned Y and RMSE.

    Parameters
    ----------
    emb_a : np.ndarray
        The reference input array.
    
    emb_b : np.ndarray
        The input array to be aligned.
    
    Returns
    -------
    Y_aligned : np.ndarray
        `emb_b` aligned to `emb_a`
    
    rmse : float
        The root-mean-squared error of the alignment between input
        arrays.
    """
    X, Y = emb_a, emb_b

    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    U, _, Vt = svd(Yc.T @ Xc, full_matrices=False)
    R = U @ Vt
    Y_aligned = Yc @ R
    rmse = np.sqrt(np.mean((Xc - Y_aligned)**2))
    return Y_aligned + X.mean(0, keepdims=True), rmse

def edge_prf_on_observed(G_true: nx.Graph,
                         G_rec: nx.Graph,
                         observed_nodes: List) -> Tuple:
    """
    Function to measure recall, precision and F1 performance on edge
    reconstruction during latent graph construction, given a ground
    truth. 

    Parameters
    ----------
    G_true : nx.Graph
        The ground truth graph
    G_rec : nx.Graph
        The reconstructed latent graph.
    observed_nodes : List
        The list of observed nodes.
    
    Returns
    -------
    P : float
        The precision of the latent reconstruction.
    
    Rr : float
        The recall of the latent graph reconstruction.
    
    F1 : float
        The F1 of the latent graph reconstruciton.
    """
    T = G_true.subgraph(observed_nodes)
    R = G_rec.subgraph(observed_nodes)
    Et, Er = _edge_set(T), _edge_set(R)
    inter = Et & Er
    P = len(inter) / max(1, len(Er))
    Rr = len(inter) / max(1, len(Et))
    F1 = 0.0 if (P+Rr)==0 else 2*P*Rr/(P+Rr)
    return P, Rr, F1


def sp_rmse(Ga: nx.Graph,
            Gb: nx.Graph,
            nodes: List,
            weight_key: str = 'weight') -> float:
    """
    Function to compute the stretch RMSE between two graphs, defined
    as the difference in shortest paths between the input graphs. 

    Parameters
    ----------
    Ga : nx.Graph
        An input graph. 

    Gb : nx.Graph
        An input graph. 
    
    nodes : List
        The observed nodes. 
    
    weight_key : str, default=`weight`
        The key edge weights are stored udner.
    
    Returns
    -------
    float
        The root-mean-squared error between shortest paths in the two
        input graphs.
    """

    A = shortest_path(nx.to_scipy_sparse_array(Ga.subgraph(nodes), weight=weight_key),
                      directed=False, unweighted=False)
    B = shortest_path(nx.to_scipy_sparse_array(Gb.subgraph(nodes), weight=weight_key),
                      directed=False, unweighted=False)
    mask = np.isfinite(A) & np.isfinite(B)
    diff = (A - B)[mask]
    return float(np.sqrt(np.mean(diff**2)))

def spectral_rmse(Ga: nx.Graph,
                  Gb: nx.Graph,
                  k: int = 20) -> float:
    """
    Function to compute the spectral RMSE between two input graphs.

    Parameters
    ----------
    Ga : nx.Graph
        An input graph.
    
    Gb : nx.Graph
        An input graph.
    
    k : int, default=20
        The number of Laplacian eigenvectors to compute.

    Returns
    -------
    float
        The root-mean-squared difference between eigenvalues.
    """
    eigvals_a, _ = eigenmode_decomposition(Ga, k=k)
    eigvals_b, _ = eigenmode_decomposition(Gb, k=k)
    kmin = min(eigvals_a.shape[0], eigvals_b.shape[0])
    return float(np.sqrt(np.mean((eigvals_a[:kmin] - eigvals_b[:kmin])**2)))

def edge_length_stats(G: nx.Graph,
                       weight_key: str = 'weight') -> Dict:
    """
    Function to compute edge-length statistics.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    
    weight_key : str, default=`weight`
        The key that edge weights are stored under.
    """
    L = [d.get(weight_key, 1.0) for _,_,d in G.edges(data=True)]
    return dict(n=len(L), mean=np.mean(L), median=np.median(L), std=np.std(L))

def leaf_spanning_tree(G: nx.Graph,
                       leaves: List,
                       weight: Optional[str] = None) -> nx.Graph:
    """
    Compute a Steiner tree spanning the provided leaves in ``G``.

    Parameters
    ----------
    G : nx.Graph
        The input graph (assumed undirected for topology).
    leaves : List
        Node identifiers in ``G`` to connect.
    weight : str or None, optional
        Edge attribute to use as weight. If ``None``, unit weights.

    Returns
    -------
    nx.Graph
        A connected subgraph spanning all reachable leaves. If the
        output has multiple components, the largest component is
        returned.
    """
    if not isinstance(G, nx.Graph):
        G = G.to_undirected()
    # Filter leaves that exist in G
    L = [u for u in leaves if u in G]
    if len(L) == 0:
        return nx.Graph()
    T = steiner_tree(G, L, weight=weight)
    U = T.to_undirected()
    if U.number_of_nodes() == 0:
        return nx.Graph()
    if not nx.is_connected(U):
        comps = sorted(nx.connected_components(U), key=len, reverse=True)
        U = U.subgraph(comps[0]).copy()
    return U

def get_leaves(U: nx.Graph) -> List:
    """
    Return leaf nodes (degree == 1) of an undirected graph ``U``.
    """
    return [n for n in U.nodes() if U.degree(n) == 1]

def suppress_degree2(T: nx.Graph,
                     keep_attr_weights: bool = True,
                     weight: str = 'weight') -> nx.Graph:
    """
    Suppress internal degree-2 nodes by shortcutting their neighbors.

    Parameters
    ----------
    T : nx.Graph
        Input tree-ish graph.
    keep_attr_weights : bool, default=True
        If True, accumulates the ``weight`` attribute when collapsing
        edges.
    weight : str, default='weight'
        Edge attribute to accumulate.

    Returns
    -------
    nx.Graph
        An undirected graph with degree-2 internal nodes suppressed.
    """
    U = T.to_undirected().copy()
    changed = True
    while changed:
        changed = False
        for v in list(U.nodes()):
            if v not in U:  # may be removed earlier in loop
                continue
            if U.degree(v) == 2 and v not in get_leaves(U):
                nbs = list(U.neighbors(v))
                if len(nbs) != 2:
                    continue
                n1, n2 = nbs
                w = None
                if keep_attr_weights and U.has_edge(n1, v) and U.has_edge(v, n2):
                    d1 = U[n1][v]
                    d2 = U[v][n2]
                    if weight in d1 and weight in d2:
                        w = d1[weight] + d2[weight]
                U.remove_node(v)
                if not U.has_edge(n1, n2):
                    U.add_edge(n1, n2)
                if w is not None:
                    U[n1][n2][weight] = w
                changed = True
                break
    return U

def leaf_splits(U: nx.Graph, leaf_set: List) -> set:
    """
    Return the set of bipartitions (splits) induced by deleting each
    edge. Represent each split by the smaller side as a frozenset of
    leaf labels.

    Parameters
    ----------
    U : nx.Graph
        A tree (undirected). Must satisfy |E| = |V| - 1.
    leaf_set : List
        The leaf labels to consider when forming splits.

    Returns
    -------
    set
        Set of frozenset leaf splits (smaller side).
    """
    U = U.copy()
    if U.number_of_nodes() == 0:
        return set()
    assert U.number_of_edges() == U.number_of_nodes() - 1, "Not a tree"
    leaves = set(leaf_set)
    splits = set()
    for u, v in list(U.edges()):
        if not U.has_edge(u, v):
            continue
        U.remove_edge(u, v)
        comp1 = set(nx.node_connected_component(U, u))
        comp2 = set(U.nodes()) - comp1
        L1 = frozenset([x for x in comp1 if x in leaves])
        L2 = frozenset([x for x in comp2 if x in leaves])
        side = L1 if len(L1) <= len(L2) else L2
        if len(side) >= 2 and len(leaves) - len(side) >= 2:
            splits.add(side)
        U.add_edge(u, v)
    return splits

def rf_distance(U1: nx.Graph, U2: nx.Graph, leaves: List) -> Tuple[int, float]:
    """
    Compute the Robinson-Foulds (RF) distance and normalized RF between
    two trees ``U1``, ``U2`` restricted to a common set of leaves.

    Parameters
    ----------
    U1 : nx.Graph
        Tree 1 (undirected).
    U2 : nx.Graph
        Tree 2 (undirected).
    leaves : List
        Leaf labels to consider for splits.

    Returns
    -------
    diff : int
        The RF distance (number of differing splits).
    nrf : float
        Normalized RF in [0, 1].
    """
    S1 = leaf_splits(U1, leaves)
    S2 = leaf_splits(U2, leaves)
    diff = len(S1 - S2) + len(S2 - S1)
    max_splits = len(S1) + len(S2)
    nrf = diff / max_splits if max_splits > 0 else 0.0
    return diff, float(nrf)

def tree_rf_dissimilarity(G_truth: Union[FitnessLandscape, nx.Graph],
                          G_recon: Union[FitnessLandscape, nx.Graph],
                          leaves: Optional[List] = None,
                          weight_key: str = 'weight') -> Dict:
    """
    Compare a reconstructed latent graph to a ground-truth phylogeny
    via a tree-based Robinson-Foulds (RF) distance on shared leaves.

    This function constructs Steiner trees in each graph connecting the
    specified leaves, suppresses internal degree-2 nodes to yield a
    leaf-labeled tree, and computes the RF distance between the two
    resulting trees.

    Parameters
    ----------
    G_truth : FitnessLandscape or nx.Graph
        Ground-truth phylogenetic tree/graph.
    G_recon : FitnessLandscape or nx.Graph
        Reconstructed latent graph from superscape alignment.
    leaves : List, optional
        Node labels to treat as leaves (must exist in both graphs).
        If ``None``, uses the intersection of degree-1 nodes from the
        undirected version of ``G_truth`` and nodes present in
        ``G_recon``.
    weight_key : str, default='weight'
        Edge attribute to use as distance/weight when building Steiner
        trees.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'rf_distance' (int): RF distance between trees.
        - 'normalized_rf' (float): Normalized RF in [0, 1].
        - 'n_leaves' (int): Number of leaves used for comparison.
        - 'n_splits_truth' (int): Number of non-trivial splits in truth tree.
        - 'n_splits_recon' (int): Number of non-trivial splits in recon tree.
    """
    # Resolve nx.Graphs
    A = G_truth.graph if isinstance(G_truth, FitnessLandscape) else G_truth
    B = G_recon.graph if isinstance(G_recon, FitnessLandscape) else G_recon
    if A is None or B is None:
        raise ValueError("Both inputs must be networkx graphs or FitnessLandscape with .graph")

    Au = A.to_undirected()
    Bu = B.to_undirected()

    if leaves is None:
        truth_leaves = set(get_leaves(Au))
        recon_nodes = set(Bu.nodes())
        leaves = sorted(list(truth_leaves & recon_nodes))

    # Guard: ensure we have at least 2 leaves (RF needs >= 4 for non-trivial splits)
    leaves = [u for u in leaves if u in Au and u in Bu]
    if len(leaves) == 0:
        return {
            'rf_distance': 0,
            'normalized_rf': 0.0,
            'n_leaves': 0,
            'n_splits_truth': 0,
            'n_splits_recon': 0,
        }

    # Build leaf-spanning Steiner trees
    T_truth = leaf_spanning_tree(Au, leaves, weight=weight_key)
    T_recon = leaf_spanning_tree(Bu, leaves, weight=weight_key)

    if T_truth.number_of_nodes() == 0 or T_recon.number_of_nodes() == 0:
        return {
            'rf_distance': 0,
            'normalized_rf': 0.0,
            'n_leaves': len(leaves),
            'n_splits_truth': 0,
            'n_splits_recon': 0,
        }

    # Suppress internal degree-2 nodes (topology simplification)
    U_truth = suppress_degree2(T_truth, keep_attr_weights=True, weight=weight_key)
    U_recon = suppress_degree2(T_recon, keep_attr_weights=True, weight=weight_key)

    # Ensure trees (|E| = |V| - 1) before computing splits
    if U_truth.number_of_nodes() > 0 and U_truth.number_of_edges() != U_truth.number_of_nodes() - 1:
        # In case of accidental cycles, pick a spanning tree
        U_truth = nx.minimum_spanning_tree(U_truth, weight=weight_key)
    if U_recon.number_of_nodes() > 0 and U_recon.number_of_edges() != U_recon.number_of_nodes() - 1:
        U_recon = nx.minimum_spanning_tree(U_recon, weight=weight_key)

    # Compute RF
    S_truth = leaf_splits(U_truth, leaves)
    S_recon = leaf_splits(U_recon, leaves)
    rf, nrf = rf_distance(U_truth, U_recon, leaves)

    return {
        'rf_distance': int(rf),
        'normalized_rf': float(nrf),
        'n_leaves': int(len(leaves)),
        'n_splits_truth': int(len(S_truth)),
        'n_splits_recon': int(len(S_recon)),
    }

def evaluate_reconstruction(G_lat_truth: Union[FitnessLandscape, nx.Graph],
                            G_induced: Union[FitnessLandscape, nx.Graph],
                            G_lat_recon: Union[FitnessLandscape, nx.Graph]) -> Dict:
    """
    Function to analyse the reconstruction of the latent graph, given
    the ground truth and the induced observed graph.

    Parameters
    ----------
    G_lat_truth : FitnessLandscape or nx.Graph
        THe ground truth latent graph.
    
    G_induced : FitnessLandscape or nx.Graph
        The observed induced subgraph.
    
    G_lat_recon: FitnessLandscape or nx.Graph
        The reconstructed latent Graph.
    
    Returns
    -------
    dict
        Results dictionary with
        - edge precision.
        - edge recall. 
        - edge F1.
        - stretch precision. 
        - spectral RMSE. 
        - edge weight statistics. 
        - Total edge weights of ground-truth and reconstructed graphs.
    """
    if isinstance(G_lat_truth, FitnessLandscape):
        G_lat_truth = G_lat_truth.graph

    if isinstance(G_lat_recon, FitnessLandscape):
        G_lat_recon = G_lat_recon.graph
    
    if isinstance(G_induced, FitnessLandscape):
        G_induced = G_induced.graph
    
    observed_nodes = list(G_induced.nodes())

    P, Rr, F1 = edge_prf_on_observed(G_lat_truth, G_lat_recon, observed_nodes)
    sp_rmse_obs = sp_rmse(G_induced, G_lat_truth, observed_nodes)
    sp_rmse_rec = sp_rmse(G_lat_recon, G_lat_truth, observed_nodes)

    spec_rmse = spectral_rmse(G_lat_truth, G_lat_recon, k=20)
    stats_true = edge_length_stats(G_lat_truth)
    stats_rec  = edge_length_stats(G_lat_recon)

    return {
        "edge_precision": P,
        "edge_recall": Rr,
        "edge_F1": F1,
        "sp_RMSE_observed_vs_truth": sp_rmse_obs,
        "sp_RMSE_recon_vs_truth": sp_rmse_rec,
        "spectral_RMSE": spec_rmse,
        "true_edge_length_stats": stats_true,
        "recon_edge_length_stats": stats_rec,
        "total_weight_true": _total_weight(G_lat_truth),
        "total_weight_recon": _total_weight(G_lat_recon),
    }

def _ensure_graph(obj: Union[FitnessLandscape, nx.Graph]) -> nx.Graph:
    if isinstance(obj, FitnessLandscape):
        obj = obj.graph
    if isinstance(obj, nx.DiGraph):
        return obj.to_undirected()
    if not isinstance(obj, nx.Graph):
        raise TypeError("Expected FitnessLandscape or networkx Graph/Digraph")
    return obj

def _collect_features(G: nx.Graph,
                      *,
                      prefer_attrs: Tuple[str, ...] = ("emb_arr",),
                      spectral_k: int = 16,
                      plm_fallback: bool = True,
                      plm_model_name: str = 'facebook/esm2_t6_8M_UR50D',
                      plm_device: Optional[str] = None,
                      plm_batch_size: int = 64,
                      original_obj: Optional[Union[FitnessLandscape, nx.Graph]] = None) -> np.ndarray:
    nodes = list(G.nodes())
    n = len(nodes)
    # Try preferred node attributes first
    for key in prefer_attrs:
        vals = []
        ok = True
        d0: Optional[int] = None
        for u in nodes:
            x = G.nodes[u].get(key)
            if x is None:
                ok = False
                break
            xv = np.asarray(x)
            if xv.ndim > 1:
                xv = xv.reshape(-1)
            if d0 is None:
                d0 = int(xv.shape[0])
            if xv.shape[0] != d0:
                ok = False
                break
            vals.append(xv)
        if ok and d0 is not None:
            return np.vstack(vals)

    # Fallback 1: PLM embeddings from sequences
    if plm_fallback:
        seqs = []
        have_all = True
        for u in nodes:
            s = G.nodes[u].get('sequence', None)
            if s is None:
                have_all = False
                break
            seqs.append(s)
        if not have_all and isinstance(original_obj, FitnessLandscape):
            try:
                seqs = list(original_obj.sequences)
                have_all = len(seqs) == n
            except Exception:
                have_all = False
        if have_all and len(seqs) == n:
            try:
                E = _compute_embeddings_from_sequences(seqs,
                                                      model_name=plm_model_name,
                                                      device=plm_device,
                                                      batch_size=plm_batch_size)
                return np.asarray(E)
            except Exception:
                # fall through to spectral
                pass

    # Fallback 2: spectral features (non-trivial modes)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    k = min(max(2, spectral_k + 1), n)
    _, U = eigenmode_decomposition(G, k=k, matrix='norm_laplacian')
    if U.shape[1] > 1:
        return U[:, 1:min(k, U.shape[1])]
    deg = np.array([G.degree(u) for u in nodes], dtype=float)[:, None]
    return deg

def _collect_ohe_features(G: nx.Graph,
                          *,
                          original_obj: Optional[Union[FitnessLandscape, nx.Graph]] = None) -> np.ndarray:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    # Prefer per-node sequences
    seqs = []
    have_all = True
    for u in nodes:
        s = G.nodes[u].get('sequence', None)
        if s is None:
            have_all = False
            break
        seqs.append(s)
    if not have_all and isinstance(original_obj, FitnessLandscape):
        try:
            seqs = list(original_obj.sequences)
            have_all = len(seqs) == n
        except Exception:
            have_all = False
    if not have_all:
        return np.zeros((n, 0), dtype=float)

    feats: list[np.ndarray] = []
    dim: Optional[int] = None
    for s in seqs:
        arr = None
        if hasattr(s, 'ungapped_arr'):
            try:
                arr = np.asarray(s.ungapped_arr, dtype=float)
            except Exception:
                arr = None
        if arr is None:
            try:
                arr = np.asarray(s.to_one_hot(), dtype=float)
            except Exception:
                arr = None
        if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
            v = np.zeros((1,), dtype=float)
        else:
            v = arr.mean(axis=0)
        if dim is None:
            dim = int(v.shape[0])
        elif v.shape[0] != dim:
            if v.shape[0] < dim:
                v = np.pad(v, (0, dim - v.shape[0]))
            else:
                v = v[:dim]
        feats.append(v.astype(float))
    return np.vstack(feats)

def evaluate_isorank_alignment(Ga: Union[FitnessLandscape, nx.Graph],
                               Gb: Union[FitnessLandscape, nx.Graph],
                               *,
                               alpha: float = 0.85,
                               max_iter: int = 100,
                               tol: float = 1e-6,
                               prefer_attrs: Tuple[str, ...] = ("emb_arr",),
                               spectral_k: int = 16,
                               spectral_corr_k: int = 20,
                               use_plm_fallback: bool = True,
                               plm_model_name: str = 'facebook/esm2_t6_8M_UR50D',
                               plm_device: Optional[str] = None,
                               plm_batch_size: int = 64,
                               use_ohe_only: bool = False) -> Dict:
    """
    Align two graphs via IsoRank (with node features) and evaluate the
    induced mapping with edge precision/recall/F1 and spectral
    correlations over the first non-trivial Laplacian eigenvectors.

    Feature selection per graph:
      - If use_ohe_only=True: use averaged one-hot/ungapped encodings.
      - Else, prefer node attributes in prefer_attrs; if missing and
        use_plm_fallback=True, compute PLM embeddings from sequences;
        otherwise use spectral features.
    """
    A = _ensure_graph(Ga)
    B = _ensure_graph(Gb)
    nodes_A = list(A.nodes())
    nodes_B = list(B.nodes())
    nA, nB = len(nodes_A), len(nodes_B)
    if nA == 0 or nB == 0:
        return {
            'edge_precision': 0.0,
            'edge_recall': 0.0,
            'edge_F1': 0.0,
            'mapping': {},
            'spectral_correlation_by_mode': {},
            'spectral_correlation_mean': 0.0,
            'n_matched': 0,
            'n_A': nA,
            'n_B': nB,
        }

    # Features
    if use_ohe_only:
        FA = _collect_ohe_features(A, original_obj=Ga)
        FB = _collect_ohe_features(B, original_obj=Gb)
    else:
        FA = _collect_features(A,
                               prefer_attrs=prefer_attrs,
                               spectral_k=spectral_k,
                               plm_fallback=use_plm_fallback,
                               plm_model_name=plm_model_name,
                               plm_device=plm_device,
                               plm_batch_size=plm_batch_size,
                               original_obj=Ga)
        FB = _collect_features(B,
                               prefer_attrs=prefer_attrs,
                               spectral_k=spectral_k,
                               plm_fallback=use_plm_fallback,
                               plm_model_name=plm_model_name,
                               plm_device=plm_device,
                               plm_batch_size=plm_batch_size,
                               original_obj=Gb)

    # Equalize feature dims
    d = min(FA.shape[1] if FA.ndim == 2 else 0, FB.shape[1] if FB.ndim == 2 else 0)
    if d == 0:
        # fallback to degree only
        FA = np.array([A.degree(u) for u in nodes_A], dtype=float)[:, None]
        FB = np.array([B.degree(v) for v in nodes_B], dtype=float)[:, None]
        d = 1
    else:
        if FA.shape[1] != d:
            FA = FA[:, :d]
        if FB.shape[1] != d:
            FB = FB[:, :d]

    # Run IsoRank and Hungarian
    S = isorank_with_features(A, B, FA, FB, alpha=alpha, max_iter=max_iter, tol=tol)
    r_idx, c_idx = linear_sum_assignment(-S)
    mapping = {nodes_A[i]: nodes_B[j] for i, j in zip(r_idx, c_idx)}

    # Edge precision/recall/F1 on matched nodes (undirected)
    Au = A.to_undirected()
    Bu = B.to_undirected()
    matched_A = set(nodes_A[i] for i in r_idx)
    matched_B = set(nodes_B[j] for j in c_idx)
    E_pred = set()
    for u, v in Au.edges():
        if u in matched_A and v in matched_A:
            mu, mv = mapping[u], mapping[v]
            E_pred.add(tuple(sorted((mu, mv))))
    E_true = set(tuple(sorted(e)) for e in Bu.subgraph(matched_B).edges())
    inter = E_pred & E_true
    P = len(inter) / max(1, len(E_pred))
    Rr = len(inter) / max(1, len(E_true))
    F1 = 0.0 if (P + Rr) == 0 else 2 * P * Rr / (P + Rr)

    # Spectral correlation on first k non-trivial modes
    k_corr = min(spectral_corr_k, max(1, min(nA, nB) - 1))
    wA, UA = eigenmode_decomposition(A, k=k_corr + 1, matrix='norm_laplacian')
    wB, UB = eigenmode_decomposition(B, k=k_corr + 1, matrix='norm_laplacian')
    UA = UA[:, 1:1 + k_corr] if UA.shape[1] > 1 else UA
    UB = UB[:, 1:1 + k_corr] if UB.shape[1] > 1 else UB
    idxA = list(r_idx)
    idxB = list(c_idx)
    corr_by_mode: Dict[int, float] = {}
    for j in range(min(UA.shape[1], UB.shape[1])):
        x = UA[idxA, j]
        y = UB[idxB, j]
        rx = x - x.mean()
        ry = y - y.mean()
        sx = float(np.linalg.norm(rx))
        sy = float(np.linalg.norm(ry))
        if sx == 0.0 or sy == 0.0:
            corr = 0.0
        else:
            corr = abs(float((rx @ ry) / (sx * sy)))
        corr_by_mode[j] = corr
    corr_mean = float(np.mean(list(corr_by_mode.values()))) if corr_by_mode else 0.0

    return {
        'edge_precision': float(P),
        'edge_recall': float(Rr),
        'edge_F1': float(F1),
        'mapping': mapping,
        'spectral_correlation_by_mode': corr_by_mode,
        'spectral_correlation_mean': corr_mean,
        'n_matched': int(len(mapping)),
        'n_A': int(nA),
        'n_B': int(nB),
    }
