"""Compare observed, reconstructed, and phylogenetic graph structures."""

import math
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from networkx.algorithms.approximation.steinertree import steiner_tree
from networkx.algorithms.community import greedy_modularity_communities
from scipy.spatial.distance import cdist, squareform
from scipy.sparse.csgraph import shortest_path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans2
from scipy.stats import spearmanr
from numpy.linalg import svd
from typing import List, Union, Tuple, Dict, Optional, Sequence, Any
from .._optional import require_optional

sklearn_metrics = require_optional(
    "sklearn.metrics",
    extra="analysis",
    purpose="graph-induction alignment analysis",
)
average_precision_score = sklearn_metrics.average_precision_score
roc_auc_score = sklearn_metrics.roc_auc_score
from ..transforms.eigenmode import eigenmode_decomposition
from ..core.landscape import FitnessLandscape
from ..core.edge_schema import AUTO_EDGE_KEY
import statistics as stats
from scipy.optimize import linear_sum_assignment
from ..graph_matching import isorank_with_features
from ..utils import _compute_embeddings_from_sequences, geodesic_distance_matrix
from ..core.sequence import BaseNumpySequence
from .graph import resistance_distance_matrix


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
                  k: int = 20,
                  weight_key: str | None = AUTO_EDGE_KEY) -> float:
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
    weight_key : str or None, default="auto"
        Conductance attribute used for both spectra. ``None`` requests
        unweighted operators.

    Returns
    -------
    float
        The root-mean-squared difference between eigenvalues.
    """
    eigvals_a, _ = eigenmode_decomposition(Ga, k=k, weight_key=weight_key)
    eigvals_b, _ = eigenmode_decomposition(Gb, k=k, weight_key=weight_key)
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

    Returns
    -------
    dict
        Edge count and mean, median, and population standard deviation of the
        selected edge-length attribute.
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
    if not isinstance(G, nx.Graph) or G.is_directed():
        raise TypeError("leaf_spanning_tree requires an undirected graph")
    # Filter leaves that exist in G
    L = [u for u in leaves if u in G]
    if len(L) == 0:
        return nx.Graph()
    T = steiner_tree(G, L, weight=weight)
    U = T.copy()
    if U.number_of_nodes() == 0:
        return nx.Graph()
    if not nx.is_connected(U):
        comps = sorted(nx.connected_components(U), key=len, reverse=True)
        U = U.subgraph(comps[0]).copy()
    return U

def get_leaves(U: nx.Graph) -> List:
    """Return the degree-one nodes of an undirected graph.

    Parameters
    ----------
    U : networkx.Graph
        Input graph.

    Returns
    -------
    list
        Nodes whose degree is one.
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
    if not isinstance(T, nx.Graph) or T.is_directed():
        raise TypeError("suppress_degree_two requires an undirected graph")
    U = T.copy()
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
    Au = _ensure_graph(G_truth)
    Bu = _ensure_graph(G_recon)

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
    
    G_lat_recon : FitnessLandscape or nx.Graph
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
    G_lat_truth = _ensure_graph(G_lat_truth)
    G_lat_recon = _ensure_graph(G_lat_recon)
    G_induced = _ensure_graph(G_induced)
    
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
    if not isinstance(obj, nx.Graph) or obj.is_directed():
        raise TypeError("Expected FitnessLandscape or an undirected networkx Graph")
    return obj

def _sequence_key(seq: Union[BaseNumpySequence, np.ndarray, Sequence]) -> Tuple:
    """
    Obtain a hashable key representing the content of a sequence-like
    object. Strings are treated as iterables of characters.
    """
    if isinstance(seq, BaseNumpySequence):
        arr = seq.to_array()
    elif hasattr(seq, "to_array"):
        arr = np.asarray(seq.to_array())
    elif isinstance(seq, np.ndarray):
        arr = seq
    elif isinstance(seq, (list, tuple)):
        arr = np.asarray(seq)
    else:
        # Treat scalar/string as 1D sequence
        arr = np.asarray(list(seq))
    return tuple(map(str, np.asarray(arr).ravel()))

def _sequence_index(G: nx.Graph) -> Dict[Tuple, Tuple[Any, Any]]:
    """
    Build a mapping from sequence content -> (sequence_obj, node_id)
    using the first occurrence for each unique sequence.
    """
    index: Dict[Tuple, Tuple[BaseNumpySequence, any]] = {}
    for node, data in G.nodes(data=True):
        seq = data.get("sequence")
        if seq is None:
            continue
        try:
            key = _sequence_key(seq)
        except Exception:
            continue
        if key not in index:
            seq_obj = seq if isinstance(seq, BaseNumpySequence) else seq
            index[key] = (seq_obj, node)
    return index

def _node_sequence_key(G: nx.Graph, node) -> Tuple[Optional[BaseNumpySequence], Optional[Tuple]]:
    data = G.nodes.get(node, {})
    seq = data.get("sequence")
    if seq is None:
        return None, None
    try:
        key = _sequence_key(seq)
    except Exception:
        return None, None
    seq_obj = seq if isinstance(seq, BaseNumpySequence) else seq
    return seq_obj, key

def _resolve_sequence_alignment(diff_G: nx.Graph,
                                phy_G: nx.Graph,
                                extant_nodes: Optional[Sequence]) -> Tuple[List, List, List[Any]]:
    """
    Resolve matching nodes across graphs using sequence equality.

    Returns diffusion node order, phylogeny node order, and matching
    sequence objects (preferring diffusion sequences when available).
    """
    diff_index = _sequence_index(diff_G)
    phy_index = _sequence_index(phy_G)

    diffusion_nodes: List = []
    phy_nodes: List = []
    sequences: List[BaseNumpySequence] = []

    def _append_from_key(key: Tuple) -> bool:
        if key not in diff_index or key not in phy_index:
            return False
        seq_obj, diff_node = diff_index[key]
        phy_seq_obj, phy_node = phy_index[key]
        sequences.append(seq_obj if isinstance(seq_obj, BaseNumpySequence) else phy_seq_obj)
        diffusion_nodes.append(diff_node)
        phy_nodes.append(phy_node)
        return True

    if extant_nodes is None:
        shared_keys = [k for k in diff_index.keys() if k in phy_index]
        if shared_keys:
            for key in shared_keys:
                _append_from_key(key)
        else:
            overlap = [n for n in diff_G.nodes() if n in phy_G]
            for node in overlap:
                seq_obj, key = _node_sequence_key(diff_G, node)
                if key is None:
                    seq_obj, key = _node_sequence_key(phy_G, node)
                if key is None:
                    continue
                if _append_from_key(key):
                    continue
    else:
        for item in extant_nodes:
            diff_node = item if item in diff_G else None
            phy_node = item if item in phy_G else None
            seq_obj, key = (None, None)
            if diff_node is not None:
                seq_obj, key = _node_sequence_key(diff_G, diff_node)
            if key is None and phy_node is not None:
                seq_obj, key = _node_sequence_key(phy_G, phy_node)
            if key is None:
                try:
                    key = _sequence_key(item)
                    if isinstance(item, BaseNumpySequence):
                        seq_obj = item
                except Exception:
                    key = None
            if key is None or key not in diff_index or key not in phy_index:
                raise ValueError(f"Could not resolve shared sequence for extant node {item!r}.")
            resolved = diff_index[key]
            phy_resolved = phy_index[key]
            if diff_node is None:
                diff_node = resolved[1]
            if phy_node is None:
                phy_node = phy_resolved[1]
            sequences.append(seq_obj if isinstance(seq_obj, BaseNumpySequence) else resolved[0])
            diffusion_nodes.append(diff_node)
            phy_nodes.append(phy_node)

    if not diffusion_nodes or not phy_nodes:
        raise ValueError("No overlapping sequences between diffusion and phylogeny graphs.")

    return diffusion_nodes, phy_nodes, sequences

def _align_graph_triplet(diffusion_obj: Union[FitnessLandscape, nx.Graph],
                         knn_obj: Union[FitnessLandscape, nx.Graph],
                         phylo_obj: Union[FitnessLandscape, nx.Graph],
                         extant_nodes: Optional[Sequence]) -> Tuple[nx.Graph, nx.Graph, nx.Graph, List, List, List, List[Any]]:
    """
    Align diffusion, KNN, and phylogenetic graphs onto a shared set of
    extant nodes (preferring sequence matches when available).

    Returns the undirected graph objects followed by the aligned node
    orders for diffusion, KNN, phylogeny, and the associated sequence
    objects.
    """
    diff_G = _ensure_graph(diffusion_obj)
    knn_G = _ensure_graph(knn_obj)
    phy_G = _ensure_graph(phylo_obj)

    diff_nodes, phy_nodes, sequences = _resolve_sequence_alignment(diff_G, phy_G, extant_nodes)
    knn_index = _sequence_index(knn_G)
    knn_nodes: List = []

    for diff_node, phy_node, seq_obj in zip(diff_nodes, phy_nodes, sequences):
        # Prefer sequence-based resolution, fall back to node identifiers.
        candidate = None
        try:
            key = _sequence_key(seq_obj)
        except Exception:
            key = None
        if key is not None and key in knn_index:
            candidate = knn_index[key][1]
        elif diff_node in knn_G:
            candidate = diff_node
        elif phy_node in knn_G:
            candidate = phy_node
        else:
            raise ValueError(f"Could not align node '{diff_node}'/'{phy_node}' with the KNN graph.")
        knn_nodes.append(candidate)

    return diff_G, knn_G, phy_G, diff_nodes, knn_nodes, phy_nodes, sequences

def _patristic_distance_matrix(G: nx.Graph,
                               node_order: Sequence,
                               length_attr: str) -> np.ndarray:
    """
    Compute pairwise path lengths between ``node_order`` entries using
    ``length_attr`` as the branch length attribute.
    """
    n = len(node_order)
    D = np.full((n, n), np.inf, dtype=float)
    for i, src in enumerate(node_order):
        D[i, i] = 0.0
        if src not in G:
            continue
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=length_attr)
        for j, dst in enumerate(node_order):
            if dst in lengths:
                D[i, j] = float(lengths[dst])
    return D

def _positive_pairs_from_tree(dist_matrix: np.ndarray, k: int) -> set:
    """
    Identify positive index pairs from a distance matrix using a tree
    k-NN neighbourhood definition.
    """
    n = dist_matrix.shape[0]
    positives: set = set()
    if n <= 1 or k <= 0:
        return positives
    order = np.argsort(dist_matrix, axis=1)
    for i in range(n):
        count = 0
        for idx in order[i]:
            if idx == i:
                continue
            pair = (i, idx) if i < idx else (idx, i)
            positives.add(pair)
            count += 1
            if count >= k:
                break
    return positives

def _graph_pair_scores(G: nx.Graph,
                       node_order: Sequence,
                       weight_key: Optional[str],
                       default: float = 0.0) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Extract scores for every unordered node pair according to the
    weights (similarities) present in ``G``.
    """
    index = {node: idx for idx, node in enumerate(node_order)}
    best: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        if u not in index or v not in index:
            continue
        i, j = index[u], index[v]
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        score = float(data.get(weight_key, 1.0)) if weight_key else 1.0
        prev = best.get(key)
        if prev is None or score > prev:
            best[key] = score

    pairs: List[Tuple[int, int]] = []
    scores: List[float] = []
    n = len(node_order)
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            pairs.append(key)
            scores.append(float(best.get(key, default)))
    return pairs, np.asarray(scores, dtype=float)

def _labels_for_pairs(pairs: Sequence[Tuple[int, int]],
                      positives: set) -> np.ndarray:
    return np.asarray([1 if pair in positives else 0 for pair in pairs], dtype=int)

def _safe_ap_roc(y_true: np.ndarray,
                 y_score: np.ndarray) -> Tuple[float, float, str]:
    if y_true.size == 0:
        return float('nan'), float('nan'), "No pairs"
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan'), float('nan'), f"Degenerate labels (pos={n_pos}, neg={n_neg})"
    ap = average_precision_score(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return float(ap), float(auc), f"OK (pos={n_pos}, neg={n_neg})"

def _tree_topk_index(dist_matrix: np.ndarray, k: int) -> Dict[int, set]:
    n = dist_matrix.shape[0]
    topk: Dict[int, set] = {}
    if n == 0 or k <= 0:
        return topk
    order = np.argsort(dist_matrix, axis=1)
    for i in range(n):
        neighbours: List[int] = []
        for idx in order[i]:
            if idx == i:
                continue
            neighbours.append(idx)
            if len(neighbours) >= k:
                break
        topk[i] = set(neighbours)
    return topk

def _graph_topk_index(G: nx.Graph,
                      node_order: Sequence,
                      k: int,
                      weight_key: Optional[str]) -> Dict[int, set]:
    index = {node: idx for idx, node in enumerate(node_order)}
    topk: Dict[int, set] = {}
    if k <= 0:
        return topk
    for node in node_order:
        if node not in G:
            continue
        neighbours = [nbr for nbr in G.neighbors(node) if nbr in index]
        if weight_key:
            neighbours.sort(key=lambda nbr: float(G[node][nbr].get(weight_key, 1.0)), reverse=True)
        else:
            neighbours.sort(key=lambda nbr: index[nbr])
        topk[index[node]] = set(index[nbr] for nbr in neighbours[:k])
    return topk

def _precision_at_k(tree_topk: Dict[int, set],
                    graph_topk: Dict[int, set],
                    k: int) -> float:
    if k <= 0 or not tree_topk:
        return float('nan')
    vals: List[float] = []
    for idx, true_nbrs in tree_topk.items():
        pred = graph_topk.get(idx)
        if not pred:
            vals.append(0.0)
            continue
        vals.append(len(true_nbrs & pred) / float(k))
    if not vals:
        return float('nan')
    return float(np.mean(vals))

def _upper_triangular_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=float)
    mask = np.triu(np.ones_like(matrix, dtype=bool), 1)
    return matrix[mask]

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)

def _mutual_topk_from_weighted(G: nx.Graph,
                               node_order: Sequence,
                               k: int,
                               weight_key: Optional[str]) -> nx.Graph:
    sub = G.subgraph(node_order).copy()
    H = nx.Graph()
    H.add_nodes_from((n, dict(data)) for n, data in sub.nodes(data=True))
    if k <= 0:
        return H
    node_set = set(sub.nodes())
    topk: Dict[Any, set] = {}
    for u in sub.nodes():
        nbrs = [v for v in sub.neighbors(u) if v in node_set and v != u]
        if weight_key:
            nbrs.sort(key=lambda v: float(sub[u][v].get(weight_key, 1.0)), reverse=True)
        else:
            nbrs.sort()
        topk[u] = set(nbrs[:k])
    for u in sub.nodes():
        for v in topk.get(u, set()):
            if u in topk.get(v, set()):
                data = {}
                if weight_key and sub.has_edge(u, v):
                    data[weight_key] = float(sub[u][v].get(weight_key, 1.0))
                H.add_edge(u, v, **data)
    return H

def _prune_to_degree_k_unweighted(G: nx.Graph,
                                  node_order: Sequence,
                                  k: int,
                                  weight_key: Optional[str]) -> nx.Graph:
    sub = G.subgraph(node_order).copy()
    H = nx.Graph()
    H.add_nodes_from((n, dict(data)) for n, data in sub.nodes(data=True))
    if k <= 0:
        return H
    selection: Dict[Any, List[Any]] = {}
    for u in sub.nodes():
        nbrs = [v for v in sub.neighbors(u) if v in sub and v != u]
        if weight_key:
            nbrs.sort(key=lambda v: float(sub[u][v].get(weight_key, 1.0)), reverse=True)
        else:
            nbrs.sort()
        selection[u] = nbrs[:k]
    for u, nbrs in selection.items():
        for v in nbrs:
            if u in selection.get(v, []):
                data = {}
                if weight_key and sub.has_edge(u, v):
                    data[weight_key] = float(sub[u][v].get(weight_key, 1.0))
                H.add_edge(u, v, **data)
    return H

def compare_pairwise_rankings_to_phylogeny(diffusion_graph: Union[FitnessLandscape, nx.Graph],
                                           knn_graph: Union[FitnessLandscape, nx.Graph],
                                           phylo_graph: Union[FitnessLandscape, nx.Graph],
                                           *,
                                           extant_nodes: Optional[Sequence] = None,
                                           tree_length_key: str = "branch_length",
                                           diffusion_weight_key: Optional[str] = None,
                                           knn_weight_key: Optional[str] = None,
                                           tree_k_for_labels: int = 10,
                                           default_diffusion_score: float = 0.0,
                                           default_knn_score: float = 0.0) -> Dict[str, Any]:
    """
    Evaluate how well diffusion and KNN graphs rank phylogenetically
    close sequences above distant ones using density-free AP/ROC
    metrics derived from tree-based k-NN labels.

    Parameters
    ----------
    diffusion_graph : FitnessLandscape or nx.Graph
        Graph whose weighted edges encode diffusion similarities.
    knn_graph : FitnessLandscape or nx.Graph
        Graph capturing neighbourhood relations (typically k-NN).
    phylo_graph : FitnessLandscape or nx.Graph
        Ground-truth phylogenetic tree/graph.
    extant_nodes : Sequence, optional
        Optional subset of extant nodes or sequences to align across
        graphs. When omitted, the intersection determined via sequence
        equality is used.
    tree_length_key : str, default="branch_length"
        Edge attribute containing phylogenetic branch lengths.
    diffusion_weight_key : str, optional
        Edge attribute used as similarity score within the diffusion
        graph. If ``None`` the graph is treated as unweighted.
    knn_weight_key : str, optional
        Edge attribute used as similarity for the KNN graph.
    tree_k_for_labels : int, default=10
        Tree k-NN parameter that defines positive pairs.
    default_diffusion_score : float, default=0.0
        Score assigned to non-adjacent diffusion pairs.
    default_knn_score : float, default=0.0
        Score assigned to non-adjacent KNN pairs.

    Returns
    -------
    dict
        Results dictionary containing node identifiers, pair labels,
        raw scores, and AP/ROC metrics for both graphs.
    """
    (diff_G,
     knn_G,
     phy_G,
     diff_nodes,
     knn_nodes,
     phy_nodes,
     sequences) = _align_graph_triplet(diffusion_graph, knn_graph, phylo_graph, extant_nodes)

    n = len(sequences)
    if n < 2:
        raise ValueError("At least two aligned sequences are required for ranking comparison.")

    tree_matrix = _patristic_distance_matrix(phy_G, phy_nodes, tree_length_key)
    k_truth = min(max(1, tree_k_for_labels), n - 1)
    positives = _positive_pairs_from_tree(tree_matrix, k_truth)

    pairs, knn_scores = _graph_pair_scores(knn_G, knn_nodes, knn_weight_key, default_knn_score)
    _, diffusion_scores = _graph_pair_scores(diff_G, diff_nodes, diffusion_weight_key, default_diffusion_score)
    labels = _labels_for_pairs(pairs, positives)

    ap_knn, auc_knn, note_knn = _safe_ap_roc(labels, knn_scores)
    ap_diff, auc_diff, note_diff = _safe_ap_roc(labels, diffusion_scores)

    seq_ids = [_sequence_identifier(seq) for seq in sequences]
    pair_labels = [(seq_ids[i], seq_ids[j]) for i, j in pairs]

    return {
        "n_nodes": n,
        "sequence_ids": seq_ids,
        "tree_k_for_labels": k_truth,
        "pairs": pair_labels,
        "pair_indices": pairs,
        "labels": labels.tolist(),
        "knn": {
            "average_precision": ap_knn,
            "roc_auc": auc_knn,
            "status": note_knn,
            "scores": knn_scores.tolist(),
            "weight_key": knn_weight_key,
            "default_score": default_knn_score,
        },
        "diffusion": {
            "average_precision": ap_diff,
            "roc_auc": auc_diff,
            "status": note_diff,
            "scores": diffusion_scores.tolist(),
            "weight_key": diffusion_weight_key,
            "default_score": default_diffusion_score,
        },
    }

def compare_density_matched_geometry_to_phylogeny(diffusion_graph: Union[FitnessLandscape, nx.Graph],
                                                  knn_graph: Union[FitnessLandscape, nx.Graph],
                                                  phylo_graph: Union[FitnessLandscape, nx.Graph],
                                                  *,
                                                  extant_nodes: Optional[Sequence] = None,
                                                  tree_length_key: str = "branch_length",
                                                  diffusion_weight_key: Optional[str] = None,
                                                  knn_weight_key: Optional[str] = None,
                                                  resistance_weight_key: Optional[str] = None,
                                                  k_values: Sequence[int] = (2, 3, 4, 5, 6)) -> Dict[str, Any]:
    """
    Compare diffusion and KNN graphs to a phylogeny under matched graph
    densities by sweeping shared degree thresholds and evaluating both
    geometric fidelity (resistance vs. patristic distances) and local
    neighbourhood precision.

    Parameters
    ----------
    diffusion_graph : FitnessLandscape or nx.Graph
        Graph with diffusion weights.
    knn_graph : FitnessLandscape or nx.Graph
        Nearest neighbour graph to evaluate.
    phylo_graph : FitnessLandscape or nx.Graph
        Reference phylogeny.
    extant_nodes : Sequence, optional
        Optional subset of extant nodes/sequences to align across
        graphs.
    tree_length_key : str, default="branch_length"
        Edge attribute representing branch lengths in the phylogeny.
    diffusion_weight_key : str, optional
        Edge attribute for sorting/pruning diffusion edges.
    knn_weight_key : str, optional
        Edge attribute for sorting/pruning KNN edges.
    resistance_weight_key : str, optional
        Edge attribute to use when forming Laplacian resistance
        distances. Defaults to ``None`` (unweighted).
    k_values : Sequence[int], default=(4, 6, 8, 10, 12, 16, 20)
        Degree targets evaluated during the sweep.

    Returns
    -------
    dict
        Results dictionary with per-k summaries of resistance distance
        correlations, Precision@k, and edge counts.
    """
    (diff_G,
     knn_G,
     phy_G,
     diff_nodes,
     knn_nodes,
     phy_nodes,
     sequences) = _align_graph_triplet(diffusion_graph, knn_graph, phylo_graph, extant_nodes)

    n = len(sequences)
    if n < 2:
        raise ValueError("At least two aligned sequences are required for density-matched comparison.")

    tree_matrix = _patristic_distance_matrix(phy_G, phy_nodes, tree_length_key)
    tree_upper = _upper_triangular_values(tree_matrix)

    seq_ids = [_sequence_identifier(seq) for seq in sequences]
    rows: List[Dict[str, Any]] = []

    for k in k_values:
        k_int = int(k)
        if k_int < 1:
            continue
        k_eval = min(max(1, k_int), n - 1)

        diff_equal = _mutual_topk_from_weighted(diff_G, diff_nodes, k_eval, diffusion_weight_key)
        knn_equal = _prune_to_degree_k_unweighted(knn_G, knn_nodes, k_eval, knn_weight_key)

        R_diff = resistance_distance_matrix(diff_equal, diff_nodes, weight_key=resistance_weight_key)["resistance_mat"]
        R_knn = resistance_distance_matrix(knn_equal, knn_nodes, weight_key=resistance_weight_key)["resistance_mat"]

        rho_diff = _safe_spearman(tree_upper, _upper_triangular_values(R_diff))
        rho_knn = _safe_spearman(tree_upper, _upper_triangular_values(R_knn))

        tree_topk = _tree_topk_index(tree_matrix, k_eval)
        diff_topk = _graph_topk_index(diff_equal, diff_nodes, k_eval, diffusion_weight_key)
        knn_topk = _graph_topk_index(knn_equal, knn_nodes, k_eval, knn_weight_key)

        prec_diff = _precision_at_k(tree_topk, diff_topk, k_eval)
        prec_knn = _precision_at_k(tree_topk, knn_topk, k_eval)

        rows.append({
            "k": k_eval,
            "rho_resistance_vs_patristic_diffusion": rho_diff,
            "rho_resistance_vs_patristic_knn": rho_knn,
            "precision_at_k_diffusion": prec_diff,
            "precision_at_k_knn": prec_knn,
            "edges_diffusion": diff_equal.number_of_edges(),
            "edges_knn": knn_equal.number_of_edges(),
        })

    return {
        "n_nodes": n,
        "sequence_ids": seq_ids,
        "tree_length_key": tree_length_key,
        "diffusion_weight_key": diffusion_weight_key,
        "knn_weight_key": knn_weight_key,
        "resistance_weight_key": resistance_weight_key,
        "k_values": [int(k) for k in k_values if int(k) >= 1],
        "rows": rows,
    }

def _sequence_identifier(seq: Any) -> str:
    if isinstance(seq, BaseNumpySequence):
        return getattr(seq, "id", str(seq))
    return str(seq)

def _comb2(n: int) -> float:
    return 0.5 * n * (n - 1)

def _cluster_metrics(true_labels: Sequence[int],
                     pred_labels: Sequence[int]) -> Dict[str, float]:
    n = len(true_labels)
    if n == 0:
        return {"adjusted_rand_index": 0.0, "mutual_info": 0.0, "normalized_mutual_info": 0.0}
    table = defaultdict(int)
    counts_true = Counter()
    counts_pred = Counter()
    for t, p in zip(true_labels, pred_labels):
        table[(t, p)] += 1
        counts_true[t] += 1
        counts_pred[p] += 1

    if n < 2:
        return {"adjusted_rand_index": 0.0, "mutual_info": 0.0, "normalized_mutual_info": 0.0}

    sum_comb = sum(_comb2(v) for v in table.values())
    sum_true = sum(_comb2(v) for v in counts_true.values())
    sum_pred = sum(_comb2(v) for v in counts_pred.values())
    total = _comb2(n)
    expected = (sum_true * sum_pred) / total if total > 0 else 0.0
    max_index = 0.5 * (sum_true + sum_pred)
    denominator = max_index - expected
    ari = 0.0 if abs(denominator) < 1e-12 else (sum_comb - expected) / denominator

    # Mutual information
    mi = 0.0
    for (t, p), nij in table.items():
        if nij == 0:
            continue
        mi += (nij / n) * math.log((nij * n) / (counts_true[t] * counts_pred[p] + 1e-12) + 1e-12)
    # Entropies
    def _entropy(counter: Counter) -> float:
        h = 0.0
        for cnt in counter.values():
            if cnt == 0:
                continue
            frac = cnt / n
            h -= frac * math.log(frac + 1e-12)
        return h

    h_true = _entropy(counts_true)
    h_pred = _entropy(counts_pred)
    denom = math.sqrt(h_true * h_pred) if h_true > 0 and h_pred > 0 else 0.0
    nmi = mi / denom if denom > 0 else 0.0

    return {
        "adjusted_rand_index": float(ari),
        "mutual_info": float(mi),
        "normalized_mutual_info": float(nmi),
    }

def _spectral_cluster_labels(G: nx.Graph,
                             nodes: Sequence,
                             n_clusters: int,
                             *,
                             weight_key: Optional[str] = None,
                             random_state: Optional[int] = None) -> np.ndarray:
    n = len(nodes)
    if n == 0:
        return np.array([], dtype=int)
    if n_clusters <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    k = min(n_clusters, n)
    A = nx.to_numpy_array(G, nodelist=list(nodes), weight=weight_key)
    degrees = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    inv_sqrt[~np.isfinite(inv_sqrt)] = 0.0
    norm_adj = inv_sqrt[:, None] * A * inv_sqrt[None, :]
    L = np.eye(n) - norm_adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eig(L)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    features = eigvecs[:, 1:k] if k > 1 and eigvecs.shape[1] > 1 else eigvecs[:, :1]
    if features.ndim == 1:
        features = features[:, None]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    features = features / norms
    unique_rows = np.unique(features, axis=0)
    if unique_rows.shape[0] < k:
        k = unique_rows.shape[0]
        if k <= 1:
            return np.zeros(n, dtype=int)
    seed = random_state if random_state is not None else None
    try:
        centroids, labels = kmeans2(features, k, minit='++', seed=seed)
    except Exception:
        # Fallback: assign by nearest centroid from unique rows via simple heuristic
        base = unique_rows
        if base.shape[0] < k:
            k = base.shape[0]
        centroids = base[:k]
        dists = ((features[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    return labels.astype(int)

def _modularity_cluster_labels(G: nx.Graph,
                               nodes: Sequence,
                               weight_key: Optional[str] = None) -> np.ndarray:
    if len(nodes) == 0:
        return np.array([], dtype=int)
    subgraph = G.subgraph(nodes)
    communities = list(greedy_modularity_communities(subgraph, weight=weight_key))
    label_map: Dict[Any, int] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            label_map[node] = idx
    return np.array([label_map.get(node, -1) for node in nodes], dtype=int)

def _phylo_reference_labels(phy_G: nx.Graph,
                            nodes: Sequence,
                            n_clusters: int,
                            *,
                            weight_key: Optional[str] = None) -> np.ndarray:
    n = len(nodes)
    if n == 0:
        return np.array([], dtype=int)
    if n <= n_clusters:
        return np.arange(n, dtype=int)
    distances = np.zeros((n, n), dtype=float)
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(phy_G, src, weight=weight_key)
        for j, dst in enumerate(nodes):
            distances[i, j] = lengths.get(dst, np.inf)
    if np.isinf(distances).any():
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            raise ValueError("Phylogenetic graph is disconnected for provided nodes.")
        max_val = np.max(finite)
        distances = np.where(np.isinf(distances), max_val * 2.0, distances)
    condensed = squareform(distances, checks=False)
    Z = linkage(condensed, method="average")
    n_clusters = min(max(1, n_clusters), n)
    labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    return labels.astype(int)

def geodesic_distance_dict(G: Union[FitnessLandscape, nx.Graph],
                           nodes: Optional[Sequence] = None,
                           *,
                           weight_key: Optional[str] = None,
                           transform: Union[str, None] = "auto",
                           default_weight: float = 1.0,
                           eps: float = 1e-12) -> Dict[Tuple, float]:
    """
    Compute geodesic distances for all ordered pairs in ``nodes``.

    Parameters
    ----------
    G : FitnessLandscape or nx.Graph
        Input graph.
    nodes : Sequence, optional
        Node labels to include; defaults to all graph nodes.
    weight_key : str, optional
        Edge attribute containing weights/similarities.
    transform : str or None, optional
        Transform passed through to :func:`geodesic_distance_matrix`.
    default_weight : float, default=1.0
        Substitute value when an edge lacks ``weight_key``.
    eps : float, default=1e-12
        Numerical floor for log/inverse transforms.

    Returns
    -------
    dict
        Mapping ``(u, v) -> distance`` for each ordered node pair
        (diagonal included). Entries are symmetric, i.e. both
        ``(u, v)`` and ``(v, u)`` share the same value.
    """
    H = _ensure_graph(G)
    matrix, order = geodesic_distance_matrix(H,
                                             nodes,
                                             weight_key=weight_key,
                                             transform=transform,
                                             default_weight=default_weight,
                                             eps=eps)
    dist: Dict[Tuple, float] = {}
    n = len(order)
    for i in range(n):
        for j in range(n):
            u, v = order[i], order[j]
            val = float(matrix[i, j])
            dist[(u, v)] = val
    return dist

def compare_geodesic_distance_arrays(diffusion_graph: Union[FitnessLandscape, nx.Graph],
                                     phylo_graph: Union[FitnessLandscape, nx.Graph],
                                     *,
                                     extant_nodes: Optional[Sequence] = None,
                                     diffusion_weight_key: Optional[str] = "kernel_weight",
                                     phylo_weight_key: Optional[str] = "weight",
                                     diffusion_transform: Union[str, None] = "auto",
                                     phylo_transform: Union[str, None] = "auto",
                                     default_weight: float = 1.0,
                                     eps: float = 1e-12,
                                     drop_disconnected: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
    """
    Compute paired geodesic distances for matching sequence pairs in a
    diffusion graph and a phylogenetic graph.

    Parameters
    ----------
    diffusion_graph : FitnessLandscape or nx.Graph
        Graph whose nodes correspond to extant sequences.
    phylo_graph : FitnessLandscape or nx.Graph
        Phylogenetic graph containing the same extant labels (plus
        possible ancestors).
    extant_nodes : Sequence, optional
        Explicit list of node identifiers or sequence objects to compare.
        By default, overlap is determined via sequence equality (falling
        back to shared node identifiers only when necessary).
    diffusion_weight_key : str, optional
        Edge attribute used when computing diffusion geodesics.
    phylo_weight_key : str, optional
        Edge attribute used for phylogeny geodesics.
    diffusion_transform : str or None, optional
        Transform applied to diffusion edge weights (default 'auto').
    phylo_transform : str or None, optional
        Transform applied to phylogeny edge weights.
    default_weight : float, default=1.0
        Substitute value when an edge lacks the chosen weight key.
    eps : float, default=1e-12
        Numerical floor for log/inverse transforms.
    drop_disconnected : bool, default=True
        When True, omit node pairs where either graph has infinite
        distance.

    Returns
    -------
    tuple
        ``(diffusion_distances, phylo_distances, pairs)`` where the
        distance arrays are aligned to ``pairs`` (a list of
        sequence-object tuples ``(seq_i, seq_j)`` corresponding to
        ``diffusion_distances[k]`` and ``phylo_distances[k]``).
    """
    diff_G = _ensure_graph(diffusion_graph)
    phy_G = _ensure_graph(phylo_graph)

    diff_nodes, phy_nodes, seq_objects = _resolve_sequence_alignment(diff_G, phy_G, extant_nodes)

    diff_matrix, diff_order = geodesic_distance_matrix(diff_G,
                                                       diff_nodes,
                                                       weight_key=diffusion_weight_key,
                                                       transform=diffusion_transform,
                                                       default_weight=default_weight,
                                                       eps=eps)
    phy_matrix, phy_order = geodesic_distance_matrix(phy_G,
                                                     phy_nodes,
                                                     weight_key=phylo_weight_key,
                                                     transform=phylo_transform,
                                                     default_weight=default_weight,
                                                     eps=eps)

    if len(diff_order) != len(phy_order):
        raise ValueError("Diffusion and phylogeny geodesic matrices differ in size after alignment.")

    pairs: List[Tuple] = []
    diffusion_vals: List[float] = []
    phylo_vals: List[float] = []
    n = len(diff_order)

    if len(seq_objects) < n:
        # Pad missing sequences with node identifiers
        seq_objects = list(seq_objects) + [diff_order[i] for i in range(len(seq_objects), n)]

    for i in range(n):
        for j in range(i + 1, n):
            pair = (seq_objects[i], seq_objects[j])
            d_diff = float(diff_matrix[i, j])
            d_phy = float(phy_matrix[i, j])
            if drop_disconnected and (not np.isfinite(d_diff) or not np.isfinite(d_phy)):
                continue
            diffusion_vals.append(d_diff)
            phylo_vals.append(d_phy)
            pairs.append(pair)

    return np.asarray(diffusion_vals, dtype=float), np.asarray(phylo_vals, dtype=float), pairs

def compare_diffusion_clusters_to_phylogeny(diffusion_graph: Union[FitnessLandscape, nx.Graph],
                                            phylo_graph: Union[FitnessLandscape, nx.Graph],
                                            *,
                                            extant_nodes: Optional[Sequence] = None,
                                            n_clusters: int = 8,
                                            diffusion_weight_key: Optional[str] = None,
                                            phylo_weight_key: Optional[str] = None,
                                            random_state: Optional[int] = None,
                                            drop_unassigned: bool = True) -> Dict[str, Any]:
    """
    Compare community structure of a diffusion graph with clades from a
    ground-truth phylogeny via clustering agreement metrics.

    Parameters
    ----------
    diffusion_graph : FitnessLandscape or nx.Graph
        Graph on extant sequences (potentially dense).
    phylo_graph : FitnessLandscape or nx.Graph
        Phylogenetic tree containing extant tips (and ancestors).
    extant_nodes : Sequence, optional
        Explicit set of nodes or sequences to align. Defaults to
        overlap determined via sequence equality.
    n_clusters : int, default=8
        Target number of clades/clusters for comparison.
    diffusion_weight_key : str, optional
        Edge attribute for weighting spectral/modularity clustering.
    phylo_weight_key : str, optional
        Edge attribute holding branch lengths when building clade
        distances. If ``None``, unit lengths are assumed.
    random_state : int, optional
        Random seed forwarded to spectral clustering k-means.
    drop_unassigned : bool, default=True
        If True, nodes assigned ``-1`` (unreachable) are removed prior
        to computing metrics.

    Returns
    -------
    dict
        Dictionary containing partitions and agreement metrics for
        spectral and modularity-based communities.
    """
    diff_G = _ensure_graph(diffusion_graph)
    phy_G = _ensure_graph(phylo_graph)

    diff_nodes, phy_nodes, seq_objects = _resolve_sequence_alignment(diff_G, phy_G, extant_nodes)
    if len(diff_nodes) == 0:
        raise ValueError("No overlapping sequences between diffusion and phylogeny graphs.")

    n_clusters = max(1, min(n_clusters, len(diff_nodes)))
    spectral_labels = _spectral_cluster_labels(diff_G, diff_nodes, n_clusters,
                                               weight_key=diffusion_weight_key,
                                               random_state=random_state)
    modularity_labels = _modularity_cluster_labels(diff_G, diff_nodes,
                                                   weight_key=diffusion_weight_key)
    phylo_labels = _phylo_reference_labels(phy_G, phy_nodes, n_clusters,
                                           weight_key=phylo_weight_key)

    # Optionally drop nodes with undefined modularity label
    valid_indices = list(range(len(diff_nodes)))
    if drop_unassigned:
        valid_indices = [i for i, lab in enumerate(modularity_labels) if lab >= 0]
    if not valid_indices:
        raise ValueError("No nodes remained after filtering unassigned communities.")

    seq_ids = [_sequence_identifier(seq_objects[i]) for i in valid_indices]
    phy_labels_sel = phylo_labels[valid_indices]
    spectral_sel = spectral_labels[valid_indices]
    modularity_sel = modularity_labels[valid_indices]

    spectral_metrics = _cluster_metrics(phy_labels_sel.tolist(), spectral_sel.tolist())
    modularity_metrics = _cluster_metrics(phy_labels_sel.tolist(), modularity_sel.tolist())

    return {
        "n_nodes": len(seq_ids),
        "sequence_ids": seq_ids,
        "phylo_labels": phy_labels_sel.tolist(),
        "diffusion_nodes": [diff_nodes[i] for i in valid_indices],
        "phylo_nodes": [phy_nodes[i] for i in valid_indices],
        "spectral": {
            "labels": spectral_sel.tolist(),
            **spectral_metrics,
        },
        "modularity": {
            "labels": modularity_sel.tolist(),
            **modularity_metrics,
        },
    }

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
    _, U = eigenmode_decomposition(
        G,
        k=k,
        matrix='norm_laplacian',
        weight_key=None,
    )
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
    """Align two graphs with feature-aware IsoRank and evaluate the mapping.

    Align two graphs via IsoRank (with node features) and evaluate the
    induced mapping with edge precision/recall/F1 and spectral
    correlations over the first non-trivial Laplacian eigenvectors.

    Parameters
    ----------
    Ga : FitnessLandscape or networkx.Graph
        First undirected graph.
    Gb : FitnessLandscape or networkx.Graph
        Second undirected graph.
    alpha : float, default=0.85
        IsoRank mixing weight between topology and feature similarity.
    max_iter : int, default=100
        Maximum IsoRank iterations.
    tol : float, default=1e-6
        Frobenius-norm convergence tolerance.
    prefer_attrs : tuple of str, default=('emb_arr',)
        Node attributes considered for feature matrices in priority order.
    spectral_k : int, default=16
        Number of spectral feature modes used when preferred attributes are
        unavailable.
    spectral_corr_k : int, default=20
        Maximum non-trivial Laplacian modes compared after matching.
    use_plm_fallback : bool, default=True
        Compute protein language-model features when preferred attributes are
        absent and sequences are available.
    plm_model_name : str, default='facebook/esm2_t6_8M_UR50D'
        Hugging Face model used by the PLM fallback.
    plm_device : str, optional
        Torch device for PLM inference.
    plm_batch_size : int, default=64
        PLM inference batch size.
    use_ohe_only : bool, default=False
        Use averaged one-hot sequence features and bypass all other feature
        sources.

    Returns
    -------
    dict
        Hungarian node mapping derived from the IsoRank similarity matrix,
        mapped-edge precision/recall/F1, absolute correlation by matched
        Laplacian mode, mean spectral correlation, and graph/match sizes.

    Notes
    -----
    Unless ``use_ohe_only`` is true, node attributes in ``prefer_attrs`` are
    used first. Missing attributes trigger PLM features when enabled, then
    spectral features. Edge metrics compare the mapped edges of ``Ga`` with
    the subgraph of ``Gb`` induced by matched nodes.
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
    Au = A
    Bu = B
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
    wA, UA = eigenmode_decomposition(
        A,
        k=k_corr + 1,
        matrix='norm_laplacian',
        weight_key=None,
    )
    wB, UB = eigenmode_decomposition(
        B,
        k=k_corr + 1,
        matrix='norm_laplacian',
        weight_key=None,
    )
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
