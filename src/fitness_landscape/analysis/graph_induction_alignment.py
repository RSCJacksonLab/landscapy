import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from numpy.linalg import svd
from typing import List, Union, Tuple, Dict
from ..transforms.eigenmode import eigenmode_decomposition
from ..core.landscape import FitnessLandscape
import statistics as stats


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
    Ea, Eb = edge_set(Ga), _edge_set(Gb)
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
