from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Literal
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..graph_matching.minimum_spanning_graph import reconstruct_latent_graph_with_steiner
from ..core.landscape import FitnessLandscape


def _ensure_affinity(G: nx.Graph,
                     length_key: str="weight",
                     sim_key: str="sim",
                     tau: float=None) -> None:
    """
    Ensure an affinity on G. If sim_attr missing, build
    sim = exp(-length/tau). If tau is None, use median(length) as
    scale. Similarity is necessary for Dirichlet eigenfunctions and the
    latent graph proxy will return distances. 

    Parameters
    ----------
    G : nx.Graph
        The graph to process / validate.
    
    length_key : str, default=`weight`
        The string distance is stored under. 
    
    sim_ley : str, default=`sim`
        The string that similarity is stored under. 
    
    tau : float, default=None
        The scaling factor. 
    
    """
    has_sim = sum(1 for _,_,d in G.edges(data=True) if sim_key in d)
    if has_sim >= 0.8 * G.number_of_edges():
        return sim_key

    lengths = [float(d.get(length_key, 0.0)) for _,_,d in G.edges(data=True)]
    if lengths:
        if tau is None:
            tau = float(np.median([L for L in lengths if np.isfinite(L) and L>0])) or 1.0
        inv_tau = 1.0 / tau
        for _,_,d in G.edges(data=True):
            L = float(d.get(length_key, 0.0))
            d[sim_key] = float(np.exp(-L * inv_tau))
    else:
        for _,_,d in G.edges(data=True):
            d[sim_key] = float(d.get("weight", 1.0))

    return sim_key
    
def _outward_cut_leakage(envelope_graph: nx.Graph,
                         S: Sequence,
                         weight_key:str = "sim") -> Dict:
    """
    Helper function to define the outward cut leakage from the observed
    solution set graph.
    
    Parameters
    ----------
    envelope_graph : nx.Graph
        The full latent graph
    
    S : Iterable
        The observed solution set graph nodes.

    weight_key : str, default=`weight`
        The key that similarity measurements are stored under. that the
        weights must be similarity measurments and not geodeic
        distances.
    
    Returns
    -------
    b : Dict
        The edge leakage dict. b[i] = sum of envelope weights from i in
        S to any neighbor outside S.
    """
    Sset = set(S)
    b = {}
    for u in S:
        leak = 0.0
        if u in envelope_graph:
            for v, d in envelope_graph[u].items():
                if v not in Sset:
                    leak += float(d.get(weight_key, 1.0))
        b[u] = leak
    return b

@dataclass
class BoundaryModel:
    """
    Parameters for the boundary/leakage model used in the Dirichlet operator. Data helper class.

    Attributes
    ----------
    kind : {'degree_deficit', 'envelope', 'constant', 'custom'}
        - 'degree_deficit': target degree per node is max(internal_degree, median(internal_degree)).
        - 'envelope': target degree derived from an "envelope" supergraph; leakage is the difference
          between envelope-degree and internal-degree on S.
        - 'constant': use a constant leakage value for all nodes in S (use `constant_value`).
        - 'custom': provide explicit leakage values via `b_leak_custom`.
    alpha : float, optional (default=1.0)
        Robin interpolation between Neumann (0.0) and Dirichlet (1.0) boundary.
        The leakage vector is scaled by `alpha` before constructing the operator.
    constant_value : float, optional
        Only used when `kind == 'constant'`. Non-negative constant leakage per node.
    """
    kind: Literal['degree_deficit', 'envelope', 'constant', 'custom'] = "envelope"
    alpha: float = 1.0
    constant_value: Optional[float] = None


def _nx_to_sparse_on_nodes(G: nx.Graph,
                           nodes: Sequence) -> Tuple[sp.csr_matrix, Dict]:
    """
    Convert a NetworkX graph to a CSR adjacency restricted to `nodes` (in given order).

    Parameters
    ----------
    G : nx.Graph
        Symmetric weighted graph with edge attribute 'weight' (defaults to 1.0 if absent).
    nodes : sequence
        Node list specifying the order to use.

    Returns
    -------
    W : scipy.sparse.csr_matrix
        Sparse weighted adjacency among `nodes` only. shape (n, n)
    index : dict
        Mapping node : row/col index in W.
    """
    idx = {n: i for i, n in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            w = float(d.get("weight", 1.0))
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])
    n = len(nodes)
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
    return W, idx


def _degree_vector(W: sp.csr_matrix) -> np.ndarray:
    """
    Helper function for row sums of a sparse matrix as a 1-D float array.

    Parameters
    ----------
    W : scipy.sparse.csr_matrix
        The sparse matrix.

    Returns
    -------
    np.ndarray
        The summed row array.
    """
    return np.asarray(W.sum(axis=1)).ravel()


def build_dirichlet_operator(G: nx.Graph,
                             S: Iterable,
                             boundary: BoundaryModel = BoundaryModel(),
                             normalized: bool = True) -> Tuple[sp.csr_matrix, List]:
    """
    Construct the Dirichlet operator on a subset `S` with a chosen boundary model.

    Parameters
    ----------
    G : nx.Graph
        Symmetric weighted graph on observed nodes.

    S : iterable
        Nodes defining the domain (e.g., a functional set S_tau). Order in the
        returned matrix follows `list(S)`.

    boundary : BoundaryModel, optional
        Boundary/leakage model.

    envelope_graph : nx.Graph, optional
        If `boundary.kind == 'envelope'`, degrees from this supergraph are used
        as target degrees for the leakage calculation.

    normalized : bool, optional (default=True)
        If True, return the normalized Dirichlet operator; otherwise, combinatorial.

    Returns
    -------
    L_D : scipy.sparse.csr_matrix
        Dirichlet operator on S.
    nodes_S : list
        Node ordering used for rows/cols of L_D.
    """
    nodes_S = list(S)
    if len(nodes_S) == 0:
        raise ValueError("S is empty. Provide at least one node.")

    W_SS, idx = _nx_to_sparse_on_nodes(G, nodes_S)
    d_in = _degree_vector(W_SS)

    # Compute leakage vector b according to boundary model
    if boundary.kind == "degree_deficit":
        target = np.maximum(d_in, np.median(d_in))  # robust baseline
        b = np.clip(target - d_in, 0.0, np.inf)
    elif boundary.kind == "constant":
        if boundary.constant_value is None or boundary.constant_value < 0:
            raise ValueError("Provide non-negative constant_value for 'constant' boundary.")
        b = np.full_like(d_in, fill_value=float(boundary.constant_value), dtype=float)
    elif boundary.kind == "custom":
        raise ValueError("For 'custom' leakage, use `build_dirichlet_operator_custom_leak`.")
    else:
        raise ValueError("Unknown boundary.kind. Use 'degree_deficit', 'envelope', 'constant', or 'custom'.")

    # Robin interpolation scale
    b = float(boundary.alpha) * b

    # Combinatorial Dirichlet operator pieces
    D_in = sp.diags(d_in, offsets=0, format="csr")
    L_comb = D_in - W_SS
    B = sp.diags(b, offsets=0, format="csr")
    L_D = L_comb + B

    if not normalized:
        return L_D.tocsr(), nodes_S

    # Normalized Dirichlet operator
    Deg = D_in + B
    diag = Deg.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        invsqrt = np.zeros_like(diag)
        mask = diag > 0
        invsqrt[mask] = 1.0 / np.sqrt(diag[mask])
    Dm12 = sp.diags(invsqrt, offsets=0, format="csr")
    L_Dn = Dm12 @ L_D @ Dm12
    return L_Dn.tocsr(), nodes_S


# Always use custom leak with envelope graph.
def build_dirichlet_operator_custom_leak(G: nx.Graph,
                                         S: Iterable,
                                         b_leak: Dict,
                                         normalized: bool = True) -> Tuple[sp.csr_matrix, List]:
    """
    Construct the Dirichlet operator on `S` with a custom leakage vector.

    Parameters
    ----------
    G : nx.Graph
        Symmetric weighted graph on observed nodes.

    S : iterable
        Nodes in the solution set.

    b_leak : dict
        Mapping node : non-negative leakage value b_i. Missing nodes default to 0.

    normalized : bool, default=`True`
        If True, return the normalized Dirichlet operator; otherwise, combinatorial.

    Returns
    -------
    L_D : scipy.sparse.csr_matrix
        Dirichlet operator on S.
    nodes_S : list
        Node ordering used for rows/cols of L_D.
    """
    nodes_S = list(S)
    W_SS, idx = _nx_to_sparse_on_nodes(G, nodes_S)
    d_in = _degree_vector(W_SS)

    b = np.array([float(max(0.0, b_leak.get(n, 0.0))) for n in nodes_S], dtype=float)

    D_in = sp.diags(d_in, offsets=0, format="csr")
    L_comb = D_in - W_SS
    B = sp.diags(b, offsets=0, format="csr")
    L_D = L_comb + B

    if not normalized:
        return L_D.tocsr(), nodes_S

    Deg = D_in + B
    diag = Deg.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        invsqrt = np.zeros_like(diag)
        mask = diag > 0
        invsqrt[mask] = 1.0 / np.sqrt(diag[mask])
    Dm12 = sp.diags(invsqrt, offsets=0, format="csr")
    L_Dn = Dm12 @ L_D @ Dm12
    return L_Dn.tocsr(), nodes_S


# Note the graph must be connected.
def first_dirichlet_eigenpair(L_D: np.ndarray,
                              k: int = 1,
                              which: str = "SM",
                              tol: float = 1e-6,
                              maxiter: int = 5000) -> Tuple[float, np.ndarray]:
    """
    Compute the smallest Dirichlet eigenvalue and eigenfunction.

    Parameters
    ----------
    L_D : scipy.sparse.csr_matrix
        Symmetric Dirichlet operator (combinatorial or normalized).
    k : int, optional, default=1
        Number of eigenpairs; only the smallest is returned.
    which : 'SM' or'SA', default='SM'
        Selection mode for ARPACK (smallest magnitude or algebraic).
    tol : float, default=1e-6
        Convergence tolerance.
    maxiter : int, default=5000
        Maximum iterations for the eigensolver.

    Returns
    -------
    lambda1 : float
        The smallest Dirichlet eigenvalue.
    f1 : ndarray
        The corresponding eigenfunction. Sign is chosen so that sum(f1) >= 0. Shape (n,)
    """
    try:
        vals, vecs = spla.eigsh(L_D, k=k, which=which, tol=tol, maxiter=maxiter)
    except Exception:
        # Shift slightly to ensure numerical stability
        n = L_D.shape[0]
        Ls = L_D + 1e-6 * sp.eye(n, format="csr")
        vals, vecs = spla.eigsh(Ls, k=k, which="SM", tol=tol, maxiter=maxiter)

    f1 = vecs[:, 0]
    if float(np.sum(f1)) < 0:
        f1 = -f1
    return float(vals[0]), f1

def rank_throat_edges(G: nx.Graph,
                      nodes_S: Sequence,
                      f: np.ndarray,
                      weight_key: str = "weight",
                      degree_normalize: bool = True) -> pd.DataFrame:
    """
    Rank edges in the induced subgraph on S by Dirichlet-eigenfunction gradient.

    Parameters
    ----------
    G : nx.Graph
        Symmetric weighted graph with edge weight attribute.
    nodes_S : sequence
        Node ordering corresponding to the entries of `f`.
    f : ndarray, shape (n,)
        First Dirichlet eigenfunction values on nodes_S.
    weight_attr : str, optional (default='weight')
        Edge attribute name to use as weight. Defaults to 1.0 if missing.
    degree_normalize : bool, optional (default=True)
        If True, divide the gradient score by sqrt(deg(u)+deg(v)) to temper hubs.

    Returns
    -------
    df_edges : pandas.DataFrame
        Columns: ['u','v','weight','f_u','f_v','grad','grad_norm'], sorted by 'grad_norm' desc.
    """
    idx = {n: i for i, n in enumerate(nodes_S)}
    rows = []
    for u, v, d in G.edges(nodes_S, data=True):
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            w = float(d.get(weight_key, 1.0))
            df = abs(f[i] - f[j]) * w
            if degree_normalize:
                du = max(G.degree(u), 1)
                dv = max(G.degree(v), 1)
                df_norm = df / math.sqrt(du + dv)
            else:
                df_norm = df
            rows.append((u, v, w, f[i], f[j], df, df_norm))

    df_edges = pd.DataFrame(
        rows, columns=["u", "v", "weight", "f_u", "f_v", "grad", "grad_norm"]
    ).sort_values("grad_norm", ascending=False, ignore_index=True)
    return df_edges

def local_cheeger_sweep(G: nx.Graph,
                        S: Iterable,
                        f: np.ndarray,
                        nodes_S: Sequence,
                        weight_key: str = "weight",
                        max_half_volume: bool = True,) -> Tuple[float, Set]:
    """
    Estimate the local Cheeger constant on S via a spectral sweep over f.

    Parameters
    ----------
    G : nx.Graph
        Symmetric weighted graph.
    S : iterable
        Nodes in the domain.
    f : ndarray, shape (n,)
        Dirichlet eigenfunction values corresponding to nodes_S.
    nodes_S : sequence
        Node ordering for f.
    weight_attr : str, optional (default='weight')
        Edge weight attribute name.
    max_half_volume : bool, optional (default=True)
        If True, only consider sets T whose volume <= 0.5 * vol(S) to conform
        to the usual Cheeger convention.

    Returns
    -------
    h_est : float
        Estimated local Cheeger constant (minimum sweep conductance).
    T_star : set
        The subset achieving the minimum (argmin) in the sweep.
    """
    S = list(S)
    Sset = set(S)
    idx = {n: i for i, n in enumerate(nodes_S)}
    f_on_S = np.array([f[idx[n]] for n in S], dtype=float)

    # Sort nodes by decreasing f
    order = [n for _, n in sorted(zip(-f_on_S, S))]

    # Degrees in the full graph, but only for nodes in S
    deg = {u: 0.0 for u in S}
    for u, v, d in G.edges(S, data=True):
        w = float(d.get(weight_key, 1.0))
        if u in Sset:
            deg[u] += w
        if v in Sset:
            deg[v] += w

    vol_S = sum(deg.values())
    best_phi = float("inf")
    T_star: Set = set()
    T: Set = set()
    vol_T = 0.0

    for x in order:
        T.add(x)
        vol_T += deg[x]

        if max_half_volume and vol_T > 0.5 * max(vol_S, 1e-12):
            break

        # Cut from T to V\T measured in full G (includes edges to outside S)
        cut = 0.0
        for u in T:
            for v, d in G[u].items():
                if v not in T:
                    cut += float(d.get(weight_key, 1.0))

        phi = cut / max(vol_T, 1e-12)
        if phi < best_phi:
            best_phi = phi
            T_star = set(T)

    return float(best_phi), T_star

def calculate_local_bottleneck(fitness_landscape: Union[nx.Graph, FitnessLandscape],
                               latent_graph: Union[nx.Graph, FitnessLandscape] = None,
                               weight_key: str = 'weight',
                               sim_key: str = 'sim',
                               tau: float = None,
                               normalized_laplacian: bool = True,
                               normalize_degree: bool = True,
                               return_latent_graph: bool = False,
                               **kwargs
                               ) -> Dict:
    """
    Function to compute how locally bottlenecked an observed
    (connected) fitnesslandscape is, assuming it is an induced subgraph
    of a larger unobserved latent graph that has been sampled by
    evolution. 

    Parameters
    ----------
    fitness_lanscape : FitnessLandscape or nx.Graph
        The observed fitness landscape. 
    
    latent_graph : nx.Graph, default=`None`
        The latent graph that the observed landscape has been induced
        from. If `None`, a minimum spanning latent graph will be 
        constructed using `reconstruct_latent_graph_with_steiner`.
    
    weight_key : str, default=`weight`
        The key that edge weight attributes are stored under. 
    
    sim_key : str, default=`sim`
        The key that edge similarity attributes are stored under.
    
    tau : float, default=`None`
        The weight-to-similarity `neglog` normalization factor. If
        `None`, the median weight attribute is used. 
    
    normalized_laplacian : bool, default=`True`
        Boolean to use the normalized Laplacian in Dirichlet operator
        construction. 
    
    normalize_degree : bool, default=`True`
        Boolean to normalize the Dirichlet operator gradient by
        sqrt(deg(u)+deg(v)) to temper hubs during throat ranking.
    
    return_latent_graph : bool, default=`False`,
        Boolean to return the latent graph.

    **kwargs
        Key word args passed to the
        `reconstruct_latent_graph_with_steiner` function.

    Returns
    -------
    results : Dict
        The dictionary of results.
    """
    
    # Typing for induced graph.
    if isinstance(fitness_landscape, FitnessLandscape):
        G_obs = fitness_landscape.graph
    else:
        G_obs = fitness_landscape
    if not isinstance(G_obs, nx.Graph):
        raise ValueError(f"Expected `nx.Graph` or `FitnessLandscape`, found {type(fitness_landscape)}")
    
    # Typing for latent graph
    # Constrcut the latent spanning graph.
    if latent_graph is None:
        latent_graph = reconstruct_latent_graph_with_steiner(G_obs, **kwargs)[0]
    elif isinstance(latent_graph, FitnessLandscape):
        latent_graph = latent_graph.graph
    
    if not isinstance(latent_graph, nx.Graph):
        raise ValueError(f"Expected `nx.Graph` or `FitnessLandscape`, found {type(latent_graph)}")
    
    S = list(G_obs.nodes())

    # Ensure edges capture similarity key.
    sim_attr_env = _ensure_affinity(G_obs,
                                    length_key=weight_key,
                                    sim_key=sim_key,
                                    tau=tau)

    # Define leakage boundary model from latent graph.
    b_leak = _outward_cut_leakage(latent_graph,
                                  S,
                                  weight_key=sim_attr_env)

    # Construct Robin Laplacian Dirichlet operator.
    L_D, nodes_S = build_dirichlet_operator_custom_leak(G=G_obs,
                                                        S=S,
                                                        b_leak=b_leak,
                                                        normalized=normalized_laplacian)

    # Rank eigenfunction bottelenecks.
    lam1, f1 = first_dirichlet_eigenpair(L_D)
    throats = rank_throat_edges(G=G_obs,
                                nodes_S=nodes_S,
                                f=f1,
                                weight_key=weight_key,
                                degree_normalize=normalize_degree)

    # Cheeger cutset sweep.
    h_est, T_star = local_cheeger_sweep(G=latent_graph,
                                        S=S,
                                        f=f1,
                                        nodes_S=nodes_S,
                                        weight_key=sim_attr_env)

    results = {
        "first_dirichlet_eigenvalue": lam1,
        "first_dirichlet_eigenvector": f1,
        "dirichlet_eigenfunction_throats": throats,
        "local_cheeger_constant": h_est,
        "local_cheeger_cutset": T_star
    }

    if return_latent_graph:
        results['latent_graph'] = latent_graph
    
    return results