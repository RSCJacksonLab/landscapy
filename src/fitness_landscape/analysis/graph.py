from networkx.algorithms.bipartite import matrix
import numpy as np
import networkx as nx
from ..core.landscape import FitnessLandscape
from ..transforms.eigenmode import eigenmode_decomposition
from typing import Union, Dict, Literal, Sequence, Optional, Iterable
import scipy.sparse as sp
from scipy.sparse.linalg import splu

def graph_properties(graph: Union[FitnessLandscape, nx.Graph]) -> Dict:
    """
    Calculate graph properties relevant to fitness landscapes.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to analyze.
        
    Returns
    -------
    dict
        Dictionary of graph properties.
    """
    
    properties = ['degree', 'clustering', 'path_length', 'components', 'density']
    
    results = {}
    
    for prop in properties:
        if prop == 'degree':
            # Calculate degree statistics
            degrees = [d for _, d in graph.degree()]
            results['degree'] = {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': np.min(degrees),
                'max': np.max(degrees)
            }
        
        elif prop == 'clustering':
            # Calculate clustering coefficient
            results['clustering'] = nx.average_clustering(graph)
        
        elif prop == 'path_length':
            # Calculate average shortest path length
            if nx.is_connected(graph):
                results['path_length'] = nx.average_shortest_path_length(graph)
            else:
                # Calculate for largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                results['path_length'] = nx.average_shortest_path_length(subgraph)
                results['path_length_note'] = 'Calculated for largest connected component'
        
        elif prop == 'components':
            # Calculate connected components
            components = list(nx.connected_components(graph))
            results['components'] = {
                'count': len(components),
                'largest_size': len(max(components, key=len)),
                'sizes': [len(c) for c in components]
            }
        
        elif prop == 'density':
            # Calculate graph density
            results['density'] = nx.density(graph)
        
        else:
            raise ValueError(f"Unsupported property: {prop}")
    
    return results

def calculate_ruggedness_local_optima(landscape: FitnessLandscape,
                                      **kwargs) -> Dict:
    """
    Function to measure ruggedness as the number of local fitness
    optima / maxima. 

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    Returns
    -------
    Dict
        The results dictionary.
    """
    # Extract sequences
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    # Find local optima
    local_optima = []
    
    for i, seq in enumerate(sequences):
        # Get fitness of current sequence
        fitness = landscape.get_fitness(seq)
        
        # Get neighbors
        neighbors = list(landscape.graph.neighbors(i))
        
        # Check if fitness is higher than all neighbors
        is_local_optimum = True
        for neighbor in neighbors:
            neighbor_fitness = landscape.get_fitness(sequences[neighbor])
            if neighbor_fitness > fitness:
                is_local_optimum = False
                break
        
        if is_local_optimum:
            local_optima.append(i)
    
    # Calculate density of local optima
    density = len(local_optima) / len(sequences)
    
    # Calculate fitness statistics of local optima
    local_optima_fitness = [landscape.get_fitness(sequences[i]) for i in local_optima]
    
    if local_optima_fitness:
        mean_fitness = np.mean(local_optima_fitness)
        std_fitness = np.std(local_optima_fitness)
        max_fitness = np.max(local_optima_fitness)
        min_fitness = np.min(local_optima_fitness)
    else:
        mean_fitness = std_fitness = max_fitness = min_fitness = None
    
    return {
        'local_optima_count': len(local_optima),
        'local_optima_density': density,
        'local_optima_indices': local_optima,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'max_fitness': max_fitness,
        'min_fitness': min_fitness,
        'method': 'local_optima'
    }
    
def graph_spectral_analysis(landscape: FitnessLandscape,
                            matrix: Literal['laplacian', 'norm_laplacian'] = 'laplacian',
                            k: int = None) -> Dict:
    """
    Analyze the eigenmodes of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to analyze.
    k : int or None, optional
        Number of eigenmodes to analyze.
    matrix : str, default=`laplacian`
        The matrix to use for spectral analysis. Options are
        `laplacian` or `norm_laplcian`.
        
    Returns
    -------
    dict
        Eigenspectral analysis results. 
    """
    eigenvalues, eigenvectors = eigenmode_decomposition(landscape, matrix=matrix, k=k)
    
    w = np.asarray(eigenvalues, dtype=float)
    U = np.asarray(eigenvectors, dtype=float)
    n, m = U.shape

    pr = np.empty(m, dtype=float)
    ipr = np.empty(m, dtype=float)
    node_c = np.abs(U)

    for i in range(m):
        psi2 = U[:, i] ** 2
        pr[i] = (psi2.sum() ** 2) / (psi2 ** 2).sum()
        ipr[i] = 1.0 / pr[i]

    out = {
        'eigenvalues': w,
        'participation_ratios': pr,
        'localization': ipr,
        'node_centralities': node_c,
    }
    if m >= 2:
        # ascending-ordered eigenvalues => spectral gap between first two
        out['spectral_gap'] = float(w[1] - w[0])
    # Simple spectral density
    hist, edges = np.histogram(w, bins=min(20, m))
    out['spectral_density'] = {'histogram': hist, 'bin_edges': edges}
    return out


def resistance_distance_matrix(graph: Union[FitnessLandscape, nx.Graph],
                               nodes: Optional[Sequence] = None,
                               *,
                               weight_key: Optional[str] = None,
                               jitter: float = 1e-10,
                               sparse_threshold: int = 1000,
                               weight_epsilon: float = 1e-8,
                               weight_normalisation: bool = True) -> np.ndarray:
    """
    Compute the pairwise effective resistance distances among a subset
    of nodes in a weighted graph.

    Parameters
    ----------
    graph : FitnessLandscape or networkx.Graph
        Source graph. If a :class:`FitnessLandscape` is provided, its
        underlying graph is used.
    nodes : Sequence, optional
        Optional ordered sequence of nodes to include. Defaults to all
        nodes present in the graph.
    weight_key : str, optional
        Edge attribute representing conductance/weight. When ``None``,
        edges are treated as unweighted.
    jitter : float, default=1e-10
        Diagonal regularisation added when the Laplacian is not full
        rank to ensure a stable pseudoinverse.
    weight_epsilon : float, default=1e-8
        Small positive value added to every edge weight (via an
        unweighted Laplacian) before factorisation to prevent the sparse
        solver from encountering zero-weight conductances. This does
        not modify the underlying graph; it only affects the temporary
        Laplacian used for resistance calculations.
    weight_normalisation : bool, default=True
        If ``True``, rescales the temporary Laplacian so its largest
        absolute entry is 1.0, improving numerical stability. The final
        resistance distances are rescaled back so results remain in the
        original units.

    Returns
    -------
    np.ndarray
        Symmetric matrix ``R`` where ``R[i, j]`` is the effective
        resistance between ``nodes[i]`` and ``nodes[j]``.
    """
    G = graph.graph if isinstance(graph, FitnessLandscape) else graph
    if G is None:
        raise ValueError("Graph is required to compute resistance distances.")

    if nodes is None:
        node_order: Iterable = list(G.nodes())
    else:
        node_order = list(nodes)

    if not node_order:
        return np.zeros((0, 0), dtype=float)

    sub = G.subgraph(node_order)
    L_sparse = nx.laplacian_matrix(sub, nodelist=list(node_order), weight=weight_key).astype(float)
    if weight_epsilon:
        L_unweighted = nx.laplacian_matrix(sub, nodelist=list(node_order), weight=None).astype(float)
        L_sparse = L_sparse + weight_epsilon * L_unweighted

    norm_factor = 1.0
    if weight_normalisation and L_sparse.nnz > 0:
        max_entry = float(np.max(np.abs(L_sparse.data)))
        if max_entry > 0:
            norm_factor = max_entry
            L_sparse = L_sparse / norm_factor
    n = L_sparse.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    if n <= sparse_threshold:
        L = L_sparse.toarray()
        if np.linalg.matrix_rank(L) < n - 1:
            L = L + jitter * np.eye(n)
        try:
            L_pinv = np.linalg.pinv(L)
        except np.linalg.LinAlgError:
            L = L + jitter * np.eye(n)
            L_pinv = np.linalg.pinv(L)
        diag = np.diag(L_pinv)
        R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
        R[R < 0] = 0.0
        return R / norm_factor

    # Sparse path using grounded Laplacian solves
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    ground = n - 1
    keep = list(range(n - 1))
    L_reduced = L_sparse[keep, :][:, keep].tocsc()
    if jitter:
        L_reduced = L_reduced + jitter * sp.eye(n - 1, format="csc")
    try:
        solver = splu(L_reduced)
    except RuntimeError:
        L = L_sparse.toarray()
        attempts = 0
        rank = n
        while attempts < 5:
            try:
                rank = np.linalg.matrix_rank(L)
                break
            except np.linalg.LinAlgError:
                L = L + (10 ** attempts) * jitter * np.eye(n)
                attempts += 1
        if rank < n - 1:
            L = L + jitter * np.eye(n)
        attempts = 0
        while attempts < 5:
            try:
                L_pinv = np.linalg.pinv(L)
                break
            except np.linalg.LinAlgError:
                L = L + (10 ** attempts) * jitter * np.eye(n)
                attempts += 1
        else:
            raise
        diag = np.diag(L_pinv)
        R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
        R[R < 0] = 0.0
        return R / norm_factor

    Z = np.zeros((n - 1, n - 1), dtype=float)
    rhs = np.zeros(n - 1, dtype=float)
    for idx in range(n - 1):
        rhs[idx] = 1.0
        Z[:, idx] = solver.solve(rhs)
        rhs[idx] = 0.0

    diag = np.diag(Z)
    R_reduced = diag[:, None] + diag[None, :] - 2.0 * Z
    R_reduced[R_reduced < 0] = 0.0

    R = np.zeros((n, n), dtype=float)
    R[: n - 1, : n - 1] = R_reduced
    R[: n - 1, ground] = diag
    R[ground, : n - 1] = diag
    return R / norm_factor
