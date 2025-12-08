from networkx.algorithms.bipartite import matrix
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
import numpy as np
import networkx as nx
from functools import lru_cache
from ..core.annotation import AnnotationLayer
from ..core.fitness import CategoricalFitness, ProbabilisticCategoricalFitness
from ..core.landscape import FitnessLandscape
from ..transforms.eigenmode import eigenmode_decomposition
from typing import Any, Mapping, Union, Dict, Literal, Sequence, Optional, Iterable, Hashable
import scipy.sparse as sp
from scipy.optimize import linprog
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


def annotate_louvain_communities(
    landscape: FitnessLandscape,
    *,
    annotation_name: str = "louvain_communities",
    weight: Optional[str] = "weight",
    resolution: float = 1.0,
    threshold: float = 1e-7,
    seed: Optional[Union[int, np.random.Generator, np.random.RandomState]] = None,
    overwrite: bool = False,
) -> AnnotationLayer:
    """
    Detect graph communities with the Louvain algorithm and attach them as annotations.

    Parameters
    ----------
    landscape :
        Target landscape with an instantiated graph.
    annotation_name :
        Name for the annotation layer that will store community metadata.
    weight :
        Edge attribute representing weights. ``None`` treats the graph as unweighted.
    resolution :
        Resolution parameter passed to :func:`networkx.algorithms.community.louvain_communities`.
    threshold :
        Convergence threshold for the Louvain optimiser.
    seed :
        Optional seed controlling the stochastic parts of the algorithm.
    overwrite :
        Replace an existing annotation layer with ``annotation_name`` if present.

    Returns
    -------
    AnnotationLayer
        Newly attached annotation layer describing community assignments.
    """
    graph = landscape.graph
    if graph is None:
        raise ValueError("Cannot compute communities: landscape graph is missing.")
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot compute communities: landscape graph has no nodes.")

    if annotation_name in landscape.annotation_layers:
        if not overwrite:
            raise ValueError(
                f"Annotation layer '{annotation_name}' already exists; "
                "set overwrite=True to recompute."
            )
        landscape.detach_annotation(annotation_name)

    communities = list(
        louvain_communities(graph, weight=weight, resolution=resolution, threshold=threshold, seed=seed)
    )
    if not communities:
        raise RuntimeError("Louvain community detection returned no communities.")

    node_to_comm: Dict[Hashable, int] = {}
    comm_sizes: Dict[int, int] = {}
    for cid, members in enumerate(communities):
        comm_sizes[cid] = len(members)
        for node in members:
            node_to_comm[node] = cid

    try:
        modularity_score = float(modularity(graph, communities, weight=weight))
    except Exception:
        modularity_score = None

    node_index_map = getattr(landscape, "_nodes_by_index", None)
    if node_index_map is None:
        raise RuntimeError("Landscape is missing node index mapping; cannot align annotations.")

    n = len(landscape.sequences)
    community_ids: list[Optional[int]] = [None] * n
    community_labels: list[Optional[str]] = [None] * n
    community_sizes: list[Optional[int]] = [None] * n
    louvain_community: list[Optional[str]] = [None] * n

    for idx in range(n):
        node = node_index_map.get(idx)
        if node is None:
            continue
        cid = node_to_comm.get(node)
        if cid is None:
            continue
        community_ids[idx] = int(cid)
        label = f"community_{cid}"
        community_labels[idx] = label
        community_sizes[idx] = comm_sizes.get(cid)
        louvain_community[idx] = label

    metadata_seed: Optional[Union[int, str]] = None
    if isinstance(seed, (int, np.integer)):
        metadata_seed = int(seed)
    elif seed is not None:
        metadata_seed = seed.__class__.__name__

    metadata = {
        "algorithm": "louvain",
        "weight_key": weight,
        "resolution": resolution,
        "threshold": threshold,
        "seed": metadata_seed,
        "community_count": len(communities),
        "modularity": modularity_score,
    }

    return landscape.attach_annotation(
        name=annotation_name,
        data={
            "community_id": community_ids,
            "community_label": community_labels,
            "community_size": community_sizes,
            "louvain_community": louvain_community,
        },
        metadata=metadata,
        map_by="index",
    )

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


def _canonical_category_label(value: Any) -> Any:
    """
    Convert potentially unhashable annotation values into stable labels.
    """
    if value is None:
        return None
    try:
        hash(value)
        return value
    except TypeError:
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(value)
    return str(value)


@lru_cache(maxsize=8)
def _transport_constraint_matrix(n: int) -> sp.csr_matrix:
    """
    Build a sparse constraint matrix enforcing coupling marginals for OT.
    Cached by n to reuse across layer pairs.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for i in range(n):
        base = i * n
        rows.extend([i] * n)
        cols.extend(range(base, base + n))
        data.extend([1.0] * n)

    offset = n
    for j in range(n):
        for i in range(n):
            rows.append(offset + j)
            cols.append(i * n + j)
            data.append(1.0)

    return sp.csr_matrix((data, (rows, cols)), shape=(2 * n, n * n))


def _wasserstein_distance(cost: np.ndarray,
                          mu_a: np.ndarray,
                          mu_b: np.ndarray,
                          constraint: Optional[sp.csr_matrix] = None) -> float:
    """
    Solve the optimal transport problem between distributions mu_a and mu_b
    with cost matrix `cost`.
    """
    n = cost.shape[0]
    if cost.shape != (n, n):
        raise ValueError("Cost matrix must be square.")
    if mu_a.shape[0] != n or mu_b.shape[0] != n:
        raise ValueError("Distribution lengths must match cost matrix.")

    constraint_mat = constraint if constraint is not None else _transport_constraint_matrix(n)
    b_eq = np.concatenate([mu_a, mu_b]).astype(float, copy=False)
    res = linprog(
        cost.reshape(-1),
        A_eq=constraint_mat,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Optimal transport solver failed: {res.message}")
    return float(res.fun)


def _bilinear_form_with_solver(
    p: np.ndarray,
    q: np.ndarray,
    solver_data: dict[str, Any] | None,
    laplacian_pinv: np.ndarray | None,
) -> float:
    """
    Compute p^T L^{+} q without materialising the full resistance matrix.
    """
    if laplacian_pinv is not None:
        p0 = p - p.mean()
        q0 = q - q.mean()
        return float(p0 @ laplacian_pinv @ q0)

    if not solver_data:
        raise ValueError("Solver data is required when Laplacian pseudoinverse is unavailable.")

    solver = solver_data["solver"]
    ground = solver_data["ground"]
    p0 = p - p.mean()
    q0 = q - q.mean()
    p_red = np.delete(p0, ground)
    q_red = np.delete(q0, ground)
    y = solver.solve(q_red)
    return float(np.dot(p_red, y))


def _aggregate_distance_matrix(cost: np.ndarray,
                               probabilities: np.ndarray,
                               agg: Literal["wasserstein", "ot", "expected_pairwise"],
                               *,
                               diag_scaled: np.ndarray | None = None,
                               norm_factor: float = 1.0,
                               solver_data: dict[str, Any] | None = None,
                               laplacian_pinv: np.ndarray | None = None) -> np.ndarray:
    """
    Aggregate node-level distances into class-level distances using the
    selected aggregation strategy.
    """
    agg_mode = "wasserstein" if agg == "ot" else agg
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2-D array (nodes x categories).")
    n_nodes, n_classes = probs.shape
    dist = np.full((n_classes, n_classes), np.nan, dtype=float)
    if n_classes == 0:
        return dist

    masses = probs.sum(axis=0)
    if agg_mode == "expected_pairwise":
        if diag_scaled is None:
            numer = probs.T @ cost @ probs
            denom = masses[:, None] * masses[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                dist = numer / denom
            dist[denom == 0] = np.nan
            for i in range(n_classes):
                if masses[i] > 0:
                    dist[i, i] = 0.0
            return dist

        diag_true = np.asarray(diag_scaled, dtype=float) / norm_factor
        accum = probs.T @ diag_true
        for a in range(n_classes):
            if masses[a] <= 0:
                continue
            for b in range(a, n_classes):
                if masses[b] <= 0:
                    continue
                if a == b:
                    dist[a, b] = 0.0
                    continue
                cross_scaled = _bilinear_form_with_solver(
                    probs[:, a], probs[:, b], solver_data, laplacian_pinv
                )
                cross_true = cross_scaled / norm_factor
                val = (masses[b] * accum[a] + masses[a] * accum[b] - 2.0 * cross_true) / (
                    masses[a] * masses[b]
                )
                dist[a, b] = dist[b, a] = val
        return dist

    # Wasserstein / OT aggregation
    constraint = _transport_constraint_matrix(n_nodes)
    for a in range(n_classes):
        if masses[a] <= 0:
            continue
        mu_a = probs[:, a] / masses[a]
        for b in range(a, n_classes):
            if masses[b] <= 0:
                continue
            if a == b:
                dist[a, b] = 0.0
                continue
            mu_b = probs[:, b] / masses[b]
            d = _wasserstein_distance(cost, mu_a, mu_b, constraint=constraint)
            dist[a, b] = dist[b, a] = d
    return dist


def _probability_matrix_from_fitness_layer(landscape: FitnessLandscape,
                                           node_order: Sequence,
                                           layer_name: str) -> tuple[list[Any], np.ndarray]:
    layer = landscape.fitness_layers[layer_name]
    if not isinstance(layer, (CategoricalFitness, ProbabilisticCategoricalFitness)):
        raise TypeError(
            f"Layer '{layer_name}' is not categorical; only categorical layers can be aggregated."
        )

    categories = list(layer.categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    P = np.zeros((len(node_order), len(categories)), dtype=float)
    attr_key = f"fitness_{layer_name}"

    for row, node in enumerate(node_order):
        node_data = landscape.graph.nodes[node]
        raw = node_data.get(attr_key)
        if raw is None:
            continue
        if isinstance(raw, Mapping):
            for cat, prob in raw.items():
                idx = cat_to_idx.get(cat)
                if idx is None:
                    raise KeyError(f"Unknown category '{cat}' encountered in layer '{layer_name}'.")
                P[row, idx] = float(prob)
        elif isinstance(raw, (np.ndarray, list, tuple)):
            arr = np.asarray(raw, dtype=float).ravel()
            if arr.size != len(categories):
                raise ValueError(
                    f"Fitness layer '{layer_name}' provided {arr.size} probs, "
                    f"expected {len(categories)}."
                )
            P[row, :] = arr
        else:
            idx = cat_to_idx.get(raw)
            if idx is None:
                raise KeyError(f"Unknown category '{raw}' encountered in layer '{layer_name}'.")
            P[row, idx] = 1.0
    return categories, P


def _probability_matrices_from_annotation_layer(landscape: FitnessLandscape,
                                                node_order: Sequence,
                                                layer_name: str) -> dict[str, tuple[list[Any], np.ndarray]]:
    layer = landscape.annotation_layers[layer_name]
    columns = list(layer.columns)
    records: list[dict[str, Any]] = []
    category_sets: dict[str, set[Any]] = {col: set() for col in columns}

    for node in node_order:
        node_data = landscape.graph.nodes[node]
        record = node_data.get("annotations", {}).get(layer_name, {}) or {}
        if not isinstance(record, Mapping):
            record = {}
        records.append(record)
        for col in columns:
            val = record.get(col)
            if val is None:
                continue
            if isinstance(val, Mapping):
                for cat in val.keys():
                    label = _canonical_category_label(cat)
                    if label is not None:
                        category_sets[col].add(label)
            else:
                label = _canonical_category_label(val)
                if label is not None:
                    category_sets[col].add(label)

    matrices: dict[str, tuple[list[Any], np.ndarray]] = {}
    for col in columns:
        cats = sorted(category_sets[col], key=lambda x: str(x))
        if not cats:
            continue
        cat_to_idx = {cat: i for i, cat in enumerate(cats)}
        P = np.zeros((len(node_order), len(cats)), dtype=float)
        for row, record in enumerate(records):
            val = record.get(col)
            if val is None:
                continue
            if isinstance(val, Mapping):
                for cat, prob in val.items():
                    label = _canonical_category_label(cat)
                    if label is None:
                        continue
                    idx = cat_to_idx.get(label)
                    if idx is None:
                        raise KeyError(
                            f"Unknown category '{label}' encountered in annotation '{layer_name}:{col}'."
                        )
                    P[row, idx] = float(prob)
            elif isinstance(val, (np.ndarray, list, tuple)):
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size != len(cats):
                    raise ValueError(
                        f"Annotation '{layer_name}:{col}' provided {arr.size} probs, "
                        f"expected {len(cats)}."
                    )
                P[row, :] = arr
            else:
                label = _canonical_category_label(val)
                if label is None:
                    continue
                idx = cat_to_idx.get(label)
                if idx is None:
                    raise KeyError(
                        f"Unknown category '{label}' encountered in annotation '{layer_name}:{col}'."
                    )
                P[row, idx] = 1.0
        matrices[col] = (cats, P)
    return matrices


def _compute_resistance_matrix(G: nx.Graph,
                               node_order: Sequence,
                               *,
                               weight_key: Optional[str],
                               jitter: float,
                               sparse_threshold: int,
                               weight_epsilon: float,
                               weight_normalisation: bool) -> tuple[np.ndarray, np.ndarray, float, Optional[dict[str, Any]], Optional[np.ndarray]]:
    """
    Core resistance distance computation for a provided node ordering.
    """
    if not node_order:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float), 1.0, None, None

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
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float), norm_factor, None, None

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
        return R / norm_factor, diag, norm_factor, None, L_pinv

    if n <= 1:
        return np.zeros((n, n), dtype=float), np.zeros((n,), dtype=float), norm_factor, None, None

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
        return R / norm_factor, diag, norm_factor, None, L_pinv

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
    diag_full = np.zeros(n, dtype=float)
    diag_full[: n - 1] = diag
    solver_data = {"solver": solver, "ground": ground}
    return R / norm_factor, diag_full, norm_factor, solver_data, None


def resistance_distance_matrix(graph: Union[FitnessLandscape, nx.Graph],
                               nodes: Optional[Sequence] = None,
                               *,
                               weight_key: Optional[str] = None,
                               jitter: float = 1e-10,
                               sparse_threshold: int = 1000,
                               weight_epsilon: float = 1e-8,
                               weight_normalisation: bool = True,
                               layers: Optional[Union[str, Sequence[str]]] = None,
                               aggregation_function: Literal["wasserstein", "ot", "expected_pairwise"] = "wasserstein") -> Dict[str, Any]:
    """
    Compute pairwise effective resistance distances among nodes and,
    optionally, aggregate them by categorical fitness or annotation
    layers.

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
    layers : str or Sequence[str], optional
        Fitness or annotation layer names to aggregate. ``None`` (default)
        aggregates all categorical fitness layers and all annotation
        layers when a :class:`FitnessLandscape` is provided. Ignored for
        plain networkx graphs.
    aggregation_function : {"wasserstein", "ot", "expected_pairwise"}, default="wasserstein"
        Aggregation strategy. ``"expected_pairwise"`` computes the
        expected resistance between two independently sampled nodes from
        each class. ``"wasserstein"``/``"ot"`` solve an optimal transport
        problem over the resistance matrix.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
            - ``"resistance_mat"`` : full node-by-node resistance matrix.
            - One entry per aggregated layer with ``"categories"`` (index
              to label mapping) and ``"distance_max"``/``"distance_mat"``
              holding the aggregated class-by-class distances.
    """
    G = graph.graph if isinstance(graph, FitnessLandscape) else graph
    if G is None:
        raise ValueError("Graph is required to compute resistance distances.")

    node_order = list(G.nodes()) if nodes is None else list(nodes)
    R, diag_scaled, norm_factor, solver_data, laplacian_pinv = _compute_resistance_matrix(
        G,
        node_order,
        weight_key=weight_key,
        jitter=jitter,
        sparse_threshold=sparse_threshold,
        weight_epsilon=weight_epsilon,
        weight_normalisation=weight_normalisation,
    )
    result: Dict[str, Any] = {"resistance_mat": R}

    agg_mode = aggregation_function.lower()
    if agg_mode not in {"wasserstein", "ot", "expected_pairwise"}:
        raise ValueError(
            f"aggregation_function must be 'wasserstein', 'ot', or 'expected_pairwise', got {aggregation_function!r}"
        )

    if not isinstance(graph, FitnessLandscape):
        if layers is not None:
            raise ValueError("Layer aggregation requires a FitnessLandscape input.")
        return result

    landscape = graph

    if layers is None:
        targets: list[tuple[str, str]] = []
        for name, layer in landscape.fitness_layers.items():
            if isinstance(layer, (CategoricalFitness, ProbabilisticCategoricalFitness)):
                targets.append(("fitness", name))
        for name in landscape.annotation_layers.keys():
            targets.append(("annotation", name))
    else:
        requested = [layers] if isinstance(layers, str) else list(layers)
        targets = []
        for name in requested:
            found = False
            if name in landscape.fitness_layers:
                layer = landscape.fitness_layers[name]
                if not isinstance(layer, (CategoricalFitness, ProbabilisticCategoricalFitness)):
                    raise TypeError(
                        f"Fitness layer '{name}' is not categorical and cannot be aggregated."
                    )
                targets.append(("fitness", name))
                found = True
            if name in landscape.annotation_layers:
                if found:
                    raise ValueError(
                        f"Layer '{name}' is present in both fitness and annotation collections."
                    )
                targets.append(("annotation", name))
                found = True
            if not found:
                raise KeyError(
                    f"Requested layer '{name}' not found among fitness or annotation layers."
                )

    if not targets:
        return result

    for layer_type, layer_name in targets:
        if layer_type == "fitness":
            categories, P = _probability_matrix_from_fitness_layer(landscape, node_order, layer_name)
            if P.size == 0 or len(categories) == 0:
                continue
            dist = _aggregate_distance_matrix(
                R,
                P,
                agg_mode,
                diag_scaled=diag_scaled,
                norm_factor=norm_factor,
                solver_data=solver_data,
                laplacian_pinv=laplacian_pinv,
            )
            result[layer_name] = {
                "categories": {i: cat for i, cat in enumerate(categories)},
                "distance_max": dist,
                "distance_mat": dist,
            }
        else:
            matrices = _probability_matrices_from_annotation_layer(landscape, node_order, layer_name)
            if not matrices:
                continue
            column_entries: dict[str, dict[str, Any]] = {}
            for col, (cats, P) in matrices.items():
                if P.size == 0 or len(cats) == 0:
                    continue
                dist = _aggregate_distance_matrix(
                    R,
                    P,
                    agg_mode,
                    diag_scaled=diag_scaled,
                    norm_factor=norm_factor,
                    solver_data=solver_data,
                    laplacian_pinv=laplacian_pinv,
                )
                column_entries[col] = {
                    "categories": {i: cat for i, cat in enumerate(cats)},
                    "distance_max": dist,
                    "distance_mat": dist,
                }
            if not column_entries:
                continue
            if len(column_entries) == 1:
                col_name, entry = next(iter(column_entries.items()))
                merged = dict(entry)
                merged["column"] = col_name
                result[layer_name] = merged
            else:
                result[layer_name] = {"columns": column_entries}

    return result
