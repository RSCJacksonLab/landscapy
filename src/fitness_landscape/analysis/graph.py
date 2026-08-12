from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
import numpy as np
import networkx as nx
from functools import lru_cache
from ..core.annotation import AnnotationLayer
from ..core.fitness import CategoricalFitness, ProbabilisticCategoricalFitness
from ..core.landscape import FitnessLandscape
from ..core.edge_schema import AUTO_EDGE_KEY, resolve_edge_attribute
from ..transforms.eigenmode import eigenmode_decomposition
from typing import Any, Mapping, Union, Dict, Literal, Sequence, Optional, Iterable, Hashable
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
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
        Dictionary of graph properties. Empty graphs return zero component
        count and density, with ``numpy.nan`` for undefined degree,
        clustering, and path-length statistics. Singleton graphs return zero
        degree, clustering, path length, and density.
    """
    
    graph = graph.graph if isinstance(graph, FitnessLandscape) else graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph or FitnessLandscape")
    if graph.number_of_nodes() == 0:
        return {
            "degree": {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            },
            "clustering": np.nan,
            "path_length": np.nan,
            "path_length_note": "Undefined for an empty graph",
            "components": {"count": 0, "largest_size": 0, "sizes": []},
            "density": 0.0,
        }

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
    weight: Optional[str] = AUTO_EDGE_KEY,
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
    weight : str or None, default="auto"
        Conductance attribute. ``"auto"`` resolves constructor metadata;
        ``None`` treats the graph as unweighted.
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
    resolved_weight = resolve_edge_attribute(
        graph,
        "conductance",
        weight,
        required=False,
    )

    if annotation_name in landscape.annotation_layers:
        if not overwrite:
            raise ValueError(
                f"Annotation layer '{annotation_name}' already exists; "
                "set overwrite=True to recompute."
            )
        landscape.detach_annotation(annotation_name)

    communities = list(
        louvain_communities(
            graph,
            weight=resolved_weight,
            resolution=resolution,
            threshold=threshold,
            seed=seed,
        )
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
        modularity_score = float(
            modularity(graph, communities, weight=resolved_weight)
        )
    except Exception:
        modularity_score = None

    node_index_map = landscape.sequence_index_to_node

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
        "weight_key": resolved_weight,
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
    **kwargs
        Reserved for compatibility. No keyword is currently consumed.
    
    Returns
    -------
    Dict
        The results dictionary.
    """
    sequences = landscape.sequences
    
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    assert landscape.graph is not None, \
    'Landscape graph must be initialised.'
    
    signal = landscape.get_signal()
    local_optima = []
    for node in landscape.graph.nodes():
        sequence_index = landscape.sequence_index_for_node(node)
        fitness = signal[sequence_index]
        if all(
            signal[landscape.sequence_index_for_node(neighbor)] <= fitness
            for neighbor in landscape.graph.neighbors(node)
        ):
            local_optima.append(node)
    
    # Calculate density of local optima
    density = len(local_optima) / len(sequences)
    
    # Calculate fitness statistics of local optima
    local_optima_indices = [
        landscape.sequence_index_for_node(node) for node in local_optima
    ]
    local_optima_fitness = [signal[index] for index in local_optima_indices]
    
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
        'local_optima': local_optima,
        'local_optima_indices': local_optima_indices,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'max_fitness': max_fitness,
        'min_fitness': min_fitness,
        'method': 'local_optima'
    }
    
def graph_spectral_analysis(landscape: FitnessLandscape,
                            matrix: Literal['laplacian', 'norm_laplacian'] = 'laplacian',
                            k: int = None,
                            weight_key: str | None = AUTO_EDGE_KEY) -> Dict:
    """
    Analyze the eigenmodes of a graph.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape whose graph topology is analysed.
    matrix : str, default=`laplacian`
        The matrix to use for spectral analysis. Options are
        `laplacian` or `norm_laplcian`.
    k : int or None, optional
        Number of eigenmodes to analyze.
    weight_key : str or None, default="auto"
        Conductance attribute used for the spectral operator. ``None``
        requests an unweighted operator.
        
    Returns
    -------
    dict
        Eigenspectral analysis results. Empty graphs return consistently empty
        arrays and omit ``spectral_gap``. Singleton graphs return one zero
        eigenvalue and also omit ``spectral_gap``.
    """
    if landscape.graph.number_of_nodes() == 0:
        return {
            "eigenvalues": np.empty(0, dtype=float),
            "participation_ratios": np.empty(0, dtype=float),
            "localization": np.empty(0, dtype=float),
            "node_centralities": np.empty((0, 0), dtype=float),
            "node_order": [],
            "weight_key": resolve_edge_attribute(
                landscape.graph,
                "conductance",
                weight_key,
                required=False,
            ),
            "spectral_density": {
                "histogram": np.empty(0, dtype=int),
                "bin_edges": np.empty(0, dtype=float),
            },
        }

    eigenvalues, eigenvectors = eigenmode_decomposition(
        landscape,
        matrix=matrix,
        k=k,
        weight_key=weight_key,
    )
    
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
        'node_order': list(landscape.graph.nodes()),
        'weight_key': resolve_edge_attribute(
            landscape.graph,
            "conductance",
            weight_key,
            required=False,
        ),
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

    if np.any(np.isnan(cost)):
        raise ValueError("Cost matrix must not contain NaN values.")
    if np.any(np.isinf(cost)):
        finite_adjacency = sp.csr_matrix(np.isfinite(cost), dtype=int)
        _, component_ids = sp.csgraph.connected_components(
            finite_adjacency,
            directed=False,
        )
        total_cost = 0.0
        for component_id in np.unique(component_ids):
            indices = np.flatnonzero(component_ids == component_id)
            mass_a = float(mu_a[indices].sum())
            mass_b = float(mu_b[indices].sum())
            if not np.isclose(mass_a, mass_b, rtol=1e-10, atol=1e-12):
                return np.inf
            if mass_a <= 0.0:
                continue
            component_cost = cost[np.ix_(indices, indices)]
            if np.any(~np.isfinite(component_cost)):
                raise ValueError(
                    "Finite-cost components must contain only finite costs."
                )
            total_cost += _wasserstein_distance(
                component_cost,
                mu_a[indices],
                mu_b[indices],
            )
        return float(total_cost)

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


def _aggregate_distance_matrix(cost: np.ndarray | None,
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
            if cost is None:
                raise ValueError("Cost matrix required for expected_pairwise when diag_scaled is missing.")
            if np.any(np.isnan(cost)):
                raise ValueError("Cost matrix must not contain NaN values.")
            if np.any(np.isinf(cost)):
                for a in range(n_classes):
                    if masses[a] <= 0:
                        continue
                    for b in range(a, n_classes):
                        if masses[b] <= 0:
                            continue
                        if a == b:
                            dist[a, b] = 0.0
                            continue
                        pair_mass = np.outer(probs[:, a], probs[:, b])
                        if np.any((pair_mass > 0.0) & np.isinf(cost)):
                            value = np.inf
                        else:
                            finite_cost = np.where(np.isfinite(cost), cost, 0.0)
                            value = float(np.sum(pair_mass * finite_cost)) / (
                                masses[a] * masses[b]
                            )
                        dist[a, b] = dist[b, a] = value
                return dist
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
    if cost is None:
        raise ValueError("Cost matrix is required for Wasserstein/OT aggregation.")
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


def _probabilities_for_layer(landscape: FitnessLandscape,
                             node_order: Sequence,
                             layer_name: str,
                             annotation_field: str | None = None) -> tuple[list[Any], np.ndarray]:
    if layer_name in landscape.fitness_layers:
        return _probability_matrix_from_fitness_layer(landscape, node_order, layer_name)
    if layer_name in landscape.annotation_layers:
        mats = _probability_matrices_from_annotation_layer(landscape, node_order, layer_name)
        if annotation_field is None:
            if len(mats) != 1:
                raise ValueError(
                    f"Annotation layer '{layer_name}' has multiple columns; specify annotation_field."
                )
            return next(iter(mats.values()))
        if annotation_field not in mats:
            raise KeyError(
                f"Column '{annotation_field}' not found in annotation layer '{layer_name}'."
            )
        return mats[annotation_field]
    raise KeyError(f"Layer '{layer_name}' not found among fitness or annotation layers.")


def _compute_resistance_matrix(G: nx.Graph,
                               node_order: Sequence,
                               *,
                               weight_key: Optional[str],
                               jitter: float,
                               sparse_threshold: int,
                               weight_epsilon: float,
                               weight_normalisation: bool,
                               compute_full_matrix: bool,
                               hutchinson_samples: int = 32,
                               hutchinson_seed: int | None = None) -> tuple[Optional[np.ndarray], np.ndarray, float, Optional[dict[str, Any]], Optional[np.ndarray], bool]:
    """
    Core resistance distance computation for a provided node ordering.
    """
    if not node_order:
        return (
            np.zeros((0, 0), dtype=float),
            np.zeros((0,), dtype=float),
            1.0,
            None,
            None,
            False,
        )

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
        return (
            np.zeros((0, 0), dtype=float),
            np.zeros((0,), dtype=float),
            norm_factor,
            None,
            None,
            False,
        )

    if n <= sparse_threshold:
        L = L_sparse.toarray()
        jitter_used = False
        if np.linalg.matrix_rank(L) < n - 1:
            if jitter <= 0.0:
                raise ValueError(
                    "Connected-component Laplacian is numerically rank deficient; "
                    "provide positive jitter for conditioning."
                )
            projector = np.eye(n) - np.ones((n, n), dtype=float) / n
            L = L + jitter * projector
            jitter_used = True
        try:
            L_pinv = np.linalg.pinv(L, hermitian=True)
        except np.linalg.LinAlgError:
            if jitter <= 0.0:
                raise
            if not jitter_used:
                projector = np.eye(n) - np.ones((n, n), dtype=float) / n
                L = L + jitter * projector
                jitter_used = True
            L_pinv = np.linalg.pinv(L, hermitian=True)
        diag = np.diag(L_pinv)
        R = None
        if compute_full_matrix:
            R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
            R[R < 0] = 0.0
            R = R / norm_factor
        return R, diag, norm_factor, None, L_pinv, jitter_used

    if n <= 1:
        return (
            np.zeros((n, n), dtype=float),
            np.zeros((n,), dtype=float),
            norm_factor,
            None,
            None,
            False,
        )

    ground = n - 1
    keep = list(range(n - 1))
    L_reduced = L_sparse[keep, :][:, keep].tocsc()
    jitter_used = False
    try:
        solver = splu(L_reduced)
    except RuntimeError:
        if jitter <= 0.0:
            raise ValueError(
                "Grounded component Laplacian factorization failed; provide "
                "positive jitter for conditioning."
            )
        L_reduced = L_reduced + jitter * sp.eye(n - 1, format="csc")
        jitter_used = True
        try:
            solver = splu(L_reduced)
        except RuntimeError:
            solver = None

    if solver is None:
        L = L_sparse.toarray()
        attempts = 0
        rank = n
        while attempts < 5:
            try:
                rank = np.linalg.matrix_rank(L)
                break
            except np.linalg.LinAlgError:
                projector = np.eye(n) - np.ones((n, n), dtype=float) / n
                L = L + (10 ** attempts) * jitter * projector
                jitter_used = True
                attempts += 1
        if rank < n - 1:
            projector = np.eye(n) - np.ones((n, n), dtype=float) / n
            L = L + jitter * projector
            jitter_used = True
        attempts = 0
        while attempts < 5:
            try:
                L_pinv = np.linalg.pinv(L, hermitian=True)
                break
            except np.linalg.LinAlgError:
                projector = np.eye(n) - np.ones((n, n), dtype=float) / n
                L = L + (10 ** attempts) * jitter * projector
                jitter_used = True
                attempts += 1
        else:
            raise
        diag = np.diag(L_pinv)
        R = None
        if compute_full_matrix:
            R = diag[:, None] + diag[None, :] - 2.0 * L_pinv
            R[R < 0] = 0.0
            R = R / norm_factor
        return R, diag, norm_factor, None, L_pinv, jitter_used

    diag_full = np.zeros(n, dtype=float)
    R = None
    solver_data = {"solver": solver, "ground": ground}

    if compute_full_matrix:
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
        diag_full[: n - 1] = diag
        return (
            R / norm_factor,
            diag_full,
            norm_factor,
            solver_data,
            None,
            jitter_used,
        )

    # Approximate diagonal via Hutchinson to avoid building full matrix
    rng = np.random.default_rng(hutchinson_seed)
    diag_est = np.zeros(n, dtype=float)
    for _ in range(max(1, hutchinson_samples)):
        z = rng.choice([-1.0, 1.0], size=n)
        z0 = z - z.mean()
        z_red = np.delete(z0, ground)
        x_red = solver.solve(z_red)
        x_full = np.insert(x_red, ground, 0.0)
        diag_est += z0 * x_full
    diag_est = diag_est / max(1, hutchinson_samples)
    diag_full[:] = diag_est
    return R, diag_full, norm_factor, solver_data, None, jitter_used


def _resistance_component_node_orders(
    source: FitnessLandscape | nx.Graph,
    graph: nx.Graph,
    node_order: Sequence,
    weight_key: str | None,
) -> tuple[nx.Graph, list[list[Hashable]]]:
    """Return the positive-conductance graph and ordered components."""
    selected_nodes = set(node_order)
    if isinstance(source, FitnessLandscape):
        topological_components = [
            set(component.graph.nodes()) for component in source.get_components()
        ]
    else:
        topological_components = [
            set(component) for component in nx.connected_components(graph)
        ]

    electrical_graph = nx.Graph()
    electrical_graph.graph.update(graph.graph)
    electrical_graph.add_nodes_from(
        (node, dict(graph.nodes[node])) for node in node_order
    )
    for u, v, data in graph.subgraph(selected_nodes).edges(data=True):
        if weight_key is None or float(data[weight_key]) > 0.0:
            electrical_graph.add_edge(u, v, **dict(data))

    position = {node: index for index, node in enumerate(node_order)}
    component_orders: list[list[Hashable]] = []
    for topological_nodes in topological_components:
        retained = selected_nodes & topological_nodes
        if not retained:
            continue
        subgraph = electrical_graph.subgraph(retained)
        for component_nodes in nx.connected_components(subgraph):
            component_orders.append(
                sorted(component_nodes, key=position.__getitem__)
            )
    component_orders.sort(key=lambda nodes: position[nodes[0]])
    return electrical_graph, component_orders


def resistance_distance_matrix(graph: Union[FitnessLandscape, nx.Graph],
                               nodes: Optional[Sequence] = None,
                               *,
                               weight_key: Optional[str] = AUTO_EDGE_KEY,
                               jitter: float = 1e-10,
                               sparse_threshold: int = 1000,
                               weight_epsilon: float = 0.0,
                               weight_normalisation: bool = True,
                               compute_resistance_matrix: bool = False,
                               hutchinson_samples: int = 32,
                               hutchinson_seed: Optional[int] = None,
                               sample_nodes: Optional[int] = None,
                               sample_fraction: Optional[float] = None,
                               sample_seed: Optional[int] = None,
                               sample_method: Literal["random", "head"] = "random",
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
    weight_key : str or None, default="auto"
        Edge attribute representing conductance. ``"auto"`` uses the
        constructor-declared conductance; ``None`` treats edges as unweighted.
    jitter : float, default=1e-10
        Numerical regularisation used only when factorization fails within a
        connected component. It is never used to bridge disconnected
        components, and the result reports whether it was applied.
    sparse_threshold : int, default=1000
        Node count above which sparse grounded-Laplacian factorization replaces
        the dense pseudoinverse path.
    weight_epsilon : float, default=0.0
        Optional positive perturbation added only to positive-conductance
        edges within each electrical component. Zero-conductance edges remain
        disconnected and are never converted to finite resistance paths.
    weight_normalisation : bool, default=True
        If ``True``, rescales the temporary Laplacian so its largest
        absolute entry is 1.0, improving numerical stability. The final
        resistance distances are rescaled back so results remain in the
        original units.
    compute_resistance_matrix : bool, default=False
        When ``False``, skips explicit construction of the full node-by-node
        resistance matrix unless required by the aggregation function
        (Wasserstein/OT). Set to ``True`` to always return ``"resistance_mat"``.
    hutchinson_samples : int, default=32
        Number of Hutchinson probe vectors to estimate the diagonal of the
        pseudoinverse when ``compute_resistance_matrix`` is False on large
        graphs. Ignored for small graphs where an exact pseudoinverse is
        available.
    hutchinson_seed : int, optional
        Optional random seed for Hutchinson probes.
    sample_nodes : int, optional
        If provided, subsample at most this many nodes (without replacement)
        before computing resistances/aggregations. Useful to reduce memory
        footprint on very large graphs.
    sample_fraction : float, optional
        Alternative to ``sample_nodes``; fraction of nodes to sample. Ignored
        when ``sample_nodes`` is provided.
    sample_seed : int, optional
        Optional seed controlling random node sampling when
        ``sample_method='random'``.
    sample_method : {"random", "head"}, default="random"
        Strategy for subsampling nodes when ``sample_nodes`` or
        ``sample_fraction`` is used. ``"head"`` selects the first k nodes in
        order; ``"random"`` selects uniformly without replacement.
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
            - ``"resistance_mat"`` : full node-by-node resistance matrix
              (present only when computed), with ``numpy.inf`` between
              disconnected electrical components.
            - ``"components"`` and ``"component_ids"`` : component membership
              in the sampled node order.
            - ``"jitter_used"`` and ``"jittered_components"`` : whether
              numerical regularisation was required within any component.
            - One entry per aggregated layer with ``"categories"`` (index
              to label mapping) and ``"distance_max"``/``"distance_mat"``
              holding the aggregated class-by-class distances.

    Notes
    -----
    Cross-component resistance is ``numpy.inf``. Off-diagonal expected-pairwise
    category distance is therefore infinite whenever positive pair mass spans
    distinct components; category self-distances remain zero by convention.
    Optimal transport is finite only when the two category
    distributions assign equal total mass to every component; in that case,
    transport is solved independently inside each component. Otherwise its
    distance is infinite.
    """
    G = graph.graph if isinstance(graph, FitnessLandscape) else graph
    if G is None:
        raise ValueError("Graph is required to compute resistance distances.")
    if not isinstance(G, nx.Graph) or G.is_directed() or G.is_multigraph():
        raise TypeError(
            "Resistance distance requires a simple undirected NetworkX graph."
        )
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be a finite non-negative value.")
    if not np.isfinite(weight_epsilon) or weight_epsilon < 0.0:
        raise ValueError("weight_epsilon must be a finite non-negative value.")
    resolved_weight_key = resolve_edge_attribute(
        G,
        "conductance",
        weight_key,
        required=False,
    )

    node_order = list(G.nodes()) if nodes is None else list(nodes)
    if len(set(node_order)) != len(node_order):
        raise ValueError("nodes must not contain duplicates.")
    missing_nodes = [node for node in node_order if node not in G]
    if missing_nodes:
        raise KeyError(f"Requested nodes are not in the graph: {missing_nodes[:3]!r}.")
    n_total = len(node_order)
    if sample_nodes is not None and sample_nodes < 1:
        raise ValueError("sample_nodes must be a positive integer when provided.")
    if sample_fraction is not None and not (0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be in (0, 1].")
    target = None
    if sample_nodes is not None:
        target = min(int(sample_nodes), n_total)
    elif sample_fraction is not None:
        target = min(int(np.ceil(sample_fraction * n_total)), n_total)
    if target is not None and target < n_total:
        if sample_method not in {"random", "head"}:
            raise ValueError(f"Unknown sample_method '{sample_method}'.")
        if sample_method == "random":
            rng = np.random.default_rng(sample_seed)
            positions = rng.choice(len(node_order), size=target, replace=False)
            node_order = [node_order[int(position)] for position in positions]
        else:
            node_order = node_order[:target]
    agg_mode = aggregation_function.lower()
    if agg_mode not in {"wasserstein", "ot", "expected_pairwise"}:
        raise ValueError(
            f"aggregation_function must be 'wasserstein', 'ot', or 'expected_pairwise', got {aggregation_function!r}"
        )

    require_matrix = agg_mode in {"wasserstein", "ot"} or layers is None
    compute_matrix = compute_resistance_matrix or require_matrix

    electrical_graph, component_orders = _resistance_component_node_orders(
        graph,
        G,
        node_order,
        resolved_weight_key,
    )
    internal_matrix_required = compute_matrix or len(component_orders) != 1
    jittered_components: list[int] = []

    if not component_orders:
        R = np.zeros((0, 0), dtype=float)
        diag_scaled = np.zeros(0, dtype=float)
        norm_factor = 1.0
        solver_data = None
        laplacian_pinv = None
    elif len(component_orders) == 1:
        (
            R,
            diag_scaled,
            norm_factor,
            solver_data,
            laplacian_pinv,
            jitter_used,
        ) = _compute_resistance_matrix(
            electrical_graph,
            component_orders[0],
            weight_key=resolved_weight_key,
            jitter=jitter,
            sparse_threshold=sparse_threshold,
            weight_epsilon=weight_epsilon,
            weight_normalisation=weight_normalisation,
            compute_full_matrix=internal_matrix_required,
            hutchinson_samples=hutchinson_samples,
            hutchinson_seed=hutchinson_seed,
        )
        if jitter_used:
            jittered_components.append(0)
    else:
        R = np.full((len(node_order), len(node_order)), np.inf, dtype=float)
        np.fill_diagonal(R, 0.0)
        position = {node: index for index, node in enumerate(node_order)}
        for component_id, component_nodes in enumerate(component_orders):
            (
                component_resistance,
                _,
                _,
                _,
                _,
                jitter_used,
            ) = _compute_resistance_matrix(
                electrical_graph,
                component_nodes,
                weight_key=resolved_weight_key,
                jitter=jitter,
                sparse_threshold=sparse_threshold,
                weight_epsilon=weight_epsilon,
                weight_normalisation=weight_normalisation,
                compute_full_matrix=True,
                hutchinson_samples=hutchinson_samples,
                hutchinson_seed=hutchinson_seed,
            )
            indices = [position[node] for node in component_nodes]
            R[np.ix_(indices, indices)] = component_resistance
            if jitter_used:
                jittered_components.append(component_id)
        diag_scaled = None
        norm_factor = 1.0
        solver_data = None
        laplacian_pinv = None

    component_id_by_node = {
        node: component_id
        for component_id, component_nodes in enumerate(component_orders)
        for node in component_nodes
    }
    result: Dict[str, Any] = {
        "sampled_nodes": list(node_order),
        "weight_key": resolved_weight_key,
        "component_count": len(component_orders),
        "components": [list(nodes) for nodes in component_orders],
        "component_ids": [component_id_by_node[node] for node in node_order],
        "cross_component_resistance": np.inf,
        "jitter": float(jitter),
        "jitter_used": bool(jittered_components),
        "jittered_components": jittered_components,
        "weight_epsilon": float(weight_epsilon),
    }
    if R is not None and compute_matrix:
        result["resistance_mat"] = R

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


def category_diffusion_hierarchy(
    landscape: FitnessLandscape,
    *,
    layer: str,
    annotation_field: str | None = None,
    embedding_dim: int = 10,
    diffusion_matrix: Literal["norm_laplacian", "laplacian"] = "norm_laplacian",
    weight_key: Optional[str] = AUTO_EDGE_KEY,
    skip_first: bool = True,
    embedding: Optional[np.ndarray] = None,
    filter_small_embedding: bool = True,
    embedding_norm_threshold: float = 1e-12,
    filter_coordinate_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a category-level hierarchy using a low-dimensional diffusion
    embedding and agglomerative clustering of category centroids.

    Parameters
    ----------
    landscape : FitnessLandscape
        Landscape providing the graph and category labels.
    layer : str
        Fitness or annotation layer name used for category definitions.
    annotation_field : str, optional
        Column to use when ``layer`` references an annotation layer with
        multiple columns.
    embedding_dim : int, default=10
        Number of diffusion embedding dimensions to use (after dropping
        the first eigenvector when ``skip_first`` is True).
    diffusion_matrix : {"norm_laplacian", "laplacian"}, default="norm_laplacian"
        Graph matrix passed to :func:`eigenmode_decomposition`.
    weight_key : str or None, default="auto"
        Conductance attribute forwarded to the Laplacian construction.
        ``None`` requests an unweighted operator.
    skip_first : bool, default=True
        Drop the leading eigenvector (often the constant mode) from the
        diffusion embedding.
    embedding : np.ndarray, optional
        Precomputed node embedding to use instead of computing a diffusion
        embedding. Must align with the landscape graph node order.
    filter_small_embedding : bool, default=True
        If ``True``, drop nodes whose embedding L2 norm is below
        ``embedding_norm_threshold`` to avoid degenerate points dominating
        plots/centroids.
    embedding_norm_threshold : float, default=1e-12
        Threshold used when ``filter_small_embedding`` is True.
    filter_coordinate_threshold : float, optional
        If provided, drop nodes whose absolute value in any embedding
        dimension is below this threshold. Useful when many points lie
        exactly on coordinate axes.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing embeddings, centroids, pairwise distances,
        clustering linkage, and spread/distance summary statistics.
    """
    if landscape.graph is None:
        raise ValueError("Landscape graph is required for diffusion hierarchy.")

    node_order = list(landscape.graph.nodes())
    categories, P = _probabilities_for_layer(landscape, node_order, layer, annotation_field)
    masses = P.sum(axis=0)

    if embedding is None:
        k_eff = embedding_dim + (1 if skip_first else 0)
        w, U = eigenmode_decomposition(
            landscape,
            k=k_eff,
            matrix="norm_laplacian" if diffusion_matrix == "norm_laplacian" else "laplacian",
            weight_key=weight_key,
        )
        if skip_first and U.shape[1] > 1:
            embedding = U[:, 1 : min(U.shape[1], embedding_dim + 1)]
            eigvals = w[1 : min(len(w), embedding_dim + 1)]
        else:
            embedding = U[:, :embedding_dim]
            eigvals = w[: embedding.shape[1]]
    else:
        embedding = np.asarray(embedding, dtype=float)
        eigvals = None
        if embedding.shape[0] != len(node_order):
            raise ValueError("Provided embedding rows must match the number of graph nodes.")

    if embedding.shape[1] == 0:
        raise ValueError("Embedding has zero columns; increase embedding_dim or provide a valid embedding.")

    kept_indices = list(range(len(node_order)))
    if filter_small_embedding:
        norms = np.linalg.norm(embedding, axis=1)
        mask = norms > embedding_norm_threshold
        if not mask.any():
            raise ValueError("All embeddings fall below the norm threshold; adjust or disable filtering.")
        embedding = embedding[mask]
        P = P[mask]
        kept_indices = [idx for idx, flag in zip(kept_indices, mask) if flag]
    if filter_coordinate_threshold is not None:
        coord_mask = (np.abs(embedding) >= filter_coordinate_threshold).all(axis=1)
        if not coord_mask.any():
            raise ValueError(
                "All embeddings filtered by coordinate threshold; relax filter_coordinate_threshold."
            )
        embedding = embedding[coord_mask]
        P = P[coord_mask]
        kept_indices = [idx for idx, flag in zip(kept_indices, coord_mask) if flag]

    num_categories = len(categories)
    centroids = np.full((num_categories, embedding.shape[1]), np.nan, dtype=float)
    spreads = np.full(num_categories, np.nan, dtype=float)

    for j in range(num_categories):
        mass = masses[j]
        if mass <= 0:
            continue
        weights = P[:, j][:, None]
        centroid = (weights * embedding).sum(axis=0) / mass
        centroids[j] = centroid
        diffs = embedding - centroid
        sq_dist = (weights.ravel() * (diffs ** 2).sum(axis=1)).sum() / mass
        spreads[j] = float(sq_dist)

    dist_mat = np.full((num_categories, num_categories), np.nan, dtype=float)
    for a in range(num_categories):
        if not np.isfinite(centroids[a]).all():
            continue
        for b in range(a, num_categories):
            if not np.isfinite(centroids[b]).all():
                continue
            if a == b:
                dist_mat[a, b] = 0.0
                continue
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            dist_mat[a, b] = dist_mat[b, a] = d

    finite_offdiag = dist_mat[~np.eye(num_categories, dtype=bool)]
    finite_offdiag = finite_offdiag[np.isfinite(finite_offdiag)]
    distance_stats = {
        "mean": float(np.nanmean(finite_offdiag)) if finite_offdiag.size else np.nan,
        "min": float(np.nanmin(finite_offdiag)) if finite_offdiag.size else np.nan,
        "max": float(np.nanmax(finite_offdiag)) if finite_offdiag.size else np.nan,
    }
    spread_stats = {
        "mean": float(np.nanmean(spreads)) if np.isfinite(spreads).any() else np.nan,
        "min": float(np.nanmin(spreads)) if np.isfinite(spreads).any() else np.nan,
        "max": float(np.nanmax(spreads)) if np.isfinite(spreads).any() else np.nan,
    }

    linkage_matrix = None
    dendrogram_order = None
    valid_idx = [i for i in range(num_categories) if np.isfinite(centroids[i]).all()]
    if len(valid_idx) >= 2:
        condensed = pdist(centroids[valid_idx], metric="euclidean")
        if condensed.size:
            linkage_matrix = linkage(condensed, method="average")
            order_indices = leaves_list(linkage_matrix)
            dendrogram_order = [categories[valid_idx[i]] for i in order_indices]

    return {
        "categories": categories,
        "embedding": embedding,
        "eigenvalues": eigvals,
        "centroids": centroids,
        "pairwise_distances": dist_mat,
        "distance_stats": distance_stats,
        "spread_per_category": {categories[i]: float(spreads[i]) if np.isfinite(spreads[i]) else None for i in range(num_categories)},
        "spread_stats": spread_stats,
        "linkage": linkage_matrix,
        "dendrogram_order": dendrogram_order,
        "kept_node_indices": kept_indices,
        "kept_nodes": [node_order[index] for index in kept_indices],
        "kept_sequence_indices": [
            landscape.sequence_index_for_node(node_order[index])
            for index in kept_indices
        ],
        "filtered_node_count": len(node_order) - len(kept_indices),
    }
