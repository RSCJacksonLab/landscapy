"""Analyse discrete and continuous random walks on fitness landscapes."""

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from numbers import Real
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.edge_schema import AUTO_EDGE_KEY, resolve_edge_attribute
from ..core.fitness import CategoricalFitness, ProbabilisticCategoricalFitness
from ..core.graph import create_hamming_graph
from ..core.sequence import sequence_distance


_AUTOCORRELATION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _StationaryWalk:
    """Validated reversible random-walk data shared by all estimators."""

    node_order: tuple[Any, ...]
    transition: sp.csr_array
    symmetric_transition: sp.csr_array
    stationary: np.ndarray
    centered_signal: np.ndarray
    variance: float
    weight_key: Optional[str]


def _stationary_walk(
    landscape: FitnessLandscape,
    weight_key: Optional[str],
) -> _StationaryWalk:
    """Build the declared undirected conductance walk and stationary measure."""
    graph = landscape.graph
    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("Landscape graph contains no nodes.")
    if graph.is_directed():
        raise TypeError(
            "Autocorrelation in Landscapy 0.9 requires an undirected graph."
        )
    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        raise ValueError(
            "Autocorrelation requires one connected, non-trivial graph without "
            "isolates; analyse landscapes returned by get_components() separately."
        )

    resolved_weight_key = resolve_edge_attribute(
        graph,
        "conductance",
        weight_key,
        required=False,
    )
    node_order = tuple(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=node_order,
        weight=resolved_weight_key,
        dtype=float,
        format="csr",
    )
    degrees = np.asarray(adjacency.sum(axis=1), dtype=float).ravel()
    if np.any(~np.isfinite(degrees)) or np.any(degrees <= 0.0):
        raise ValueError(
            "The declared conductances must give every node positive finite degree; "
            "analyse connected components separately with get_components()."
        )

    support = adjacency.copy()
    support.data = (support.data > 0.0).astype(np.int8)
    support.eliminate_zeros()
    component_count = sp.csgraph.connected_components(
        support,
        directed=False,
        return_labels=False,
    )
    if component_count != 1:
        raise ValueError(
            "Positive-conductance edges must form one connected graph; analyse "
            "connected components separately with get_components()."
        )

    inverse_degree = 1.0 / degrees
    transition = (sp.diags(inverse_degree) @ adjacency).tocsr()
    inverse_root_degree = 1.0 / np.sqrt(degrees)
    symmetric_transition = (
        sp.diags(inverse_root_degree)
        @ adjacency
        @ sp.diags(inverse_root_degree)
    ).tocsr()
    stationary = degrees / np.sum(degrees)

    raw_signal = np.asarray(landscape.get_node_signal(node_order), dtype=float)
    if raw_signal.shape != (len(node_order),) or np.any(~np.isfinite(raw_signal)):
        raise ValueError("Autocorrelation requires one finite scalar per graph node.")
    centered_signal = raw_signal - float(stationary @ raw_signal)
    variance = float(stationary @ np.square(centered_signal))
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(
            "Autocorrelation is undefined for a constant or zero-stationary-variance "
            "fitness signal."
        )

    return _StationaryWalk(
        node_order=node_order,
        transition=transition,
        symmetric_transition=symmetric_transition,
        stationary=stationary,
        centered_signal=centered_signal,
        variance=variance,
        weight_key=resolved_weight_key,
    )


def _validate_lag_max(lag_max: Optional[int], default: int) -> int:
    """Return a non-negative integer maximum lag."""
    if lag_max is None:
        return default
    if isinstance(lag_max, (bool, np.bool_)) or not isinstance(
        lag_max, (int, np.integer)
    ):
        raise TypeError("lag_max must be a non-negative integer or None.")
    if lag_max < 0:
        raise ValueError("lag_max must be a non-negative integer or None.")
    return int(lag_max)


def _bounded_autocorrelation(values: np.ndarray) -> np.ndarray:
    """Enforce the exact normalized-correlation bound up to roundoff."""
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise RuntimeError("Autocorrelation evaluation produced a non-finite value.")
    if np.any(np.abs(values) > 1.0 + _AUTOCORRELATION_TOLERANCE):
        raise RuntimeError("Autocorrelation evaluation violated |C| <= 1.")
    return np.clip(values, -1.0, 1.0)


def _equivalent_single_exponential_length(values: np.ndarray) -> Optional[float]:
    """Return the lag-one-matched geometric envelope, never a mixing time."""
    if len(values) < 2:
        return None
    magnitude = abs(float(values[1]))
    if magnitude <= _AUTOCORRELATION_TOLERANCE:
        return 0.0
    if magnitude >= 1.0 - _AUTOCORRELATION_TOLERANCE:
        return np.inf
    return float(-1.0 / np.log(magnitude))


def _elementary_eigenvalue(walk: _StationaryWalk) -> Optional[float]:
    """Return the sole transition eigenvalue, or ``None`` for a modal mixture."""
    mode_signal = np.sqrt(walk.stationary) * walk.centered_signal
    transformed = walk.symmetric_transition @ mode_signal
    eigenvalue = float(mode_signal @ transformed / (mode_signal @ mode_signal))
    residual = transformed - eigenvalue * mode_signal
    scale = float(np.linalg.norm(mode_signal))
    if np.linalg.norm(residual) <= 1e-10 * scale:
        return eigenvalue
    return None


def _autocorrelation_result(values: np.ndarray, lags: np.ndarray) -> Dict[str, Any]:
    """Build the discrete result without assigning a generic mixing length."""
    return {
        "autocorrelation": values,
        "lags": lags,
        "correlation_length": None,
        "equivalent_single_exponential_length": (
            _equivalent_single_exponential_length(values)
        ),
    }


def calculate_ruggedness_autocorrelation_analytical(
    landscape: FitnessLandscape,
    lag_max: Optional[int] = None,
    weight_key: str | None = AUTO_EDGE_KEY,
    _eigenvectors: Optional[np.ndarray] = None,
    _eigenvalues: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate stationary discrete-time random-walk autocorrelation.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    lag_max : int, optional
        Maximum integer Markov lag. The returned lags are ``0, ..., lag_max``.
        By default, the graph diameter is used as a finite convenience range.
    weight_key : str or None, default="auto"
        Conductance attribute used to construct the transition kernel. ``None``
        requests the unweighted simple random walk.
    _eigenvectors : ndarray, optional
        Deprecated compatibility parameter. Markov autocorrelation is evaluated
        directly and does not use precomputed combinatorial-Laplacian modes.
    _eigenvalues : ndarray, optional
        Deprecated compatibility parameter corresponding to ``_eigenvectors``.

    Returns
    -------
    dict
        Autocorrelation, integer lags, an explicit lag-one equivalent
        single-exponential descriptor, and a ``None`` generic correlation-length
        compatibility field.

    Notes
    -----
    This evaluates ``x_c.T @ diag(pi) @ P**k @ x_c`` divided by stationary
    variance. It preserves negative and oscillatory correlations of a non-lazy
    walk. The equivalent single-exponential length matches only ``abs(C(1))``;
    it is not a generic decay scale or mixing time. The private eigenpair
    parameters are accepted only for source compatibility and are ignored.
    """
    del _eigenvectors, _eigenvalues
    walk = _stationary_walk(landscape, weight_key)
    max_lag = _validate_lag_max(lag_max, nx.diameter(landscape.graph))
    values = np.empty(max_lag + 1, dtype=float)
    propagated = walk.centered_signal.copy()
    weighted_signal = walk.stationary * walk.centered_signal
    for lag in range(max_lag + 1):
        values[lag] = float(weighted_signal @ propagated / walk.variance)
        propagated = walk.transition @ propagated
    values = _bounded_autocorrelation(values)
    values[0] = 1.0
    result = _autocorrelation_result(values, np.arange(max_lag + 1))
    result["elementary"] = _elementary_eigenvalue(walk) is not None
    result["weight_key"] = walk.weight_key
    return result


def _validate_times(
    times: Optional[Union[Real, Iterable[Real]]],
    default: int,
) -> np.ndarray:
    """Return a one-dimensional array of finite non-negative diffusion times."""
    if times is None:
        return np.arange(default + 1, dtype=float)
    raw_times = (
        [times]
        if isinstance(times, Real) and not isinstance(times, (bool, np.bool_))
        else times
    )
    try:
        values = list(raw_times)
    except TypeError as error:
        raise TypeError(
            "times must be a real number or an iterable of real numbers."
        ) from error
    if not values:
        raise ValueError("times must contain at least one evaluation time.")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in values
    ):
        raise TypeError("times must contain only finite non-negative real numbers.")
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("times must contain only finite non-negative real numbers.")
    return result


def time_continuous_autocorrelation(
    landscape: FitnessLandscape,
    times: Optional[Union[Real, Iterable[Real]]] = None,
    weight_key: str | None = AUTO_EDGE_KEY,
) -> Dict[str, Any]:
    """Calculate stationary continuous-time random-walk autocorrelation.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    times : real or iterable of real, optional
        Finite non-negative diffusion times. By default, evaluate at real-valued
        times corresponding numerically to ``0, ..., graph diameter``.
    weight_key : str or None, default="auto"
        Conductance attribute used to construct the transition kernel. ``None``
        requests the unweighted simple random walk.

    Returns
    -------
    dict
        Continuous-time autocorrelation and its requested diffusion times.

    Notes
    -----
    This evaluates ``x_c.T @ diag(pi) @ exp(-t * (I - P)) @ x_c`` divided by
    stationary variance. Diffusion time is continuous and is not a number of
    graph steps. A general multimode curve is not collapsed to one decay time.
    """
    walk = _stationary_walk(landscape, weight_key)
    evaluation_times = _validate_times(times, nx.diameter(landscape.graph))
    symmetric_laplacian = (
        sp.eye(len(walk.node_order), format="csr") - walk.symmetric_transition
    )
    mode_signal = np.sqrt(walk.stationary) * walk.centered_signal
    values = np.empty(len(evaluation_times), dtype=float)
    for index, time in enumerate(evaluation_times):
        if time == 0.0:
            values[index] = 1.0
        else:
            propagated = spla.expm_multiply(
                -time * symmetric_laplacian,
                mode_signal,
            )
            values[index] = float(mode_signal @ propagated / walk.variance)
    values = _bounded_autocorrelation(values)
    elementary_eigenvalue = _elementary_eigenvalue(walk)
    elementary_time = (
        float(1.0 / (1.0 - elementary_eigenvalue))
        if elementary_eigenvalue is not None
        else None
    )
    return {
        "autocorrelation": values,
        "times": evaluation_times,
        "correlation_time": None,
        "elementary": elementary_eigenvalue is not None,
        "elementary_correlation_time": elementary_time,
        "weight_key": walk.weight_key,
    }


def calculate_ruggedness_autocorrelation_stochastic(
    landscape: FitnessLandscape,
    n_walks: int = 100,
    steps: int = 100,
    lag_max: Optional[int] = None,
    seed: Optional[int] = None,
    weight_key: str | None = AUTO_EDGE_KEY,
) -> Dict[str, Any]:
    """Estimate stationary discrete-time random-walk autocorrelation.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    n_walks : int, default=100
        Number of independent random walks.
    steps : int, default=100
        Number of sampled states in each trajectory.
    lag_max : int, optional
        Maximum integer lag, included in the result. Defaults to ``steps // 2``.
    seed : int, optional
        Seed for the random-number generator.
    weight_key : str or None, default="auto"
        Conductance attribute used to construct the transition kernel. ``None``
        requests the unweighted simple random walk.

    Returns
    -------
    dict
        Pair-count-weighted autocorrelation estimates and integer lags.

    Notes
    -----
    Trajectories start in the stationary distribution. Every product is centred
    by the global stationary mean and divided by the exact common stationary
    variance, so the estimator targets the same signed, non-lazy ``P**k``
    quantity as :func:`calculate_ruggedness_autocorrelation_analytical`.
    """
    if isinstance(n_walks, (bool, np.bool_)) or not isinstance(
        n_walks, (int, np.integer)
    ):
        raise TypeError("n_walks must be a positive integer.")
    if n_walks <= 0:
        raise ValueError("n_walks must be a positive integer.")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(
        steps, (int, np.integer)
    ):
        raise TypeError("steps must be a positive integer.")
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")

    walk = _stationary_walk(landscape, weight_key)
    max_lag = _validate_lag_max(lag_max, int(steps) // 2)
    if max_lag >= steps:
        raise ValueError("lag_max must be smaller than steps.")
    rng = np.random.default_rng(seed)
    transition = walk.transition
    product_sums = np.zeros(max_lag + 1, dtype=float)
    pair_counts = np.zeros(max_lag + 1, dtype=np.int64)

    for _ in range(int(n_walks)):
        current = int(rng.choice(len(walk.node_order), p=walk.stationary))
        trajectory = np.empty(int(steps), dtype=float)
        for step in range(int(steps)):
            trajectory[step] = walk.centered_signal[current]
            row_start = transition.indptr[current]
            row_end = transition.indptr[current + 1]
            destinations = transition.indices[row_start:row_end]
            probabilities = transition.data[row_start:row_end]
            current = int(rng.choice(destinations, p=probabilities))
        for lag in range(max_lag + 1):
            products = trajectory[: int(steps) - lag] * trajectory[lag:]
            product_sums[lag] += float(np.sum(products))
            pair_counts[lag] += products.size

    estimates = product_sums / pair_counts / walk.variance
    estimates[0] = 1.0
    result = _autocorrelation_result(estimates, np.arange(max_lag + 1))
    result["pair_counts"] = pair_counts
    result["weight_key"] = walk.weight_key
    return result


def category_boundary_crossing_times(
    landscape: FitnessLandscape,
    *,
    layer: Optional[str] = None,
    n_walks: int = 100,
    max_steps: int = 100,
    seed: Optional[int] = None,
    weight_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate expected category-to-category boundary crossing times via random walks.

    The active fitness layer (or a specified one) must be categorical or probabilistic
    categorical. For each ordered pair of categories (a, b), random walks are started
    from nodes drawn according to the category-a probability mass and steps are counted
    until the walk first visits category b.

    Parameters
    ----------
    landscape : FitnessLandscape
        Source landscape with a graph and categorical fitness layer.
    layer : str, optional
        Fitness layer to use. Defaults to the active layer.
    n_walks : int, default=100
        Number of random walks per source-target category pair.
    max_steps : int, default=100
        Maximum steps per walk before aborting (counts as a miss).
    seed : int, optional
        Random seed.
    weight_key : str, optional
        Edge attribute used to weight neighbour transitions. If ``None``,
        neighbours are sampled uniformly. Missing attributes default to 1.0,
        matching NetworkX's weighted-matrix convention.

    Returns
    -------
    Dict[str, Any]
        {
            "categories": {i: name},
            "mean_crossing_time": matrix,
            "hit_counts": matrix,
            "params": {...},
        }
    """
    if landscape.graph is None:
        raise ValueError("Landscape graph is required for boundary crossing time estimation.")

    layer_obj = landscape.active_layer if layer is None else landscape.view(layer)
    if not isinstance(layer_obj, (CategoricalFitness, ProbabilisticCategoricalFitness)):
        raise TypeError("Active/selected fitness layer must be categorical.")

    categories = list(layer_obj.categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    node_order = list(landscape.graph.nodes())
    n_nodes = len(node_order)
    n_cat = len(categories)

    P = np.zeros((n_nodes, n_cat), dtype=float)
    attr_key = f"fitness_{layer_obj.name}"
    for row, node in enumerate(node_order):
        raw = landscape.graph.nodes[node].get(attr_key)
        if raw is None:
            continue
        if isinstance(layer_obj, ProbabilisticCategoricalFitness):
            # Accept vector-like or mapping of category -> prob
            if isinstance(raw, dict):
                for cat, prob in raw.items():
                    idx = cat_to_idx.get(cat)
                    if idx is None:
                        continue
                    P[row, idx] = float(prob)
            else:
                vals = np.asarray(raw, dtype=float)
                if vals.size != n_cat:
                    raise ValueError("Probabilistic fitness vector size mismatch.")
                P[row, :] = vals
        else:
            idx = cat_to_idx.get(raw)
            if idx is None:
                continue
            P[row, idx] = 1.0

    masses = P.sum(axis=0)
    rng = np.random.default_rng(seed)

    mean_mat = np.full((n_cat, n_cat), np.nan, dtype=float)
    std_mat = np.full((n_cat, n_cat), np.nan, dtype=float)
    hits_mat = np.zeros((n_cat, n_cat), dtype=int)

    neighbors: dict[Any, list[Any]] = {}
    neighbor_probabilities: dict[Any, Optional[np.ndarray]] = {}
    for node in node_order:
        node_neighbors = list(landscape.graph.neighbors(node))
        neighbors[node] = node_neighbors
        if weight_key is None or not node_neighbors:
            neighbor_probabilities[node] = None
            continue
        weights = np.asarray(
            [landscape.graph.edges[node, neighbor].get(weight_key, 1.0) for neighbor in node_neighbors],
            dtype=float,
        )
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(
                f"Edge weights for '{weight_key}' must be finite and non-negative."
            )
        total_weight = float(weights.sum())
        if total_weight <= 0:
            raise ValueError(
                f"Node {node!r} has no positive transition weight for '{weight_key}'."
            )
        neighbor_probabilities[node] = weights / total_weight
    node_to_idx = {node: i for i, node in enumerate(node_order)}

    for a in range(n_cat):
        if masses[a] <= 0:
            continue
        start_probs = P[:, a] / masses[a]
        start_indices = np.arange(n_nodes)
        for b in range(n_cat):
            if a == b or masses[b] <= 0:
                if a == b:
                    mean_mat[a, b] = 0.0
                continue
            hits: list[int] = []
            for _ in range(n_walks):
                current_row = rng.choice(start_indices, p=start_probs)
                current_node = node_order[current_row]
                steps = 0
                while steps < max_steps:
                    probs = P[current_row]
                    if probs.sum() > 0:
                        cat_hit = rng.choice(n_cat, p=probs / probs.sum())
                        if cat_hit == b:
                            hits.append(steps)
                            break
                    neigh = neighbors.get(current_node, [])
                    if not neigh:
                        break
                    neighbor_position = rng.choice(
                        len(neigh),
                        p=neighbor_probabilities.get(current_node),
                    )
                    current_node = neigh[int(neighbor_position)]
                    current_row = node_to_idx[current_node]
                    steps += 1
            if hits:
                mean_mat[a, b] = float(np.mean(hits))
                std_mat[a, b] = float(np.std(hits, ddof=1)) if len(hits) > 1 else 0.0
                hits_mat[a, b] = len(hits)

    return {
        "categories": {i: c for i, c in enumerate(categories)},
        "mean_crossing_time": mean_mat,
        "std_crossing_time": std_mat,
        "hit_counts": hits_mat,
        "params": {
            "layer": layer_obj.name,
            "n_walks": n_walks,
            "max_steps": max_steps,
            "seed": seed,
            "weight_key": weight_key,
        },
    }
