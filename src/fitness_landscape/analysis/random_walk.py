import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.fitness import CategoricalFitness, ProbabilisticCategoricalFitness
from ..core.graph import create_hamming_graph
from ..core.sequence import sequence_distance
from ..transforms import graph_fourier_transform, eigenmode_decomposition

def calculate_ruggedness_autocorrelation_analytical(landscape: FitnessLandscape,
                                                      lag_max: int = None,
                                                      _eigenvectors: Optional[np.ndarray] = None,
                                                      _eigenvalues: Optional[np.ndarray] = None) -> Dict:
    """
    Function to determine the analytical autocorrelation of a fitness
    landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    lag_max : int, default=`None`
        The maximum lag to include in the autocorrelation calculation.
    
    Returns
    -------
    results : Dict
        The analytical autocorrelation results.
    """
 
    # Center signal on 0
    raw_signal = landscape.get_signal()
    mean_centered_signal = raw_signal - np.mean(raw_signal)

    # Perform computation in fourier basis
    eigenvectors, _, gft_coefficients = graph_fourier_transform(
        landscape,
        signal=mean_centered_signal,
        _eigenvectors=_eigenvectors,
        _eigenvalues=_eigenvalues,
    )
    power_spectrum = np.abs(gft_coefficients)**2
    autocov_matrix = eigenvectors @ np.diag(power_spectrum) @ eigenvectors.T
    
    # Average over distances
    dist_matrix = nx.floyd_warshall_numpy(landscape.graph)
    max_dist = int(np.max(dist_matrix))
    autocorr_by_lag = [[] for _ in range(max_dist + 1)]
    num_nodes = landscape.graph.number_of_nodes()
    
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            k = int(dist_matrix[i, j])
            autocorr_by_lag[k].append(autocov_matrix[i, j])
            
    r_k = np.array([np.mean(vals) if vals else 0 for vals in autocorr_by_lag])
    
    # Normalize the final autocorrelation function by r(0).
    if r_k[0] != 0:
        r_k /= r_k[0]
    
    # Calculate the correlation length.
    if len(r_k) > 1 and 0 < np.abs(r_k[1]) < 1:
        correlation_length = -1 / np.log(np.abs(r_k[1]))
    else:
        correlation_length = np.inf

    if lag_max is not None:
        r_k = r_k[:lag_max + 1]
    
    return {
        'autocorrelation': r_k,
        'correlation_length': correlation_length
    }

def calculate_ruggedness_autocorrelation_stochastic(landscape: FitnessLandscape,
                                                      n_walks: int = 100,
                                                      steps: int = 100,
                                                      lag_max: int = None,
                                                      seed: Optional[int] = None) -> Dict:
    """
    Function to determine the stochastic autocorrelation of a fitness
    landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    steps : int, default=1000
        The number of random walk steps.
    
    lag_max : int, default=`None`
        The maximum lag to include in the autocorrelation calculation.
    
    seed : int, default=`None`
        The seed for the RNG.
    
    Returns
    -------
    results : Dict
        The stochastic autocorrelation results.
    """
    rng = np.random.default_rng(seed)
    
    if lag_max is None:
        lag_max = steps // 2
        
    all_autocorrs = []
    
    for _ in range(n_walks):
        # Start each walk at a random node
        current_node = rng.choice(list(landscape.graph.nodes()))
        
        fitness_trajectory = []
        for _ in range(steps):
            fitness_trajectory.append(landscape.get_fitness(landscape.sequences[current_node]))
            neighbors = list(landscape.graph.neighbors(current_node))
            if not neighbors:
                # Handle nodes with no neighbors 
                break
            current_node = rng.choice(neighbors)
        
        if len(fitness_trajectory) < 2:
            continue # Skip walks that are too short
            
        # De-mean the signal for this trajectory
        fitness_trajectory = np.array(fitness_trajectory)
        mean_fitness = np.mean(fitness_trajectory)
        
        # Calculate autocorrelation for this single walk
        autocorr = np.correlate(fitness_trajectory - mean_fitness, fitness_trajectory - mean_fitness, mode='full')
        
        # Normalize and slice
        autocorr = autocorr[len(fitness_trajectory)-1 : len(fitness_trajectory) -1 + lag_max]
        if autocorr[0] != 0:
            autocorr /= autocorr[0]
            all_autocorrs.append(autocorr)
            
    if not all_autocorrs:
        return {
            'autocorrelation': np.array([1.0] + [0.0] * (lag_max - 1)),
            'correlation_length': np.inf
        }

    # Average the autocorrelation functions from all the walks
    avg_autocorr = np.mean(all_autocorrs, axis=0)
    
    # Calculate correlation length from the averaged autocorrelation
    if len(avg_autocorr) > 1 and 0 < np.abs(avg_autocorr[1]) < 1:
        correlation_length = -1 / np.log(np.abs(avg_autocorr[1]))
    else:
        correlation_length = np.inf
    
    return {
        'autocorrelation': avg_autocorr,
        'correlation_length': correlation_length
    }


def category_boundary_crossing_times(
    landscape: FitnessLandscape,
    *,
    layer: Optional[str] = None,
    n_walks: int = 100,
    max_steps: int = 100,
    seed: Optional[int] = None,
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

    neighbors = {node: list(landscape.graph.neighbors(node)) for node in node_order}
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
                    current_node = rng.choice(neigh)
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
        },
    }
