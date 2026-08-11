from .._optional import require_optional

gudhi = require_optional(
    "gudhi",
    extra="analysis",
    purpose="persistent-homology analysis",
)
import networkx as nx
import numpy as np
from collections import defaultdict
SimplexTree = gudhi.SimplexTree
from typing import Dict, Literal, Optional, Tuple
from ..core.landscape import FitnessLandscape
from ..utils import get_landscape_dist_mat


def vietoris_rips_complex(landscape: FitnessLandscape,
                          max_dim: int = 2,
                          max_distance: Optional[float] = None,
                          dist_matrix: Optional[np.ndarray] = None,
                          weighted: bool = False) -> SimplexTree:
    '''
    Compute the Vietoris-Rips complex of a fitness landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    max_dim : int, default=2
        The maximum dimension of the simplices to compute.
    
    max_distance : float, default=None
        The maximum distance to consider for the simplices.

    dist_matrix : np.ndarray,default=`None`
        The distance matrix to use. If `None`, the graph walk distance
        matrix will be used.

    weighted : bool, default=`False`
        Whether to use weighted edges in the graph representation.
    
    Returns
    -------
    simplices : list
        List of simplices in the Vietoris-Rips complex.
    '''
    # get distance matrix
    if dist_matrix is None:
        dist_matrix = get_landscape_dist_mat(landscape,
                                        weighted=weighted)

    # get vietoris-rips complex
    if max_distance is None:
        max_distance = np.max(dist_matrix[np.isfinite(dist_matrix)])
    rips_complex = gudhi.RipsComplex(distance_matrix=dist_matrix,
                                     max_edge_length=max_distance)
    # create simplex tree
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)

    return simplex_tree


def delauny_cech_complex(landscape: FitnessLandscape) -> SimplexTree:
    """
    Compute the Čech complex of a fitness landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
    
    max_dim : int, default=2
        The maximum dimension of the simplices to compute.
    
    max_distance : float, default=None
        The maximum distance to consider for the simplices.
    
    Returns
    -------
    simplices : list
        List of simplices in the Čech complex.
    """
    
    # get datapoints
    sequences = landscape.sequences
    if not sequences:
        raise ValueError("Landscape contains no sequences.")
    
    reps = np.array([seq.to_array() for seq in sequences])
    delaunay_cech_complex = gudhi.DelaunayComplex(points=reps)
    
    # create simplex tree
    simplex_tree = delaunay_cech_complex.create_simplex_tree()

    return simplex_tree


def compute_persistent_homology(landscape: FitnessLandscape,
                                max_dim: int = 2,
                                dist_matrix: np.ndarray = None,
                                max_distance: Optional[float] = None,
                                weighted: bool = False) -> dict:
    """
    Compute the persistent homology of a fitness landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.

    max_dim : int, default=2
        The maximum dimension of the simplices to compute.
    dist_matrix : np.ndarray,default=`None`
        The distance matrix to use in vietrois-rips complex
        computations. If `None`, the graph walk distance matrix will
        be used.
    max_distance : float, optional
        Maximum Vietoris-Rips edge filtration value. If omitted, use the
        largest finite entry of the distance matrix.
    weighted : bool, default=False
        When deriving graph distances, use the graph's ``weight`` edge
        attribute. Ignored when ``dist_matrix`` is supplied.

    Returns
    -------
    dict
        GUDHI simplex tree, persistence pairs and intervals, Betti numbers,
        and finite-lifetime summaries grouped by homology dimension.

    Notes
    -----
    This function always builds a Vietoris-Rips filtration. Persistence
    entropy is the Shannon entropy of finite lifetimes normalized to sum to
    one; essential intervals are counted separately and excluded from finite
    lifetime statistics.
    """
    # landscape filtration via vietoris-rips complex
    simplex_tree = vietoris_rips_complex(landscape,
                                            max_dim=max_dim,
                                            max_distance=max_distance,
                                            dist_matrix=dist_matrix,
                                            weighted=weighted)

    # compute persistent homology
    simplex_tree.compute_persistence()

    # persistent pairs
    persistence_pairs = simplex_tree.persistence_pairs()
    persistence_intervals = simplex_tree.persistence()

    # betti numbers
    betti_numbers = simplex_tree.betti_numbers()

    # stats for each dim
    stats = {}
    stats["by_dim"] = {}
    for dim in range(max_dim + 1):
        dim_pairs = [
            (death - birth) 
            for d, (birth, death) in persistence_intervals
            if d == dim and death < float('inf')
        ]
        if dim_pairs:
            stats["by_dim"][f"dim_{dim}"] = {
                "count": len(dim_pairs),
                "mean": np.mean(dim_pairs),
                "std": np.std(dim_pairs) if len(dim_pairs) > 1 else 0,
                "max": np.max(dim_pairs),
                "total_persistence": np.sum(dim_pairs)
            }
        else:
            stats["by_dim"][f"dim_{dim}"] = {
                "count": 0,
                "mean": 0,
                "std": 0,
                "max": 0,
                "total_persistence": 0
            }

    # global stats
    stats['global'] = {}

    ## lifetime stats
    all_lifetimes = [
        death - birth for _, (birth, death) in persistence_intervals
        if death < float('inf')
    ]

    if all_lifetimes:
        stats['global']["all_lifetimes"] = {
            "count": len(all_lifetimes),
            "mean": np.mean(all_lifetimes),
            "std": np.std(all_lifetimes) if len(all_lifetimes) > 1 else 0,
            "max": np.max(all_lifetimes),
            "total_persistence": np.sum(all_lifetimes)
        }

        # Persistence entropy
        probs = np.array(all_lifetimes) / np.sum(all_lifetimes)
        entropy = -np.sum(probs * np.log(probs))
        stats['global']["persistence_entropy"] = entropy
    else:
        stats['global']["all_lifetimes"] = {
            "count": 0,
            "mean": 0,
            "std": 0,
            "max": 0,
            "total_persistence": 0
        }
        stats['global']["persistence_entropy"] = 0

    ## infinite features
    n_infinite_features = sum(
        1 for _, (_, death) in persistence_intervals
        if death == float('inf')
    )
    stats['global']["n_infinite_features"] = n_infinite_features

    return {
        "simplex_tree": simplex_tree,
        "persistence_pairs": persistence_pairs,
        "persistence_intervals": persistence_intervals,
        "betti_numbers": betti_numbers,
        "stats": stats,
    }


def compute_betti_curves(persistence_intervals: Tuple,
                         max_dim: int = 2,
                         resolution: int = 100) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    '''
    Compute the Betti curves from the persistence intervals.

    Parameters
    ----------
    persistence_intervals : list
        List of persistence intervals. Can be obtained from the
        `compute_persistent_homology` function.
    
    max_dim : int, default=2
        The maximum dimension of the simplices to compute.
    
    Returns
    -------
    betti_curves : dict
        Dictionary of Betti curves for each dimension.
    '''
    # get betti curves for each dimension
    pairs_by_dim = defaultdict(list)
    for dim, (birth, death) in persistence_intervals:
        if dim <= max_dim:
            pairs_by_dim[dim].append((birth, death))
    
    # determine range for filtration value
    if persistence_intervals:
        min_birth = min(birth for birth, _ in persistence_intervals)
        max_death = max(death for _, death in persistence_intervals)
    else:
        min_birth, max_death = 0, 1

    # filtration range
    filtration_range = np.linspace(min_birth, max_death, resolution)
    step = filtration_range[1] - filtration_range[0]
    filtration_range = np.append(filtration_range, filtration_range[-1] + step)

    # compute betti curves
    betti_curves = {}
    for dim in range(max_dim + 1):
        betti_curve = np.zeros(len(filtration_range))
        for t_idx, t in enumerate(filtration_range):
            # Count features that are born before or at t and die after t
            count = sum(1 for birth, death in pairs_by_dim[dim] 
                      if birth <= t and (death > t or death == float('inf')))
            betti_curve[t_idx] = count
        betti_curves[dim] = betti_curve
    
    return betti_curves, filtration_range
