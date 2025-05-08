from typing import Union, Dict, Set, Any
import numpy as np
import networkx as nx
from networkx.algorithms.cuts import conductance
from eigenmode import eigenmode_decomposition
from ..core.landscape import FitnessLandscape, _GraphLike

def cheeger_sweep_cut(
        graph: Union[FitnessLandscape,
                     _GraphLike],
                     weight: str = "weight") -> Dict[str, Any]:
    """
    Approximate the Cheeger constant via the classic Fiedler sweep cut.
    """
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    elif not isinstance(graph, _GraphLike):
        raise ValueError("Graph must be graph-like or FitnessLandscape")

    lam, vecs = eigenmode_decomposition(
        graph=graph,
        k=2,
        matrix="norm_laplacian",
        return_eigenvectors=True
    )
    lambda_2 = lam[1]
    fiedler = vecs[:, 1] # shape (|V|,)

    nodes = list(graph.nodes())
    order = [n for n, _ in sorted(zip(nodes, fiedler), key=lambda x: x[1])]

    best_phi = float("inf")
    best_set: Set = set()
    S: Set = set()
    
    # Sweep through splits in graph. Scales in O(|E|) time.
    for v in order[:-1]: 
        S.add(v)
        phi = conductance(graph, S, weight=weight)
        if phi < best_phi:
            best_phi = phi
            best_set = S.copy()

    max_deg = max(dict(graph.degree(weight=weight)).values())
    lower = lambda_2 / 2
    upper = np.sqrt(2 * max_deg * lambda_2)

    return {
        "h_approx": best_phi,
        "best_set": best_set,
        "fiedler_value": lambda_2,
        "h_lower_bound": lower,
        "h_upper_bound": upper
    }

def compute_cheeger_energy_floor() -> Dict:
    """
    
    """
    



#TODO: DE floor from Cheeger

#TODO: OR negative curvature

