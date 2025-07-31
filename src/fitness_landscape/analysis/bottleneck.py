# from typing import Union, Dict, Set, Any
# import numpy as np
# import networkx as nx
# from networkx.algorithms.cuts import conductance
# from eigenmode import eigenmode_decomposition
# from ..core.landscape import FitnessLandscape, _GraphLike

# #TODO: Broken - need to fix (theory and technical)

# def cheeger_sweep_cut(
#         graph: Union[FitnessLandscape,
#                      _GraphLike],
#                      weight: str = "weight") -> Dict[str, Any]:
#     """
#     Approximate the Cheeger constant via the classic Fiedler sweep cut.
#     """
#     if isinstance(graph, FitnessLandscape):
#         graph = graph.graph
#     elif not isinstance(graph, _GraphLike):
#         raise ValueError("Graph must be graph-like or FitnessLandscape")

#     lam, vecs = eigenmode_decomposition(
#         graph=graph,
#         k=2,
#         matrix="norm_laplacian",
#         return_eigenvectors=True
#     )
#     lambda_2 = lam[1]
#     fiedler = vecs[:, 1] # shape (|V|,)

#     nodes = list(graph.nodes())
#     order = [n for n, _ in sorted(zip(nodes, fiedler), key=lambda x: x[1])]

#     best_phi = float("inf")
#     best_set: Set = set()
#     S: Set = set()
    
#     # Sweep through splits in graph. Scales in O(|E|) time.
#     for v in order[:-1]: 
#         S.add(v)
#         phi = conductance(graph, S, weight=weight)
#         if phi < best_phi:
#             best_phi = phi
#             best_set = S.copy()

#     max_deg = max(dict(graph.degree(weight=weight)).values())
#     lower = lambda_2 / 2
#     upper = np.sqrt(2 * max_deg * lambda_2)

#     return {
#         "h_approx": best_phi,
#         "best_set": best_set,
#         "fiedler_value": lambda_2,
#         "h_lower_bound": lower,
#         "h_upper_bound": upper
#     }

# def _vol(graph: nx.Graph,
#          weight_key: str = None) -> float:
#     """
#     Helper function to sum the volume of a graph. 
#     Parameters
#     -----------
#     graph : nx.Graph    
#         The graph to analyze.
    
#     weight_key : str, default=`None`
#         The attribute key edge wweights are stored under.
    
#     Returns
#     -------
#     float 
#         The volume of the graph.
#     """
#     return sum(dict(graph.degree(weight=weight_key)).values())


# def _internal_edge_weight(graph: nx.Graph,
#                           weight_key: str = None) -> float:
#     """
    
#     Helper function to sum the total weight of edges in the graph.
    
#     Parameters
#     -----------
#     graph : nx.Graph
#         The graph to analyze.
    
#     weight_key : str, default=`None`
#         The attribute key edhe weights are stored under.
    
#     Returns
#     -------
#     float
#         The sum of internal weights.
#     """
#     return sum(d.get(weight_key, 1.0) for *_, d in graph.edges(data=True))


# def cheeger_energy_bound(G: nx.Graph,
#                          min_delta : float = None,
#                          weight_key: str = None ) -> Dict[str, float]:
#     """
#     """
#     h_S = cheeger_sweep_cut(G, weight_key)['h_approx']
#     vol_S = _vol(G, weight_key)
#     W_int = _internal_edge_weight(G, weight_key)

#     # If no minimum value, no deterministic lower bound and instead
#     # an expected energy based on a uniform distribution of threshold
#     # values in the open interval [0,1].

#     if min_delta is None:
#         E_lower = (11.0 / 72.0) * h_S * vol_S
    
    
#     # If there is a minimum value for difference in the survived graph
#     # and unobsered graph, determine the bound. 
#     else:
#         E_lower = 0.5 * (min_delta ** 2) * h_S * vol_S

#     # Upper bound is always the ceiling of all edges at a maximum
#     # squared difference of 1 (i.e., the binary case).
#     E_upper = W_int 

#     return {
#         "energy_lower_bound": E_lower,
#         "energy_upper_bound": E_upper,
#         "h_approx": h_S,
#         "vol_s": vol_S
#     }
    
# #TODO: OR negative curvature

