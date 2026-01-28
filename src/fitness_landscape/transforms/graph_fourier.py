import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from .eigenmode import eigenmode_decomposition

def graph_fourier_transform(graph: Union[nx.Graph, FitnessLandscape],
                            signal: np.ndarray = None,
                            matrix: Literal['laplacian', 'norm_laplacian'] = 'laplacian',
                            k: int = None) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute graph Fourier transform of a signal on the graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph representation or fitness landscape.
    signal : array-like, default=`None`
        Signal on the graph (e.g., fitness values). If `None` and graph
        is a FitnessLandscape, fitness values are used.
    matrix : str, default=`laplacian`
        The matrix eigenbasis to use. Options are `laplacian` or
        `norm_laplacian`.
    k : int, default=`None`
        Number of eigenvectors to use. If None, compute all eigenvectors
        (dense; may be expensive for large graphs).
        
    Returns
    -------
    tuple
        (eigenvectors, eigenvalues, coefficients)
    """
    if isinstance(graph, FitnessLandscape):
        if signal is None:
            signal = graph.get_signal()  # single pass over active layer
        graph = graph.graph

    w, U = eigenmode_decomposition(graph, k=k, matrix=matrix)

    coeffs = None
    if signal is not None:
        x = np.asarray(signal, dtype=float)
        if x.shape[0] != U.shape[0]:
            raise ValueError(f"Signal length {x.shape[0]} does not match graph size {U.shape[0]}")
        coeffs = U.T @ x
    return U, w, coeffs
