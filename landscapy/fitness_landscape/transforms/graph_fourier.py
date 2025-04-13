import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..analysis.eigenmode import eigenmode_decomposition


def graph_fourier_transform(graph: Union[nx.Graph, FitnessLandscape],
                            signal=None,
                            k=None,
                            backend: Literal['numpy, torch'] = 'numpy') -> Union[torch.Tensor, np.ndarray]:
    """
    Compute graph Fourier transform of a signal on the graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph representation or fitness landscape.
    signal : array-like, default=`None`
        Signal on the graph (e.g., fitness values). If `None` and graph
        is a FitnessLandscape, fitness values are used.
    k : int, default=`None`
        Number of eigenvectors to use. If None, use all eigenvectors.
    backend : str, default=`numpy`
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    tuple
        (eigenvectors, eigenvalues, coefficients)
    """
    # Handle FitnessLandscape input
    if isinstance(graph, FitnessLandscape):
        
        if signal is None:
            # Extract fitness values as signal
            signal = np.array([graph.get_fitness(seq) for seq in graph.sequences])
        graph = graph.graph
    
    # Ensure graph is a NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph or FitnessLandscape")
    
    # Compute graph Fourier transform based on backend
    if backend == 'numpy':
        return _graph_fourier_transform_numpy(graph, signal, k)
    elif backend == 'torch':
        return _graph_fourier_transform_torch(graph, signal, k)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _graph_fourier_transform_numpy(graph: Union[nx.Graph, FitnessLandscape],
                                   signal: np.ndarray) -> Tuple:
    """
    Helper function to compute the GFT using the numpy backend.

    Parameters
    ----------
    graph : nx.Graph or FitnessLandscape
        The fitness landscape of graph representation.
    
    signal : np.ndarray
        The signal to project onto the eigenbasis of the Laplacian.
    
    Returns
    -------
    eigenvectors : np.ndarray
        The eigenvectors of the Laplacian. 
    
    eigenvalues : np.ndarray
        The eigenvalues of the Laplacian. 
    
    coefficients : np.ndarray
        The GFT coefficients. 
    """

    eigenvalues, eigenvectors = eigenmode_decomposition(graph=graph,
                                                        matrix='laplacian',
                                                        k=None,
                                                        backend='numpy')

    # Compute Fourier coefficients if signal is provided    
    coefficients = None
    if signal is not None:
        signal = np.asarray(signal)
        coefficients = np.dot(eigenvectors.T, signal)

    return eigenvectors, eigenvalues, coefficients

def _graph_fourier_transform_torch(graph: Union[nx.Graph, FitnessLandscape],
                                   signal: np.ndarray) -> Tuple:
    """
    Helper function to compute the GFT using the torch backend.

    Parameters
    ----------
    graph : nx.Graph or FitnessLandscape
        The fitness landscape of graph representation.
    
    signal : np.ndarray or torch.Tensor
        The signal to project onto the eigenbasis of the Laplacian.
    
    Returns
    -------
    eigenvectors : np.ndarray or torch.Tensor
        The eigenvectors of the Laplacian. 
    
    eigenvalues : np.ndarray or torch.Tensor
        The eigenvalues of the Laplacian. 
    
    coefficients : np.ndarray or torch.Tensor
        The GFT coefficients. 
    """
    
    eigenvalues, eigenvectors = eigenmode_decomposition(graph=graph,
                                                        matrix='laplacian',
                                                        k=None,
                                                        backend='torch')
    
    # Sort eigenvalues and eigenvectors
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute Fourier coefficients if signal is provided
    coefficients = None
    if signal is not None:
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        coefficients = torch.matmul(eigenvectors.T, signal_tensor)
    
    return eigenvectors, eigenvalues, coefficients


def inverse_graph_fourier_transform(eigenvectors: Union[torch.Tensor, np.ndarray],
                                    coefficients: Union[torch.Tensor, np.ndarray],
                                    backend: Literal['torch', 'numpy'] = 'numpy') -> Union[torch.Tensor, np.ndarray]:
    """
    Compute inverse graph Fourier transform.
    
    Parameters
    ----------
    eigenvectors : np.ndarray or torch.Tensor
        Eigenvectors of the graph Laplacian.
    
    coefficients : np.ndarray or torch.Tensor
        GFT coefficients.
    
    backend : str, default=`numpy`
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    signal : np.ndarray or torch.Tensor
        Reconstructed signal.
    """
    if backend == 'numpy':
        return _inverse_graph_fourier_transform_numpy(eigenvectors, coefficients)
    elif backend == 'torch':
        return _inverse_graph_fourier_transform_torch(eigenvectors, coefficients)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _inverse_graph_fourier_transform_numpy(eigenvectors: np.ndarray,
                                           coefficients: np.ndarray) -> np.ndarray:
    """
    Helper function to compute the inverse GFT using the numpy backend.

    Parameters
    ----------
    eigenvectors : np.ndarray
        The eigenvectors of the Laplacian. 
    
    coefficients : np.ndarray
        The GFT coefficients. 
    
    Returns
    -------
    signal : np.ndarray
        The reconstructed signal vector.  
    """
    eigenvectors = np.asarray(eigenvectors)
    coefficients = np.asarray(coefficients)
    
    # Reconstruct signal
    signal = np.dot(eigenvectors, coefficients)
    
    return signal


def _inverse_graph_fourier_transform_torch(eigenvectors: Union[np.ndarray, torch.Tensor],
                                           coefficients: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
        Helper function to compute the inverse GFT using the torch backend.

    Parameters
    ----------
    eigenvectors : np.ndarray or torch.Tensor
        The eigenvectors of the Laplacian. 
    
    coefficients : np.ndarray or torch.Tensor
        The GFT coefficients. 
    
    Returns
    -------
    signal : torch.Tensor
        The reconstructed signal vector.  
    """
    # Convert to PyTorch tensors
    if isinstance(eigenvectors, np.ndarray):
        eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
    
    if isinstance(coefficients, np.ndarray):
        coefficients = torch.tensor(coefficients, dtype=torch.float32)
    
    # Reconstruct signal
    signal = torch.matmul(eigenvectors, coefficients)
    
    return signal
