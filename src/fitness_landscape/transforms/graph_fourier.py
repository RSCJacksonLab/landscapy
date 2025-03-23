"""
Graph Fourier transform implementations for fitness landscape analysis.

This module provides functions for computing graph Fourier transforms of fitness landscapes,
which allow spectral analysis of functions defined on graphs.
"""

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape


def graph_fourier_transform(graph, signal=None, k=None, backend='numpy'):
    """
    Compute graph Fourier transform of a signal on the graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph representation or fitness landscape.
    signal : array-like or None, optional
        Signal on the graph (e.g., fitness values). If None and graph is a FitnessLandscape,
        fitness values are used.
    k : int or None, optional
        Number of eigenvectors to use. If None, use all eigenvectors.
    backend : str, optional
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


def _graph_fourier_transform_numpy(graph, signal, k=None):
    """Compute graph Fourier transform using NumPy."""
    # Compute graph Laplacian
    laplacian = nx.normalized_laplacian_matrix(graph)
    
    # Convert to dense matrix for eigendecomposition
    laplacian_dense = laplacian.todense()
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Limit to k eigenvectors if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    
    # Compute Fourier coefficients if signal is provided
    coefficients = None
    if signal is not None:
        signal = np.asarray(signal)
        coefficients = np.dot(eigenvectors.T, signal)
    
    return eigenvectors, eigenvalues, coefficients


def _graph_fourier_transform_torch(graph, signal, k=None):
    """Compute graph Fourier transform using PyTorch."""
    # Compute graph Laplacian
    laplacian = nx.normalized_laplacian_matrix(graph)
    
    # Convert to dense matrix for eigendecomposition
    laplacian_dense = laplacian.todense()
    
    # Convert to PyTorch tensor
    laplacian_tensor = torch.tensor(laplacian_dense, dtype=torch.float32)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_tensor)
    
    # Sort eigenvalues and eigenvectors
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Limit to k eigenvectors if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    
    # Compute Fourier coefficients if signal is provided
    coefficients = None
    if signal is not None:
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        coefficients = torch.matmul(eigenvectors.T, signal_tensor)
    
    return eigenvectors, eigenvalues, coefficients


def inverse_graph_fourier_transform(eigenvectors, coefficients, backend='numpy'):
    """
    Compute inverse graph Fourier transform.
    
    Parameters
    ----------
    eigenvectors : array-like
        Eigenvectors of the graph Laplacian.
    coefficients : array-like
        Fourier coefficients.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    array-like
        Reconstructed signal.
    """
    if backend == 'numpy':
        return _inverse_graph_fourier_transform_numpy(eigenvectors, coefficients)
    elif backend == 'torch':
        return _inverse_graph_fourier_transform_torch(eigenvectors, coefficients)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _inverse_graph_fourier_transform_numpy(eigenvectors, coefficients):
    """Compute inverse graph Fourier transform using NumPy."""
    eigenvectors = np.asarray(eigenvectors)
    coefficients = np.asarray(coefficients)
    
    # Reconstruct signal
    signal = np.dot(eigenvectors, coefficients)
    
    return signal


def _inverse_graph_fourier_transform_torch(eigenvectors, coefficients):
    """Compute inverse graph Fourier transform using PyTorch."""
    # Convert to PyTorch tensors
    eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    
    # Reconstruct signal
    signal = torch.matmul(eigenvectors, coefficients)
    
    return signal


def laplacian_eigenvectors(graph, k=None, backend='numpy'):
    """
    Compute eigenvectors of the graph Laplacian.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph representation or fitness landscape.
    k : int or None, optional
        Number of eigenvectors to compute. If None, compute all eigenvectors.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    tuple
        (eigenvectors, eigenvalues)
    """
    # Handle FitnessLandscape input
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    
    # Ensure graph is a NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph or FitnessLandscape")
    
    # Compute eigenvectors based on backend
    if backend == 'numpy':
        return _laplacian_eigenvectors_numpy(graph, k)
    elif backend == 'torch':
        return _laplacian_eigenvectors_torch(graph, k)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _laplacian_eigenvectors_numpy(graph, k=None):
    """Compute Laplacian eigenvectors using NumPy."""
    # Compute graph Laplacian
    laplacian = nx.normalized_laplacian_matrix(graph)
    
    # Convert to dense matrix for eigendecomposition
    laplacian_dense = laplacian.todense()
    
    # Compute eigendecomposition
    if k is not None and k < laplacian_dense.shape[0]:
        # Use sparse eigendecomposition for efficiency
        eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian, k=k, which='SM')
    else:
        # Compute full eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_dense)
    
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvectors, eigenvalues


def _laplacian_eigenvectors_torch(graph, k=None):
    """Compute Laplacian eigenvectors using PyTorch."""
    # Compute graph Laplacian
    laplacian = nx.normalized_laplacian_matrix(graph)
    
    # Convert to dense matrix for eigendecomposition
    laplacian_dense = laplacian.todense()
    
    # Convert to PyTorch tensor
    laplacian_tensor = torch.tensor(laplacian_dense, dtype=torch.float32)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_tensor)
    
    # Sort eigenvalues and eigenvectors
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Limit to k eigenvectors if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    
    return eigenvectors, eigenvalues


def filter_graph_signal(graph, signal, filter_function, k=None, backend='numpy'):
    """
    Apply spectral filter to graph signal.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph representation or fitness landscape.
    signal : array-like
        Signal on the graph (e.g., fitness values).
    filter_function : callable
        Function that takes eigenvalues and returns filter coefficients.
    k : int or None, optional
        Number of eigenvectors to use. If None, use all eigenvectors.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    array-like
        Filtered signal.
    """
    # Compute graph Fourier transform
    eigenvectors, eigenvalues, coefficients = graph_fourier_transform(
        graph, signal, k=k, backend=backend
    )
    
    # Apply filter
    if backend == 'numpy':
        filtered_coefficients = coefficients * filter_function(eigenvalues)
    elif backend == 'torch':
        filtered_coefficients = coefficients * torch.tensor(filter_function(eigenvalues))
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    # Compute inverse transform
    filtered_signal = inverse_graph_fourier_transform(
        eigenvectors, filtered_coefficients, backend=backend
    )
    
    return filtered_signal
