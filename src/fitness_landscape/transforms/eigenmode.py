"""
Eigenmode decomposition implementations for fitness landscape analysis.

This module provides functions for computing eigenmode decomposition of network structures,
which allows analysis of fundamental patterns in fitness landscapes.
"""

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape


def eigenmode_decomposition(graph, k=None, backend='numpy'):
    """
    Compute eigenmode decomposition of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to decompose.
    k : int or None, optional
        Number of eigenmodes to compute.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    tuple
        (eigenvalues, eigenvectors)
    """
    # Handle FitnessLandscape input
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    
    # Ensure graph is a NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX Graph or FitnessLandscape")
    
    # Compute adjacency matrix
    adjacency = nx.adjacency_matrix(graph)
    
    # Compute eigenmode decomposition based on backend
    if backend == 'numpy':
        return _eigenmode_decomposition_numpy(adjacency, k)
    elif backend == 'torch':
        return _eigenmode_decomposition_torch(adjacency, k)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _eigenmode_decomposition_numpy(adjacency, k=None):
    """Compute eigenmode decomposition using NumPy."""
    # Convert to dense matrix for eigendecomposition
    adjacency_dense = adjacency.todense()
    
    # Compute eigendecomposition
    if k is not None and k < adjacency_dense.shape[0]:
        # Use sparse eigendecomposition for efficiency
        eigenvalues, eigenvectors = sp.linalg.eigsh(adjacency, k=k, which='LM')
    else:
        # Compute full eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency_dense)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def _eigenmode_decomposition_torch(adjacency, k=None):
    """Compute eigenmode decomposition using PyTorch."""
    # Convert to dense matrix for eigendecomposition
    adjacency_dense = adjacency.todense()
    
    # Convert to PyTorch tensor
    adjacency_tensor = torch.tensor(adjacency_dense, dtype=torch.float32)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(adjacency_tensor)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Limit to k eigenvectors if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    
    return eigenvalues, eigenvectors


def reconstruct_from_eigenmodes(eigenvectors, coefficients, backend='numpy'):
    """
    Reconstruct graph from eigenmodes and coefficients.
    
    Parameters
    ----------
    eigenvectors : array-like
        Eigenvectors of the graph.
    coefficients : array-like
        Coefficients for reconstruction.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    array-like
        Reconstructed adjacency matrix.
    """
    if backend == 'numpy':
        return _reconstruct_from_eigenmodes_numpy(eigenvectors, coefficients)
    elif backend == 'torch':
        return _reconstruct_from_eigenmodes_torch(eigenvectors, coefficients)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _reconstruct_from_eigenmodes_numpy(eigenvectors, coefficients):
    """Reconstruct graph from eigenmodes using NumPy."""
    eigenvectors = np.asarray(eigenvectors)
    coefficients = np.asarray(coefficients)
    
    # Reconstruct adjacency matrix
    reconstruction = np.zeros((eigenvectors.shape[0], eigenvectors.shape[0]))
    
    for i, coef in enumerate(coefficients):
        outer_product = np.outer(eigenvectors[:, i], eigenvectors[:, i])
        reconstruction += coef * outer_product
    
    return reconstruction


def _reconstruct_from_eigenmodes_torch(eigenvectors, coefficients):
    """Reconstruct graph from eigenmodes using PyTorch."""
    # Convert to PyTorch tensors
    eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    
    # Reconstruct adjacency matrix
    reconstruction = torch.zeros((eigenvectors.shape[0], eigenvectors.shape[0]))
    
    for i, coef in enumerate(coefficients):
        outer_product = torch.outer(eigenvectors[:, i], eigenvectors[:, i])
        reconstruction += coef * outer_product
    
    return reconstruction


def eigenmode_analysis(graph, k=None, backend='numpy'):
    """
    Analyze eigenmodes of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to analyze.
    k : int or None, optional
        Number of eigenmodes to analyze.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    dict
        Analysis results including eigenvalues, participation ratios,
        localization metrics, and node centralities.
    """
    # Compute eigenmode decomposition
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, k=k, backend=backend)
    
    # Handle FitnessLandscape input
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    
    # Compute analysis metrics based on backend
    if backend == 'numpy':
        return _eigenmode_analysis_numpy(graph, eigenvalues, eigenvectors)
    elif backend == 'torch':
        return _eigenmode_analysis_torch(graph, eigenvalues, eigenvectors)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _eigenmode_analysis_numpy(graph, eigenvalues, eigenvectors):
    """Analyze eigenmodes using NumPy."""
    n_nodes = eigenvectors.shape[0]
    n_modes = eigenvectors.shape[1]
    
    # Initialize results
    results = {
        'eigenvalues': eigenvalues,
        'participation_ratios': np.zeros(n_modes),
        'localization': np.zeros(n_modes),
        'node_centralities': np.zeros((n_nodes, n_modes))
    }
    
    # Compute participation ratio and localization for each mode
    for i in range(n_modes):
        # Get eigenvector
        eigenvector = eigenvectors[:, i]
        
        # Compute participation ratio
        # PR = (sum(psi_i^2))^2 / sum(psi_i^4)
        # Higher values indicate more delocalized modes
        psi_squared = eigenvector**2
        participation_ratio = np.sum(psi_squared)**2 / np.sum(psi_squared**2)
        results['participation_ratios'][i] = participation_ratio
        
        # Compute localization (inverse participation ratio)
        # IPR = sum(psi_i^4) / (sum(psi_i^2))^2
        # Higher values indicate more localized modes
        results['localization'][i] = 1.0 / participation_ratio
        
        # Compute node centralities for this mode
        results['node_centralities'][:, i] = np.abs(eigenvector)
    
    # Compute spectral gap
    if n_modes >= 2:
        results['spectral_gap'] = eigenvalues[0] - eigenvalues[1]
    
    # Compute spectral density
    hist, bin_edges = np.histogram(eigenvalues, bins=min(20, n_modes))
    results['spectral_density'] = {
        'histogram': hist,
        'bin_edges': bin_edges
    }
    
    return results


def _eigenmode_analysis_torch(graph, eigenvalues, eigenvectors):
    """Analyze eigenmodes using PyTorch."""
    n_nodes = eigenvectors.shape[0]
    n_modes = eigenvectors.shape[1]
    
    # Convert to NumPy for compatibility with NetworkX
    eigenvalues_np = eigenvalues.cpu().numpy()
    eigenvectors_np = eigenvectors.cpu().numpy()
    
    # Initialize results
    results = {
        'eigenvalues': eigenvalues_np,
        'participation_ratios': np.zeros(n_modes),
        'localization': np.zeros(n_modes),
        'node_centralities': np.zeros((n_nodes, n_modes))
    }
    
    # Compute participation ratio and localization for each mode
    for i in range(n_modes):
        # Get eigenvector
        eigenvector = eigenvectors[:, i]
        
        # Compute participation ratio
        # PR = (sum(psi_i^2))^2 / sum(psi_i^4)
        # Higher values indicate more delocalized modes
        psi_squared = eigenvector**2
        participation_ratio = (torch.sum(psi_squared)**2 / torch.sum(psi_squared**2)).item()
        results['participation_ratios'][i] = participation_ratio
        
        # Compute localization (inverse participation ratio)
        # IPR = sum(psi_i^4) / (sum(psi_i^2))^2
        # Higher values indicate more localized modes
        results['localization'][i] = 1.0 / participation_ratio
        
        # Compute node centralities for this mode
        results['node_centralities'][:, i] = torch.abs(eigenvector).cpu().numpy()
    
    # Compute spectral gap
    if n_modes >= 2:
        results['spectral_gap'] = (eigenvalues[0] - eigenvalues[1]).item()
    
    # Compute spectral density
    hist, bin_edges = np.histogram(eigenvalues_np, bins=min(20, n_modes))
    results['spectral_density'] = {
        'histogram': hist,
        'bin_edges': bin_edges
    }
    
    return results


def project_signal_on_eigenmodes(graph, signal, k=None, backend='numpy'):
    """
    Project a signal onto the eigenmodes of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph whose eigenmodes to use.
    signal : array-like
        Signal to project.
    k : int or None, optional
        Number of eigenmodes to use.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    tuple
        (eigenvalues, eigenvectors, projection_coefficients)
    """
    # Compute eigenmode decomposition
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, k=k, backend=backend)
    
    # Project signal onto eigenmodes
    if backend == 'numpy':
        signal = np.asarray(signal)
        projection = np.dot(eigenvectors.T, signal)
    elif backend == 'torch':
        signal = torch.tensor(signal, dtype=torch.float32)
        projection = torch.matmul(eigenvectors.T, signal)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return eigenvalues, eigenvectors, projection
