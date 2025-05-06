import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape, DirectedFitnessLandscape


def directed_eigenmode_decomposition(directed_landscape: DirectedFitnessLandscape,
                                     k: int = None,
                                     matrix: Literal['sym_laplacian', 'sym_transition'] = 'sym_laplacian',
                                     backend: Literal['numpy', 'torch'] = 'numpy'):
    """
    Compute eigenmode decomposition of a directed graph. Matrices
    must be symmetrical.
    
    Parameters
    ----------
    directed_landscape : DirectedFitnessLandscape
        Directed graph to decompose.
    matrix : str, default = `sym_laplacian`
        The graph matrix to decompose. Either symmetric Laplacian
        matrix or the symmetric transition matrix. 
    k : int or None, optional
        Number of eigenmodes to compute.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    tuple
        (eigenvalues, eigenvectors)
    """    
    # Ensure graph is a NetworkX graph
    if not isinstance(directed_landscape, DirectedFitnessLandscape):
        raise TypeError("Landscape must be a `DirectedFitnessLandscape`")
    
    if matrix == 'sym_laplacian':
    # Compute adjacency matrix
        eig_mat = directed_landscape.directed_laplacian
    
    elif matrix == 'sym_transition':
        eig_mat = directed_landscape.transition_matrix

    else:
        raise ValueError(f"Unsupported matrix: {matrix}")
    
    # Compute eigenmode decomposition based on backend
    if backend == 'numpy':
        return _eigenmode_decomposition_numpy(eig_mat, k)
    elif backend == 'torch':
        return _eigenmode_decomposition_torch(eig_mat, k)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def eigenmode_decomposition(graph: Union[nx.Graph, FitnessLandscape],
                            k: int = None,
                            matrix: Literal['adjacency', 'laplacian', 'transition'] = 'laplacian',
                            backend: Literal['numpy', 'torch'] = 'numpy'):
    """
    Compute eigenmode decomposition of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to decompose.
    matrix : str, default = `laplacian`
        The graph matrix to decompose. Either Laplacian matrix or the
        adjacency matrix. 
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
    
    if matrix == 'adjacency':
    # Compute adjacency matrix
        eig_mat = nx.adjacency_matrix(graph)
    
    elif matrix == 'laplacian':
        eig_mat = nx.laplacian_matrix(graph)
    
    # Row stochastic Markov transition matrix.
    elif matrix == 'transition':
        A = nx.to_numpy_array(graph)
        degrees = A.sum(axis=1)
        degrees[degrees == 0] = 1
        eig_mat = A / degrees[:, None]

    else:
        raise ValueError(f"Unsupported matrix: {matrix}")
    
    # Compute eigenmode decomposition based on backend
    if backend == 'numpy':
        return _eigenmode_decomposition_numpy(eig_mat, k)
    elif backend == 'torch':
        return _eigenmode_decomposition_torch(eig_mat, k)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _eigenmode_decomposition_numpy(eig_mat, k=None) -> Tuple:
    """
    Helper function for eigenmode matrix decomposition using a numpy
    backend. 

    Parameters
    ----------
    eig_mat : np.ndarray
        The matrix to decompose.
    
    k : int
        The number of eigenmodes to factor.
    
    Returns
    -------
    eigenvalues : np.ndarray
        The eig_mat eigenvalues.
    
    eigenvectors : np.ndarray
        The eig_max eigenvectors.
    """
    # Convert to dense matrix for eigendecomposition
    mat_dense = eig_mat.todense()
    
    # Compute eigendecomposition
    if k is not None and k < mat_dense.shape[0]:
        # Use sparse eigendecomposition for efficiency
        eigenvalues, eigenvectors = sp.linalg.eigsh(eig_mat,
                                                    k=k,
                                                    which='LM')
    else:
        # Compute full eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(mat_dense)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def _eigenmode_decomposition_torch(eig_mat: Union[np.ndarray, torch.Tensor],
                                   k: int=None,
                                   return_torch: bool = True) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:

    """
    Helper function to perform spectral decomposition usingthe torch
    backend. 

    Parameters
    ----------
    eig_mat: np.ndarray or torch.Tensor
        The matrix to decompose. 
    
    k: int
        The number of eigenmodes to compute.
    
    return_torch: bool, default=`True`
        Boolean to return eigenvectors / eigenvalues a tensor.
    
    Returns
    -------
    eigenvalues : np.ndarray or torch.Tensor
        The eig_matx eigenvalues
    
    eigenvectors : np.ndarray or torch.Tensor
        The eig_mat eigenvectors.
    """
    if isinstance(eig_mat, np.ndarray):    

        # Convert to dense matrix for eigendecomposition
        mat_dense = eig_mat.todense()
        # Convert to PyTorch tensor
        mat_tensor = torch.tensor(mat_dense, dtype=torch.float32)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(mat_tensor)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Limit to k eigenvectors if specified
    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
    
    if not return_torch:

        eigenvalues = eigenvalues.numpy()
        eigenvectors = eigenvectors.numpy()

    return eigenvalues, eigenvectors

def reconstruct_from_eigenmodes(eigenvectors: Union[np.ndarray, torch.Tensor],
                                coefficients: Union[np.ndarray, torch.Tensor],
                                backend: Literal['numpy', 'torch'] = 'numpy') -> Union[np.ndarray, torch.Tensor]:
    """
    Function to reconstruct graph from eigenmodes and coefficients.
    
    Parameters
    ----------
    eigenvectors : np.ndarray or torch.Tensor
        Eigenvectors of the graph.
    coefficients : np.ndarray, torch.Tensor
        Coefficients for reconstruction.
    backend : str
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    array-like
        Reconstructed matrix.
    """
    if backend == 'numpy':
        return _reconstruct_from_eigenmodes_numpy(eigenvectors, coefficients)
    elif backend == 'torch':
        return _reconstruct_from_eigenmodes_torch(eigenvectors, coefficients)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _reconstruct_from_eigenmodes_numpy(eigenvectors: np.ndarray,
                                       coefficients: np.ndarray) -> np.ndarray:
    """
    Helper function to reconstruct a matrix from the the eigenvectors
    and coefficients using the numpy backend. 

    Parameters
    ----------
    eigenvectors : np.ndarray
        The matrix eigenvectors. 
    
    coefficients : np.ndarray
        The matrix eigenvalues. 
    
    Returns
    -------
    reconstruction : np.ndarray
        The reconstructed matrix. 
    """
    eigenvectors = np.asarray(eigenvectors)
    coefficients = np.asarray(coefficients)
    
    # Reconstruct adjacency matrix
    reconstruction = np.zeros((eigenvectors.shape[0], eigenvectors.shape[0]))
    
    for i, coef in enumerate(coefficients):
        outer_product = np.outer(eigenvectors[:, i], eigenvectors[:, i])
        reconstruction += coef * outer_product
    
    return reconstruction


def _reconstruct_from_eigenmodes_torch(eigenvectors: Union[np.ndarray, torch.Tensor],
                                       coefficients: Union[np.ndarray, torch.Tensor],
                                       return_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Helper function to reconstruct a matrix from eigenvectors and
    eigenvectors using the torch backend.

    Parameters
    ----------
    eigenvectors : np.ndarray or torch.Tensor
        The eigenvectors.
    
    eigenvalues : np.ndarray or torch.Tensor
        The eigenvalues. 
    
    Returns
    -------
    reconstruction : torch.Tensor or np.ndarray
        The reconstructed matrix. 
    """
    
    if isinstance(eigenvectors, np.ndarray):
    # Convert to PyTorch tensors
        eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)
    
    if isinstance(coefficients, np.ndarray):
        coefficients = torch.tensor(coefficients, dtype=torch.float32)
    
    # Reconstruct adjacency matrix
    reconstruction = torch.zeros((eigenvectors.shape[0], eigenvectors.shape[0]))
    
    for i, coef in enumerate(coefficients):
        outer_product = torch.outer(eigenvectors[:, i], eigenvectors[:, i])
        reconstruction += coef * outer_product
    
    if not return_torch:
        reconstruction = reconstruction.numpy()

    return reconstruction

def graph_spectral_analysis(graph: Union[nx.Graph, FitnessLandscape],
                            k: int = None,
                            matrix: Literal['adjacency', 'laplacian'] = 'laplacian',
                            backend: Literal['numpy', 'torch'] = 'numpy') -> Dict:
    """
    Analyze the eigenmodes of a graph.
    
    Parameters
    ----------
    graph : networkx.Graph or FitnessLandscape
        Graph to analyze.
    k : int or None, optional
        Number of eigenmodes to analyze.
    matrix : str, default = `laplacian`
        The matrix to decompose.
    backend : str, default = `numpy`
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    dict
        Eigenspectral analysis results. 
    """
    # Compute eigenmode decomposition
    eigenvalues, eigenvectors = eigenmode_decomposition(graph, matrix=matrix, k=k, backend=backend)
    
    # Handle FitnessLandscape input
    if isinstance(graph, FitnessLandscape):
        graph = graph.graph
    
    # Compute analysis metrics based on backend
    if backend == 'numpy':
        return _eigenmode_analysis_numpy(eigenvalues, eigenvectors)
    elif backend == 'torch':
        return _eigenmode_analysis_torch(eigenvalues, eigenvectors)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _eigenmode_analysis_numpy(eigenvalues: np.ndarray,
                              eigenvectors: np.ndarray) -> Dict:
    """
    Helper function to analyze eigenmodes using numpy backend.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues. 
    
    eigenvectors : np.ndarray
        The eigenvectors. 
    
    Returns
    -------
    results : Dict
        The results dict of eigenmode analysis. 
    """
    n_nodes = eigenvectors.shape[0]
    n_modes = eigenvectors.shape[1]
    
    # Initialize results
    results = {
        'eigenvalues': eigenvalues,
        'participation_ratios': np.zeros(n_modes),
        'localization': np.zeros(n_modes),
        'node_centralities': np.zeros((n_nodes, n_modes)),
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

def _eigenmode_analysis_torch(eigenvalues: Union[np.ndarray, torch.Tensor],
                              eigenvectors: Union[np.ndarray, torch.Tensor]) -> Dict:
    """
    Helper function to analyze eigenmodes using torch backend.

    Parameters
    ----------
    eigenvalues : np.ndarray or torch.Tensor
        The eigenvalues. 
    
    eigenvectors : np.ndarray or torch.Tensor
        The eigenvectors. 
    
    Returns
    -------
    results : Dict
        The results dict of eigenmode analysis. 
    """
    n_nodes = eigenvectors.shape[0]
    n_modes = eigenvectors.shape[1]
    
    # Convert to NumPy for compatibility with NetworkX
    if isinstance(eigenvalues, torch.Tensor):
        eigenvalues_np = eigenvalues.cpu().numpy()
    if isinstance(eigenvectors, torch.Tensor):
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