import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.graph import create_hamming_graph
from ..core.sequence import sequence_distance, Sequence
from logging import Logger
from .eigenmode import eigenmode_decomposition

def calculate_ruggedness_dirichlet_energy(landscape: FitnessLandscape,
                                          edge_weight_bins: Union[np.ndarray, List] = None,
                                          fitness_bins: Union[np.ndarray, List] = None,
                                          normalize: bool = True) -> Dict:
    """
    Function to determine the analytical dirichlet energy of a fitness
    landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    edge_weight_bins : array-like, default=`None`
        The edge weight bins to collect edges into. If `None`,
        Dirichlet energy is determined as the global energy without
        edge contribution pooling. 
    
    fitness_bins : array-like, default=`None`
        The fitness value bin to collect nodes into. it `None`,
        Dirichlet energy is determined as the global with fitness
        contribution pooling. 

    normalise : bool, default=`True`
        Boolean to normalise the global energy to the number of edges
        in the graph. 
    
    Returns
    -------
    results : Dict
        The Dirichlet energy results dict.
    """

def _collect_edges(landscape: Union[FitnessLandscape, nx.Graph],
                   bins: Union[np.ndarray, List]) -> Dict:
    """
    Helper function to collect edges into distance bins. 

    Parameters
    ----------
    landscape : FitnessLandscape or nx.Graph
        The fitness landscape to analyze.
    
    bins : array-like
        The edge-weight bins to collect edges into.
    
    Returns
    -------
    Dict
        The dictionary of bins containing the edge indices.

    """

def _collect_nodes(landscape: Union[FitnessLandscape, nx.Graph],
                   bins: Union[np.ndarray, List]) -> Dict:
    """
    Helper function to collect nodes into fitness bins. 

    Parameters
    ----------
    landscape : FitnessLandscape or nx.Graph
        The fitness landscape to analyze.
    
    bins : array-like
        The fitness bins to collect edges into.
    
    Returns
    -------
    Dict
        The dictionary of bin labels containing the node indices.
    """

def calculate_local_dirichlet_energy(landscape: FitnessLandscape,
                                     sequence: Union[Sequence, List] = None) -> None:
    """
    Function to determine the local Dirichlet energy of a node.
    Energies are appended as node attributes in the landscape graph.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze. 
    
    sequences : Sequence or list, default=`None`
        The sequence/s to limit the analysis to. 
    """