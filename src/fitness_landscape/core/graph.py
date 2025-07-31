import numpy as np
import networkx as nx
from typing import List, Union, Literal
from .sequence import BaseNumpySequence, sequence_distance
import gudhi
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances


def create_hamming_graph(sequences: List[BaseNumpySequence],
                         fitness_values: Union[np.ndarray, List] = None,
                         weight_by_fitness: bool = False) -> nx.Graph:
    """
    Create a Hamming graph from sequences and fitness values. In a
    Hamming graph, nodes represent sequences and edges connect
    sequences that differ by exactly one position (Hamming
    distance = 1).
    
    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    fitness_values : array-like
        Fitness values corresponding to sequences.
    weight_by_fitness : bool, default = `False`
        Whether to weight edges by fitness differences.
        
    Returns
    -------
    networkx.Graph
        Hamming graph.
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes with sequence and fitness attributes
    for i, seq in enumerate(sequences):
        if not isinstance(seq, BaseNumpySequence):
            seq = BaseNumpySequence(seq)
        
        # Add node with sequence attribute
        G.add_node(i, sequence=seq)
        
    
    # Add edges between sequences with Hamming distance = 1
    for i in range(len(sequences)):
        seq_i = sequences[i]
        for j in range(i + 1, len(sequences)):
            seq_j = sequences[j]
            
            # Calculate Hamming distance
            dist = sequence_distance(seq_i, seq_j, metric='hamming')
            
            if dist == 1:
                # Add edge with weight
                if weight_by_fitness and fitness_values is not None:
                    weight = abs(float(fitness_values[i]) - float(fitness_values[j]))
                    G.add_edge(i, j, weight=weight, distance=dist)
                else:
                    G.add_edge(i, j, weight=1.0, distance=dist)
    return G

def create_cknn_graph(sequences: List[BaseNumpySequence],
                      embeddings: np.ndarray = None,
                      k: int = 3,
                      **kwargs) -> nx.Graph:
    """
    Creates a graph using the Continuous k-Nearest Neighbors (ck-NN)
    algorithm, which is parameter-free and density adaptive.

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        Sequences to form the nodes of the graph.

    embeddings : np.ndarray, default=`None`
        A numpy array of shape (n_sequences, n_dimensions) containing
        the high-dimensional embeddings for each sequence. If `None`,
        Hamming distance matrix is computed and used.

    k : int, default=3
        The number of neighbors to use for the initial local density
        estimation. This is not a sensitive parameter, and small
        values (3-5) typically work well.

    **kwargs : dict, optional
        Additional keyword arguments for API consistency.

    Returns
    -------
    networkx.Graph
        A graph where nodes are sequence indices and edges represent
        adaptive, density-aware proximity.
    """

    n_sequences = len(sequences)
    if n_sequences < k + 1:
        raise ValueError(f"Number of sequences ({n_sequences}) must be > k ({k}).")

    if embeddings is not None:
        dist_matrix = euclidean_distances(embeddings)
    else:
        # Fallback to OHE euclidean distance if no embeddings are provided
        from ..utils import get_distance_matrix
        dist_matrix = get_distance_matrix(sequences, metric='euclidean')

    sorted_distances = np.sort(dist_matrix, axis=1)
    
    # The k-th neighbor is at index k, as index 0 is the point itself.
    sigma_k = sorted_distances[:, k]
    sigma_k[sigma_k == 0] = 1e-9  # Avoid division by zero

    sigma_product = np.outer(sigma_k, sigma_k)
    exp_term = np.exp(-dist_matrix**2 / sigma_product)
    k_continuous = exp_term.sum(axis=1)

    G = nx.Graph()
    for i, seq in enumerate(sequences):
        G.add_node(i, sequence=seq)

    k_product_matrix = np.outer(k_continuous, k_continuous)
    rows, cols = np.where(np.triu(k_product_matrix >= n_sequences, k=1))
    
    for i, j in zip(rows, cols):
        weight = dist_matrix[i, j]
        G.add_edge(i, j, weight=weight, distance=weight)

    return G


def create_knn_graph(sequences: List[BaseNumpySequence],
                     k: int,
                     metric: Literal['hamming'] = 'hamming', 
                     weight_by_distance: bool = True, **kwargs) -> nx.Graph:
    """
    Create a k-nearest neighbor graph. In a KNN graph, nodes represent
    sequences and edges connect each sequence to its k nearest
    neighbors according to the specified distance metric.
    
    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    k : int
        Number of neighbors.
    metric : str, optional
        Distance metric ('hamming') // More to add
    weight_by_distance : bool, default=`True`
        Whether to weight edges by distance.
        
    Returns
    -------
    networkx.Graph
        KNN graph.
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes with sequence and fitness attributes
    for i, seq in enumerate(sequences):
        if not isinstance(seq, BaseNumpySequence):
            seq = BaseNumpySequence(seq)
        
        # Add node with sequence attribute
        G.add_node(i, sequence=seq)
        
    
    # Calculate all pairwise distances
    n_sequences = len(sequences)
    distances = np.zeros((n_sequences, n_sequences))
    
    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):
            dist = sequence_distance(sequences[i], sequences[j], metric=metric)
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Connect each sequence to its k nearest neighbors
    for i in range(n_sequences):
        # Get indices of k nearest neighbors (excluding self)
        nearest_indices = np.argsort(distances[i])
        nearest_indices = nearest_indices[1:k+1]  # Skip self (index 0)
        
        for j in nearest_indices:
            # Add edge with weight
            if weight_by_distance:
                weight = distances[i, j]
            else:
                weight = 1.0
            
            G.add_edge(i, j, weight=weight, distance=distances[i, j])
    
    return G

def create_tda_graph(sequences: List[BaseNumpySequence],
                     embeddings: np.ndarray,
                     n_components: int = 3,
                     reweight_simplex_edges: bool = False,
                     **kwargs) -> nx.Graph:
    """
    Function to construct a graph based on persisent homology, using
    the alpha complex and dimensionality reduced embedding features.

    Parameters
    ----------
    sequences : List[BaseNumpySequences]
        Sequences to connect.
    
    embeddings : np.ndarray
        The sequence embeddings, indexed according to sequence order.

    n_components : int, default=3
        The number of principle components to use for alpha complex
        creation.
    
    reweight_simplex_edges : bool, default=`False`
        Bool to reweight graph edges by triangle simplexes.
    
    Returns
    -------
    G : nx.graph
        The constructed graph with `BaseNumpySequence` features stored
        under `sequence`.
    """
    if len(sequences) != embeddings.shape[0]:
        raise ValueError("Number of sequences must match the number of embeddings.")

    if embeddings.shape[0] == 0:
        return nx.Graph()

    # Reduce dimensionality with PCA.
    # Alpha complex scales with dimension.
    pca = PCA(n_components=n_components)
    low_dim_data = pca.fit_transform(embeddings)
    alpha_complex = gudhi.AlphaComplex(points=low_dim_data)
    simplex_tree = alpha_complex.create_simplex_tree()
    persistence_0d = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    
    # Get all finite death times for 0D features (connected components)
    finite_deaths = [p[1][1] for p in persistence_0d if p[0] == 0 and p[1][1] < float('inf')]
    
    if not finite_deaths:
        # If all points are isolated or form one component, use a small default
        chosen_alpha_square = 0.01 
    else:
        # Choose the 95th percentile of death times as a robust threshold
        chosen_alpha_square = np.percentile(finite_deaths, 95)

    alpha_complex_for_graph = gudhi.AlphaComplex(points=low_dim_data)
    simplex_tree_for_graph = alpha_complex_for_graph.create_simplex_tree(max_alpha_square=chosen_alpha_square)
    edge_generator = simplex_tree_for_graph.get_skeleton(1)

    G = nx.Graph()
    
    for i, seq in enumerate(sequences):
        G.add_node(i, sequence=seq)

    for simplex, _ in edge_generator:
        if len(simplex) == 2:
            node1, node2 = simplex[0], simplex[1]
            dist = np.linalg.norm(low_dim_data[node1] - low_dim_data[node2])
            G.add_edge(node1, node2, weight=dist, distance=dist)
            
    if reweight_simplex_edges:
        G = _reweight_graph_by_simplices(G=G,
                                         simplex_tree=simplex_tree)

    return G

def _reweight_graph_by_simplices(G: nx.Graph,
                                 simplex_tree) -> nx.Graph:
    """
    Helper function to reweight the edges of a graph based on how many
    triangles are present in the TDA.

    Parameters
    ----------
    G : nx.Graph
        The constructed network graph to reweight.
    
    simplex_tree : Any
        The 0d persistence simplex tree used to construct `G`.
    
    Returns
    -------
    G : nx.Graph
        The input network graph with updated simplex edge weights.
    """
    G_weighted = G.copy()
    
    # A dictionary to count triangle participation for each edge
    triangle_counts = {}
    
    # Iterate through all triangles in the simplex tree
    for simplex, _ in simplex_tree.get_skeleton(2):
        if len(simplex) == 3:
            # For each edge in the triangle, increment its count
            for i in range(3):
                u, v = simplex[i], simplex[(i + 1) % 3]
                # Ensure the edge is stored in a canonical order (u < v)
                edge = tuple(sorted((u, v)))
                triangle_counts[edge] = triangle_counts.get(edge, 0) + 1
    
    # Update the weights in the new graph
    for u, v in G_weighted.edges():
        edge = tuple(sorted((u, v)))
        G_weighted[u][v]['simplicial_weight'] = 1 + triangle_counts.get(edge, 0)
        
    return G_weighted


def create_diffusion_graph(sequences: List[BaseNumpySequence],
                           embeddings: np.ndarray,
                           t: int = 5,
                           k: int = 5,
                           connectivity_threshold: float = 1e-4,
                           **kwargs) -> nx.Graph:
    """
    Function to construct a graph based on expected diffusion
    behaviour in a high-dimensional embedding space. 

    Parameters
    ----------
    sequences : List[BaseNumpySequences]
        Sequences to connect.
    
    embeddings : np.ndarray
        The sequence embeddings, indexed according to sequence order.

    t : int, default=`5`
        The Markov transition matrix exponent.
    
    k : int, default=`5`
        Nearest neighbors to scale the rbf gamma parameter.
    
    connectivity_threshold : float, default=`1e-04`
        The threshold the define discrete connectivity.

    Returns
    -------
    G : nx.graph
        The constructed graph with `BaseNumpySequence` features stored
        under `sequence`.
    """
    k_for_scale = k
    if embeddings.shape[0] <= k_for_scale:
        k_for_scale = embeddings.shape[0] - 1

    nn = NearestNeighbors(n_neighbors=k_for_scale + 1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    
    # The scale for each point is the distance to its k-th neighbor
    sigma = distances[:, k_for_scale]
    median_sigma_sq = np.median(sigma[sigma > 0])**2

    if median_sigma_sq == 0:
        median_sigma_sq = 1.0
        
    gamma = 1.0 / (2 * median_sigma_sq)
    kernel_matrix = rbf_kernel(embeddings, gamma=gamma)
    
    # Create the Markov transition matrix.
    row_sums = kernel_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated points
    row_sums[row_sums == 0] = 1.0
    transition_matrix = kernel_matrix / row_sums

    # Power the matrix for diffusion process.
    diffused_matrix = np.linalg.matrix_power(transition_matrix, t)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(sequences)))
    
    # Get the upper triangle to avoid duplicate edges
    rows, cols = np.where(np.triu(diffused_matrix > connectivity_threshold, k=1))
    
    for i, j in zip(rows, cols):
        G.add_edge(i, j, weight=diffused_matrix[i, j])
        
    for i, seq in enumerate(sequences):
        G.nodes[i]['sequence'] = seq
        
    return G

