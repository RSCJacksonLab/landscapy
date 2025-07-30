import numpy as np
import networkx as nx
from typing import List
import torch
from .core.sequence import BaseNumpySequence, SoftSequence
from .embedding.soft_embedding import ESMEmbedder


def cosine_similarity_matrix(A, B):
    """
    Computes cosine similarity between two matrices of vectors.

    Parameters
    ----------
    A : np.ndarray
        First matrix of shape (m, d) where m is the number of vectors
        and d is the dimension.
    B : np.ndarray
        Second matrix of shape (n, d) where n is the number of vectors
        and d is the dimension.

    Returns
    -------
    np.ndarray
        Cosine similarity matrix of shape (m, n) where the entry at
        (i, j) is the cosine similarity between the i-th vector in
        A and the j-th vector in B.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def get_landscape_dist_mat(landscape: 'FitnessLandscape',
                           weighted: bool = False) -> np.ndarray:
    """
    Compute the distance matrix for a fitness landscape.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to analyze.
.
    weighted : bool, default=`False`
        Whether to use weighted edges in the graph representation.

    Returns
    -------
    dist_mat : np.ndarray
        The distance matrix for the fitness landscape.
    """

    if weighted:
        dist_mat = nx.floyd_warshall_numpy(landscape.graph, 
                                            weight='weight')
    else:
        dist_mat = nx.floyd_warshall_numpy(landscape.graph,
                                            weight=None)

    return dist_mat


def _compute_embeddings_from_sequences(sequences: List[BaseNumpySequence],
                                       model_name: str = 'facebook/esm2_t6_8M_UR50D',
                                       batch_size: int = 64) -> np.ndarray:
    """
    Function to compute soft node embeddings from a list of sequnce
    objects. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of sequences to embed.
    
    model_name : str, default=`facebook/esm2_t6_8M_UR50D`
        The embedding model huggingface repository.
    
    batch_size : int, default=`64`
        The batch size. 
    
    Returns
    -------
    embeddings : np.ndarray
        Array of embedded (soft) sequences.
    """
    
    ohe_arrays = []
    for seq in sequences:
        if isinstance(seq, SoftSequence):

            # For SoftSequence, the posterior is the OHE
            ohe_arrays.append(seq.posterior)
        else:

            # For standard sequences, generate the OHE
            ohe_arrays.append(seq.to_one_hot())
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedder = ESMEmbedder(model_name=model_name,
                           device=device,
                           batch_size=batch_size)
    
    embeddings = embedder.embed_relaxed_seqs(
        relaxed_seqs=ohe_arrays)
    
    return embeddings