import numpy as np
import networkx as nx
from typing import List, Union
import torch
from .core.sequence import BaseNumpySequence, SoftSequence
from .embedding.soft_embedding import ESMEmbedder
from ._const import ALPHABET_21, PROT_20
from cogent3 import ArrayAlignment, make_aligned_seqs, ArrayAlignment 

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
                                       device: str = None,
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
        sequences=ohe_arrays
        )
    
    return embeddings


#TODO: def reorder sequence from one alphabet to new alphabet.

def _reorder_matrix(matrix: np.ndarray,
                    matrix_alphabet: List[str] = PROT_20,
                    target_alphabet: List[str] = PROT_20) -> np.ndarray:
    """
    Helper to reorder a substitution matrix to match a target alphabet.

    Parameters
    ----------
    matrix : np.ndarray
        The original (N, N) substitution matrix.
    
    matrix_alphabet : List[str])
        The alphabet corresponding to the original matrix.
    
    target_alphabet : List[str])
        The desired alphabet order.

    Returns
    -------
    np.ndarray
        The reordered (M, M) matrix, where M is the length of
        target_alphabet.
    """
    # If no reindexing necessary return replacement matrix.
    if matrix_alphabet == target_alphabet:
        return matrix
    
    # Create a mapping from the original alphabet characters to their indices
    original_map = {aa: i for i, aa in enumerate(matrix_alphabet)}
    
    # Get the size of the target alphabet
    target_size = len(target_alphabet)
    
    # Initialize the new reordered matrix
    reordered_matrix = np.zeros((target_size, target_size), dtype=matrix.dtype)
    
    # Create a list of indices to select and reorder rows/columns from the original matrix
    try:
        remap_indices = [original_map[aa] for aa in target_alphabet]
    except KeyError as e:
        raise ValueError(
            f"Character '{e.args[0]}' from target_alphabet is not present in the "
            "substitution matrix alphabet."
        )

    # Use advanced numpy indexing to efficiently reorder the matrix
    reordered_matrix = matrix[np.ix_(remap_indices, remap_indices)]
            
    return reordered_matrix

def calculate_gapped_soft_score(aligned_seq1: np.ndarray,
                                aligned_seq2: np.ndarray,
                                q: np.ndarray,
                                gap_penalty: float = -2.0) -> float:
    """
    Computes the distance between two "soft" sequences. This function
    calculates the total expected score between two aligned sequences,
    where each position in the sequence is represented by a probability
    distribution over the alphabet.

    Parameters
    ----------
    p_seq1 : np.ndarray
        The first soft sequence, an (L, alphabet_size) array of
        probabilities. Rows must sum to 1.

    p_seq2 : np.ndarray
        The second soft sequence, an (L, alphabet_size) array of
        probabilities. Rows must sum to 1.

    q : np.ndarray
        The replacement matrix, an (alphabet_size, alphabet_size)
        array of scores. Note that q must match the sequence alphabet.

    Returns
    -------
    total_score : float
    The total alignment score.
    """

    if aligned_seq1.shape != aligned_seq2.shape:
        raise ValueError("Aligned soft sequence arrays must have the same shape.")
    
    alphabet_size = q.shape[0]
    if aligned_seq1.shape[1] != alphabet_size + 1:
        raise ValueError(
            f"Sequence array has {aligned_seq1.shape[1]} columns, but expected "
            f"{alphabet_size + 1} (alphabet + gap)."
        )

    p1_aa = aligned_seq1[:, :alphabet_size]
    p2_aa = aligned_seq2[:, :alphabet_size]
    p1_gap = aligned_seq1[:, alphabet_size]
    p2_gap = aligned_seq2[:, alphabet_size]

    expected_aa_scores = np.sum((p1_aa @ q) * p2_aa, axis=1)
    prob_aa_vs_aa = (1 - p1_gap) * (1 - p2_gap)
    prob_any_gap = p1_gap * (1 - p2_gap) + (1 - p1_gap) * p2_gap
    
    positional_scores = (expected_aa_scores * prob_aa_vs_aa) + (gap_penalty * prob_any_gap)
    
    total_score = np.sum(positional_scores)
    
    return total_score

def get_ohe_seq(sequence: Union[str, np.ndarray, torch.Tensor],
                alphabet: List = PROT_20 + ["-"]) -> np.ndarray:
    """
    Get sequence from OHE representation. 

    Parameters
    ----------
    sequence : str, np.ndarray or torch.Tensor
        The sequence to convert. 
    
    alphabet : List, default=`PROT_20`
        The alphabet. Default is the alphabetical.
    """
    if isinstance(sequence, str):
        return sequence
    elif isinstance(sequence, np.ndarray):
        if sequence.ndim == 1:
            sequence = sequence[np.newaxis, :]
        return ''.join([alphabet[np.argmax(aa)] for aa in sequence])
    elif isinstance(sequence, torch.Tensor):
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)
        return "".join([alphabet[aa.argmax().item()] for aa in sequence])
    else:
        raise ValueError("Input must be a string, numpy array, or torch tensor.")

def alignment_to_base_numpy_sequences(alignment: ArrayAlignment,
                                      alphabet: List[str] = PROT_20) -> List[BaseNumpySequence]:
    """
    Converts a cogent3 ArrayAlignment object to a list of
    `BaseNumpySequence` objects.

    Parameters
    ----------
    alignment : ArrayAlignment
        The cogent3 alingmnet object
    
    alphabet : List[str], default=`PROT_20`
        The alphabet to use for BaseNumpySequence construction.

    Returns
    -------
    sequences : List[BaseNumpySequence]
        A list of BaseNumpySequence objects with gaps removed.
        
    """
    sequences = []
    for seq in alignment.iter_seqs():

        ungapped_seq_str = str(seq).replace('-', '')

        base_numpy_seq = BaseNumpySequence(
            list(ungapped_seq_str),
            alphabet=PROT_20,
            sequence_id=seq.name
        )
        sequences.append(base_numpy_seq)
    return sequences

def moving_window_alignment(alignment: ArrayAlignment,
                            window_size: int,
                            overlap: int) -> List[ArrayAlignment]:
    """
    Splits a cogent3 ArrayAlignment object into a list of smaller
    ArrayAlignment objects using a moving window.

    Parameters
    ----------
    alignment : cogent3.core.alignment.ArrayAlignment
        The alignment to be split.
    window_size : int
        The number of sites (columns) in each window.
    overlap : int
        The number of sites to overlap between consecutive windows.

    Returns
    -------
    list
        A list of cogent3.core.alignment.ArrayAlignment objects.
    """    
    if window_size <= overlap:
        raise ValueError("Window size must be greater than the overlap.")

    alignment_items = list(alignment.named_seqs.items())
    
    # The step size is the amount to move the window forward in each iteration.
    step_size = window_size - overlap
    alignment_length = alignment.array_seqs.shape[0]
    
    windows = []
    
    for start in range(0, alignment_length - window_size + 1, step_size):
        end = start + window_size
        window = alignment_items[start:end]
        window_dict = {
            seq_id : seq for seq_id, seq in window
        }
        windows.append(make_aligned_seqs(window_dict, moltype='protein'))
        
    return windows