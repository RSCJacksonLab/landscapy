import numpy as np
import networkx as nx
from typing import List, Union
import torch
from scipy import sparse as sp
from scipy.spatial import cKDTree, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .core.sequence import BaseNumpySequence, SoftSequence
from .core.landscape import FitnessLandscape

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
    
    from .embedding.soft_embedding import ESMEmbedder

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

def make_latent_geometric_graph_connected(n_latent: int = 120,
                                          d_target: int = 4,
                                          k_edges: int = 16,
                                          seed: int = None) -> nx.Graph:
    """
    Utils functio nto build a connected latent geometric graph in 2D.
    Algorithm samples positions in [0,1]^2, uses Euclidean MST to
    guarantee connectivity and adds short kNN edges greedily until each
    node reaches degree d_target

    Parameters
    ----------
    n_latent : int, default=120
        The number of latent nodes. 
    d_target : int, default=4
        The regular degree.
    k_edges : int, default=16
        K value used durring KNN edge addition. 
    seed : int, default=`None`
        The random state. 
    
    Returns
    -------
    G : nx.Graph
        The synthetic, geometric graph.
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((n_latent, 2))
    D = distance_matrix(pos, pos)
    mst = minimum_spanning_tree(D)  # SciPy returns a sparse CSR with minimal total weight
    G = nx.Graph()
    for i in range(n_latent):
        G.add_node(i, pos=pos[i])
    mst_coo = mst.tocoo()
    for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        u = int(u); v = int(v)
        G.add_edge(u, v, weight=float(w))

    # Build kNN candidate edges (short and local)
    tree = cKDTree(pos)
    candidates = set()
    for i in range(n_latent):
        dists, idxs = tree.query(pos[i], k=min(k_edges+1, n_latent))
        for j in idxs[1:]:
            u, v = (i, j) if i < j else (j, i)
            candidates.add((u, v))

    # Sort candidates by Euclidean length
    cand_sorted = sorted(candidates, key=lambda e: np.linalg.norm(pos[e[0]] - pos[e[1]]))

    # Greedily add edges until degree targets are hit
    deg = dict(G.degree())
    for u, v in cand_sorted:
        if deg[u] < d_target and deg[v] < d_target and not G.has_edge(u, v):
            w = float(np.linalg.norm(pos[u] - pos[v]))
            G.add_edge(u, v, weight=w)
            deg[u] += 1; deg[v] += 1
        if all(deg[i] >= d_target for i in G.nodes()):
            break

    if not nx.is_connected(G):
        raise RuntimeError("Latent graph unexpectedly disconnected (should not happen).")

    return G

def sample_observed_induced_connected(G_lat: nx.Graph,
                                      node_keep: float = 0.6,
                                      edge_keep: float = 0.6,
                                      seed: int = None) -> nx.Graph:
    """
    Util function to induce a connected subgraph from a latent graph.

    Parameters
    ----------
    G_lat : nx.Graph
        The latent graph to sample from. 
    node_keep : float, default=0.6
        The proportion of nodes to keep in the induced graph.
    edge_keep : float, default=0.6
        The proportion of edges to keep in the induced graph. 
    seed : int, default=`None`
        The random state seed.

    Returns
    -------
    G_obs : nx.Graph
        The connected induced graph with edge weights preserved from
        the latent graph. 
    """
    if not nx.is_connected(G_lat):
        raise ValueError("G_lat must be connected.")

    rng = np.random.default_rng(seed)
    n = G_lat.number_of_nodes()
    target = max(2, int(np.ceil(node_keep * n)))

    nodes = list(G_lat.nodes())
    start = int(rng.integers(low=0, high=n))
    start_node = nodes[start]

    visited = {start_node}
    frontier = [start_node]
    while len(visited) < target and frontier:
        u = frontier.pop(0)
        for v in G_lat.neighbors(u):
            if v not in visited:
                visited.add(v)
                frontier.append(v)
            if len(visited) >= target:
                break
    if len(visited) < target:
        remaining = [x for x in nodes if x not in visited]
        spd = nx.single_source_dijkstra_path_length(G_lat, start_node, weight='weight')
        remaining.sort(key=lambda x: spd.get(x, np.inf))
        for v in remaining:
            visited.add(v)
            if len(visited) >= target:
                break

    sub_nodes = list(visited)
    G_obs_full = G_lat.subgraph(sub_nodes).copy()

    for (u, v) in G_obs_full.edges():
        if 'weight' not in G_obs_full[u][v]:
            pu = np.asarray(G_lat.nodes[u]['pos']); pv = np.asarray(G_lat.nodes[v]['pos'])
            G_obs_full[u][v]['weight'] = float(np.linalg.norm(pu - pv))

    if G_obs_full.number_of_edges() > 0:
        mst_obs = nx.minimum_spanning_tree(G_obs_full, weight='weight')
    else:
        mst_obs = G_obs_full.copy()

    G_obs = mst_obs.copy()
    mst_edge_set = set(map(lambda e: tuple(sorted(e)), mst_obs.edges()))
    for (u, v) in G_obs_full.edges():
        key = tuple(sorted((u, v)))
        if key in mst_edge_set:
            continue
        if rng.random() < edge_keep:
            G_obs.add_edge(u, v, **G_obs_full[u][v])

    if not nx.is_connected(G_obs):
        # fallback: just return MST (connected)
        G_obs = mst_obs

    return G_obs


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

@dataclass
class HammingCheckResult:
    is_full_hamming: bool
    L: int
    n: int
    is_binary: bool
    all_same_length: bool
    has_all_genotypes: bool
    no_duplicates: bool
    graph_is_hypercube: Optional[bool]
    reason: Optional[str]
    lex_perm: Optional[np.ndarray]  # permutation to lexicographic binary order
    codes: Optional[np.ndarray]     # integer code for each sequence (in `landscape.sequences` order)

def _seq_array_to_bits(arr: np.ndarray) -> np.ndarray:
    """
    Coerce an array-like (chars or ints) of length L to 0/1 np.uint8.

    Parameters
    ----------
    arr : np.ndarray
        The input sequence array. 
    
    Returns
    -------
    The array as an 8-bit integer.
    """
    if arr.dtype.kind in ('U', 'S', 'O'):
        return np.array([0 if (x == '0' or x == 0) else 1 for x in arr], dtype=np.uint8)
    elif arr.dtype.kind in ('i','u','b'):
        x = arr.astype(np.uint8)
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Non-binary integer values present.")
        return x
    else:
        x = arr.astype(float)
        if not np.all((x == 0.0) | (x == 1.0)):
            raise ValueError("Non-binary numeric values present.")
        return x.astype(np.uint8)

def _bits_to_int_code(bits: np.ndarray) -> int:
    """
    Map a 0/1 vector (length L) to an integer code using big-endian bit
    order.

    Parameters
    ----------
    bits : np.ndarray
        The sequence bits array 
    
    Returns
    -------
    out : int
        The integer code.
    """
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out

def _compute_codes_and_L(sequences: List[BaseNumpySequence]) -> Tuple[np.ndarray, int]:
    """
    Convert all sequences to bit-vectors and integer codes. Ensures
    same length.

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of input sequences. 
    
    Returns
    -------
    Tuple
        Tuple of integer codes and sequence length.
    """
    first = sequences[0].to_array()
    L = len(first)
    codes = np.empty(len(sequences), dtype=np.int64)
    for i, seq in enumerate(sequences):
        arr = np.asarray(seq.to_array())
        if len(arr) != L:
            raise ValueError(f"Sequences have differing lengths: found {len(arr)} vs expected {L}.")
        bits = _seq_array_to_bits(arr)
        codes[i] = _bits_to_int_code(bits)
    return codes, L

def _check_graph_is_hypercube(G: nx.Graph,
                              codes: np.ndarray,
                              L: int,
                              node_to_seq_code: Dict) -> Tuple[bool, Optional[str]]:
    """
    Function to validate that the provided graph is the L-dimensional
    hypercube on the given nodes.

    Parameters
    ----------
    G : nx.Graph
        The input graph. 
    
    codes : np.ndarray
        The integer array for sequence encoding. 
    
    L : int
        The sequence length. 
    
    node_to_seq_code : Dict
        Dictionary of node index to integer sequence code. 
    
    Returns
    -------
    Tuple
        Boolean of whether fully connected Hamming graph and error 
        message if `False`.
    """
    n = len(codes)
    # Basic counts
    if G.number_of_nodes() != n:
        return False, f"Graph has {G.number_of_nodes()} nodes but sequences have {n}."

    expected_edges = (n * L) // 2
    if G.number_of_edges() != expected_edges:
        return False, f"Graph has {G.number_of_edges()} edges; expected {expected_edges} for L={L}."

    # Degree L everywhere
    for v, deg in G.degree():
        if deg != L:
            return False, f"Node {v} has degree {deg}; expected {L}."

    # Every edge flips exactly one bit
    for u, v in G.edges():
        cu = node_to_seq_code.get(u, None)
        cv = node_to_seq_code.get(v, None)
        if cu is None or cv is None:
            return False, "Graph nodes missing 'sequence' attribute mapping to a code."
        
        # XOR has Hamming weight 1 iff Hamming distance == 1
        x = cu ^ cv
        if x == 0 or (x & (x - 1)) != 0:
            return False, "Found an edge that is not Hamming distance 1."

    # Connectivity
    if not nx.is_connected(G):
        return False, "Graph is not connected."

    return True, None

def check_full_hamming(landscape: FitnessLandscape,
                       *,
                       check_graph: bool = True,
                       return_info: bool = True) -> HammingCheckResult:
    """
    Function to verify that a `FitnessLandscape` lives on a full binary
    Hamming hypercube and (optionally) that `landscape.graph` equals
    the L-dimensional hypercube.

    Parameters
    ----------
    landscape : FitnessLandscape
        The fitness landscape to check. 
    
    check_graph : bool, default=`True`
        Boolean to check graph connectivity. 
    
    return_info : bool, default=`True`
        Boolean to return `HammingCheckResult`.
    
    Returns
    -------
    HammingCheckResult
        a HammingCheckResult with diagnostics and the permutation to
        lex order.
    """
    n = len(landscape.sequences)
    reason = None

    # Sequence checks
    try:
        codes, L = _compute_codes_and_L(landscape.sequences)
        is_binary = True
        all_same_length = True
    except ValueError as e:
        # non-binary or varying lengths
        return HammingCheckResult(
            is_full_hamming=False, L=-1, n=n, is_binary=False, all_same_length=False,
            has_all_genotypes=False, no_duplicates=False, graph_is_hypercube=None,
            reason=str(e), lex_perm=None, codes=None
        )

    # uniqueness and coverage
    uniq_codes, counts = np.unique(codes, return_counts=True)
    no_duplicates = np.all(counts == 1)
    has_all = (n == (1 << L)) and (len(uniq_codes) == n)
    if not has_all:
        missing = (1 << L) - n
        reason = f"Missing genotypes or duplicates: got {n} unique={len(uniq_codes)} expected {1<<L}."
    if not no_duplicates:
        reason = "Duplicate genotypes detected."

    # permutation to lexicographic order (0..2^L-1)
    lex_perm = np.argsort(codes)

    # Graph mapping (node -> code)
    graph_ok = None
    if check_graph:
        node_to_seq_code = {}
        for node, data in landscape.graph.nodes(data=True):
            seq = data.get('sequence', None)
            if seq is None:
                graph_ok = False
                reason = "Graph node missing 'sequence' attribute."
                break
            bits = _seq_array_to_bits(np.asarray(seq.to_array()))
            node_to_seq_code[node] = _bits_to_int_code(bits)
        if graph_ok is not False:
            graph_ok, g_reason = _check_graph_is_hypercube(landscape.graph, codes, L, node_to_seq_code)
            if not graph_ok:
                reason = g_reason

    is_full = (is_binary and all_same_length and has_all and no_duplicates and ((graph_ok is True) if check_graph else True))

    return HammingCheckResult(
        is_full_hamming=is_full,
        L=L,
        n=n,
        is_binary=is_binary,
        all_same_length=all_same_length,
        has_all_genotypes=has_all,
        no_duplicates=no_duplicates,
        graph_is_hypercube=graph_ok,
        reason=reason,
        lex_perm=lex_perm,
        codes=codes
    )