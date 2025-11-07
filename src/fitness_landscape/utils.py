import math
import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Sequence, Callable
import torch
from scipy import sparse as sp
from scipy.spatial import cKDTree, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .core.sequence import BaseNumpySequence, SoftSequence
from dataclasses import dataclass
from softalign.soft_alignment import align_soft_sequences
from ._const import ALPHABET_21, PROT_20
from cogent3.core.alignment import Alignment, make_aligned_seqs
from cogent3 import load_aligned_seqs
from pathlib import Path
from .core.sequence import BaseNumpySequence

def sanitize_alignment(aln: Alignment,
                       *,
                       legal_amino_acids: list[str] | None = None,
                       gap_char: str = '-') -> Alignment:
    """
    Sanitize an aligned protein FASTA by enforcing only canonical 20 amino acids
    and the gap character. Any illegal or unknown symbol is replaced with a gap.

    Parameters
    ----------
    aln : Alignment
        The input alignment (protein sequences expected).
    legal_amino_acids : list[str], optional
        List of allowed amino acids. Defaults to the canonical 20.
    gap_char : str, default='-'
        The gap character to enforce. '.' is treated as a gap.

    Ensures sequence IDs are unique by appending a numeric suffix when duplicates are found.

    Returns
    -------
    Alignment
        A new alignment object with illegal symbols replaced by gaps and all
        residue letters uppercased.
    """
    if legal_amino_acids is None:
        legal_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    legal = set(legal_amino_acids)
    # include gap
    legal_with_gap = legal | {gap_char}

    def _clean_char(ch: str) -> str:
        if ch == '.':
            return gap_char
        up = ch.upper()
        return up if up in legal_with_gap else gap_char

    # Build dict unique_name -> cleaned sequence
    cleaned = {}
    used_names: set[str] = set()

    def _unique_name(name: str) -> str:
        base = str(name).strip() or 'seq'
        base = base.replace(' ', '_')
        if base not in used_names:
            used_names.add(base)
            return base
        i = 1
        while f"{base}_{i}" in used_names:
            i += 1
        new = f"{base}_{i}"
        used_names.add(new)
        return new

    for name in aln.names:
        s = str(aln.get_gapped_seq(name))
        unique = _unique_name(name)
        cleaned[unique] = ''.join(_clean_char(c) for c in s)

    return make_aligned_seqs(cleaned, moltype='protein')

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

def geodesic_distance_matrix(G: Union['FitnessLandscape', nx.Graph],
                             nodes: Optional[Sequence] = None,
                             *,
                             weight_key: Optional[str] = None,
                             transform: Union[str, Callable[[float], float], None] = "auto",
                             default_weight: float = 1.0,
                             eps: float = 1e-12) -> Tuple[np.ndarray, List]:
    """
    Compute a dense geodesic distance matrix over ``nodes`` in ``G``.

    Parameters
    ----------
    G : FitnessLandscape or nx.Graph
        Input graph (directed graphs are treated as undirected).
    nodes : Sequence, optional
        Node identifiers to include. Defaults to all nodes in ``G``.
    weight_key : str, optional
        Edge attribute containing similarities/weights. When ``None``,
        unit length edges are assumed.
    transform : {'auto', 'neglog', 'inverse', 'identity'} or callable, optional
        Transform applied to edge weights before computing geodesics.
        ``'auto'`` chooses ``'neglog'`` when ``weight_key=='kernel_weight'``
        and ``'identity'`` otherwise. A callable may also be provided.
    default_weight : float, default=1.0
        Value used when an edge lacks ``weight_key``.
    eps : float, default=1e-12
        Numerical floor used by logarithmic/inverse transforms.

    Returns
    -------
    tuple
        Pair ``(D, order)`` where ``D`` is an ``(n, n)`` numpy array of
        geodesic distances and ``order`` records the node labels used.
    """
    graph_attr = getattr(G, "graph", None)
    if isinstance(graph_attr, nx.Graph):
        graph_obj = graph_attr
    else:
        graph_obj = G
    if isinstance(graph_obj, nx.DiGraph):
        graph_obj = graph_obj.to_undirected()
    if not isinstance(graph_obj, nx.Graph):
        raise TypeError("Expected FitnessLandscape or networkx Graph/Digraph.")

    if nodes is None:
        node_list = list(graph_obj.nodes())
    else:
        node_list = [n for n in nodes if n in graph_obj]

    H = graph_obj.copy()
    length_attr = "__geodesic_length__"

    def _transform_weight(w: float) -> float:
        if callable(transform):
            return float(transform(w))
        mode = "identity" if transform is None else str(transform).lower()
        if mode == "auto":
            mode = "neglog" if weight_key == "kernel_weight" else "identity"
        if mode == "neglog":
            return float(-math.log(max(w, eps)))
        if mode in {"inverse", "reciprocal"}:
            return float(1.0 / max(w, eps))
        if mode in {"identity", "none"}:
            return float(w)
        raise ValueError(f"Unsupported transform '{transform}'.")

    for u, v, data in H.edges(data=True):
        if weight_key is None:
            w = float(default_weight)
        else:
            w = float(data.get(weight_key, default_weight))
        data[length_attr] = _transform_weight(w)

    n = len(node_list)
    D = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        D[i, i] = 0.0

    index = {node: i for i, node in enumerate(node_list)}
    for src in node_list:
        if src not in H:
            continue
        dist = nx.single_source_dijkstra_path_length(H, src, weight=length_attr)
        i = index[src]
        for tgt, val in dist.items():
            j = index.get(tgt)
            if j is not None:
                D[i, j] = float(val)

    return D, node_list


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
                                      seed: int = None,
                                      *, 
                                      return_graph: bool = False,) -> Union[nx.Graph, 'FitessLandscape']:
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
    # Dependency injection is the only option.. 
    from .core import FitnessLandscape
    
    if isinstance(G_lat, nx.Graph):
        L = None
        G_lat = G_lat

    # If not graph, must be FitnessLanscape // avoid import as leads to partially init module.        
    else:
        L = G_lat
        G_lat = L.graph


    if not nx.is_connected(G_lat):
        raise ValueError("G_lat must be connected.")

    rng = np.random.default_rng(seed)
    n = G_lat.number_of_nodes()
    target = max(2, int(np.ceil(node_keep * n)))

    nodes = list(G_lat.nodes())
    start_node = nodes[int(rng.integers(low=0, high=n))]

    # BFS-like growth until we hit the target number of nodes
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

    # If still short, greedily add nearest-by-shortest-path nodes
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

    # Ensure every sampled edge has a weight if possible
    for (u, v) in G_obs_full.edges():
        if 'weight' not in G_obs_full[u][v]:
            pu = G_lat.nodes[u].get('pos', None)
            pv = G_lat.nodes[v].get('pos', None)
            if pu is not None and pv is not None:
                pu = np.asarray(pu); pv = np.asarray(pv)
                G_obs_full[u][v]['weight'] = float(np.linalg.norm(pu - pv))

    # Keep MST edges to guarantee connectivity, then thin remnant edges
    if G_obs_full.number_of_edges() > 0:
        mst_obs = nx.minimum_spanning_tree(G_obs_full, weight='weight')
    else:
        mst_obs = G_obs_full.copy()

    G_obs = mst_obs.copy()
    mst_edge_set = set(tuple(sorted(e)) for e in mst_obs.edges())
    for (u, v) in G_obs_full.edges():
        key = tuple(sorted((u, v)))
        if key in mst_edge_set:
            continue
        if rng.random() < edge_keep:
            G_obs.add_edge(u, v, **G_obs_full[u][v])

    if not nx.is_connected(G_obs):
        # Fallback: MST is connected by construction
        G_obs = mst_obs

    # If the caller gave us a Landscape, annotate nodes with layer values and
    if L is not None:
        seq_to_idx = {tuple(seq.to_array()): i for i, seq in enumerate(L.sequences)}
        numeric_layers = []
        layer_arrays = {}
        for lname, layer in L.fitness_layers.items():
            if getattr(layer, "dtype", None) == "numeric":
                numeric_layers.append(lname)
                layer_arrays[lname] = layer.to_scalar()

        # Copy/ensure required node attributes
        for node, data in G_obs.nodes(data=True):
            # Ensure 'sequence' present (copy from original)
            if 'sequence' not in data:
                orig_seq = G_lat.nodes[node].get('sequence', None)
                if orig_seq is None:
                    raise ValueError("Subgraph node missing 'sequence'; cannot build FitnessLandscape.")
                data['sequence'] = orig_seq

            tup = tuple(data['sequence'].to_array())
            idx = seq_to_idx.get(tup, None)
            if idx is None:
                raise ValueError("Subgraph node's sequence not found in parent landscape.")

            # Attach per-layer scalars
            for lname in numeric_layers:
                data[f"fitness_{lname}"] = float(layer_arrays[lname][idx])

            # Stub arrays some pipelines expect
            data.setdefault("gapped_arr", np.zeros((1, 21)))
            data.setdefault("ungapped_arr", np.zeros((1, 20)))

        if return_graph:
            return G_obs

        # Build and return the sub-landscape (edges preserved verbatim)
        G_obs = nx.convert_node_labels_to_integers(G_obs, ordering="sorted")
        subL = FitnessLandscape.from_graph(G_obs, emb_nodes=False)
        return subL

    # Bare graph path
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

def alignment_to_base_numpy_sequences(alignment: Alignment,
                                      alphabet: List[str] = PROT_20) -> List[BaseNumpySequence]:
    """
    Converts a cogent3 Alignment object to a list of
    `BaseNumpySequence` objects.

    Parameters
    ----------
    alignment : Alignment
        The cogent3 alingmnet object
    
    alphabet : List[str], default=`PROT_20`
        The alphabet to use for BaseNumpySequence construction.

    Returns
    -------
    sequences : List[BaseNumpySequence]
        A list of BaseNumpySequence objects with gaps removed.
        
    """
    # Collect sequences as strings and ensure uniform length in alignment
    names = list(alignment.names)
    seq_strs = [str(alignment.get_gapped_seq(n)) for n in names]
    if not seq_strs:
        return []
    L = len(seq_strs[0])
    if any(len(s) != L for s in seq_strs):
        # Defensive: cogent3 alignments should be rectangular
        raise ValueError("Alignment sequences have differing lengths")

    # Keep only columns with no gaps across any sequence
    keep = [j for j in range(L) if all(s[j] != '-' for s in seq_strs)]
    if not keep:
        raise ValueError("All alignment columns contain gaps; cannot build ungapped sequences")

    sequences: list[BaseNumpySequence] = []
    legal = set(alphabet)
    for name, s in zip(names, seq_strs):
        ungapped = ''.join(s[j] for j in keep)
        bad = {ch for ch in set(ungapped.upper()) if ch not in legal}
        if bad:
            raise ValueError(f"Non-canonical residues {sorted(bad)} found in sequence {name!r}; expected only PROT_20: {alphabet}")
        base_numpy_seq = BaseNumpySequence(
            list(ungapped),
            alphabet=alphabet,
            sequence_id=name,
        )
        sequences.append(base_numpy_seq)
    return sequences

def fasta_to_prot20_sequences(filepath: str | Path,
                              *,
                              strict: bool = True,
                              return_gapped: bool = False
                              ) -> Union[List[BaseNumpySequence], Tuple[List[BaseNumpySequence], Optional[List[BaseNumpySequence]]]]:
    """
    Load a FASTA file that may be aligned or unaligned and return a
    sanitised list of BaseNumpySequence with the canonical PROT_20
    alphabet and no gaps.

    Behavior
    - If the file parses as an alignment, gaps are removed per sequence
      after sanitising with sanitize_alignment.
    - Otherwise, sequences are treated as unaligned; any '-' or '.' are
      stripped defensively and non-canonical residues are deleted.

    Parameters
    ----------
    filepath : str | Path
        Path to the FASTA file.

    return_gapped : bool, default=`False`
        When True, also return the sanitised gapped alignment as
        `BaseNumpySequence` objects using the PROT_20 + '-' alphabet.

    Returns
    -------
    sequences : List[BaseNumpySequence]
        List of ungapped, PROT_20 sequences.
    aligned_sequences : Optional[List[BaseNumpySequence]]
        Only returned when ``return_gapped`` is True and the input
        parsed as an alignment. Contains the sanitised gapped
        sequences (all of equal length). Otherwise ``None``.
    """
    p = Path(filepath)
    gap_alphabet = PROT_20 + ["-"]

    def _aligned_sequences_from(aln: Alignment) -> List[BaseNumpySequence]:
        return [
            BaseNumpySequence(
                list(str(aln.get_gapped_seq(name))),
                alphabet=gap_alphabet,
                sequence_id=name,
            )
            for name in aln.names
        ]

    def _ungapped_sequences_from(aln: Alignment) -> List[BaseNumpySequence]:
        seqs: list[BaseNumpySequence] = []
        for name in aln.names:
            ungapped = [ch for ch in str(aln.get_gapped_seq(name)) if ch != '-']
            if not ungapped:
                continue
            seqs.append(BaseNumpySequence(ungapped, alphabet=PROT_20, sequence_id=name))
        return seqs
    try:
        with open(p, 'r'):
            pass
    except FileNotFoundError:
        raise
    # Try alignment path first
    try:
        aln_loaded = load_aligned_seqs(str(p), moltype='protein')
    except Exception:
        aln_loaded = None
    if aln_loaded is not None:
        if strict:
            raw_chars: set[str] = set()
            with open(p, 'r') as fh2:
                for line in fh2:
                    if not line or line.startswith('>'):
                        continue
                    raw_chars.update(ch.upper() for ch in line.strip())
            raw_chars.discard('-'); raw_chars.discard('.')
            illegal = {ch for ch in raw_chars if ch and ch not in set(PROT_20)}
            if illegal:
                raise ValueError(f"Non-canonical residues {sorted(illegal)} present in FASTA; expected only PROT_20: {PROT_20}")

        aln = sanitize_alignment(aln_loaded)
        aligned_sequences: Optional[List[BaseNumpySequence]] = _aligned_sequences_from(aln) if return_gapped else None
        try:
            sequences = alignment_to_base_numpy_sequences(aln, alphabet=PROT_20)
        except ValueError as err:
            if "All alignment columns contain gaps" not in str(err):
                raise
            sequences = _ungapped_sequences_from(aln)

        if return_gapped:
            return sequences, aligned_sequences
        return sequences

    # Fallback path: manually parse FASTA
    legal = set(PROT_20)
    gap_aliases = {'-', '.'}
    names: list[str] = []
    seqs_raw: list[str] = []
    current_name = None
    current_seq = []
    with open(p, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_name is not None:
                    names.append(current_name)
                    seqs_raw.append(''.join(current_seq))
                current_name = line[1:].strip() or 'seq'
                current_seq = []
            else:
                current_seq.append(line)
        # finalize last
        if current_name is not None:
            names.append(current_name)
            seqs_raw.append(''.join(current_seq))

    # Attempt to recover an alignment by replacing illegal residues with gaps
    if return_gapped and seqs_raw:
        lengths = {len(s) for s in seqs_raw}
        if len(lengths) == 1 and next(iter(lengths)) > 0:
            illegal: set[str] = set()

            used_names: set[str] = set()

            def _unique_name(name: str) -> str:
                base = name.strip() or 'seq'
                base = base.replace(' ', '_')
                if base not in used_names:
                    used_names.add(base)
                    return base
                i = 1
                while f"{base}_{i}" in used_names:
                    i += 1
                unique = f"{base}_{i}"
                used_names.add(unique)
                return unique

            seq_dict: Dict[str, str] = {}
            for name, s in zip(names, seqs_raw):
                cleaned_chars = []
                has_residue = False
                for ch in s:
                    up = ch.upper()
                    if up in legal:
                        cleaned_chars.append(up)
                        has_residue = True
                    elif up in gap_aliases:
                        cleaned_chars.append('-')
                    else:
                        illegal.add(up)
                        cleaned_chars.append('-')
                if has_residue:
                    seq_dict[_unique_name(name)] = ''.join(cleaned_chars)

            if strict and illegal:
                raise ValueError(
                    f"Non-canonical residues {sorted(illegal)} present in FASTA; expected only PROT_20: {PROT_20}"
                )

            if seq_dict:
                try:
                    aln = make_aligned_seqs(seq_dict, moltype='protein')
                    aln = sanitize_alignment(aln)
                    aligned_sequences = _aligned_sequences_from(aln) if return_gapped else None
                    try:
                        sequences = alignment_to_base_numpy_sequences(aln, alphabet=PROT_20)
                    except ValueError as err:
                        if "All alignment columns contain gaps" not in str(err):
                            raise
                        sequences = _ungapped_sequences_from(aln)
                    if return_gapped:
                        return sequences, aligned_sequences
                    return sequences
                except ValueError as err:
                    if "All alignment columns contain gaps" not in str(err):
                        raise
                    # otherwise fall through to unaligned handling

    # Treat as unaligned: delete illegal symbols if strict=False
    legal = set(PROT_20)

    out: list[BaseNumpySequence] = []
    for name, s in zip(names, seqs_raw):
        # strip gaps/dots and uppercase
        s_str = s.replace('-', '').replace('.', '').upper()
        if strict:
            bad = {ch for ch in set(s_str) if ch not in legal}
            if bad:
                raise ValueError(f"Non-canonical residues {sorted(bad)} found in sequence {name!r}; expected only PROT_20: {PROT_20}")
            s_filtered = s_str
        else:
            # delete any illegal residues (e.g. X, B, Z, etc.)
            s_filtered = ''.join(ch for ch in s_str if ch in legal)
        # if sequence becomes empty after filtering, skip it
        if not s_filtered:
            continue
        out.append(BaseNumpySequence.from_string(s_filtered, alphabet=PROT_20, moltype='protein', sequence_id=name))
    if return_gapped:
        return out, None
    return out

def moving_window_alignment(alignment: Alignment,
                            window_size: int,
                            overlap: int) -> List[Alignment]:
    """
    Split an alignment into sub-alignments by TIPS (sequences), using a
    moving window over the sequence list with overlap.

    Parameters
    ----------
    alignment : Alignment
        The input alignment (all tips retained in full length per sub-align).
    window_size : int
        Number of tips (sequences) per sub-alignment.
    overlap : int
        Number of tips of overlap between consecutive windows.

    Returns
    -------
    list
        A list of Alignment objects, each containing `window_size`
        tips (except possibly the last), all columns preserved.
    """
    if window_size <= overlap:
        raise ValueError("Window size must be greater than the overlap.")

    # Order-preserving list of (name, seq_str) using cogent3 Alignment API
    items = [(name, str(alignment.get_gapped_seq(name))) for name in alignment.names]
    n_tips = len(items)
    if n_tips == 0:
        return []

    step = window_size - overlap
    windows: list[Alignment] = []

    # Primary windows
    for start in range(0, max(n_tips - window_size + 1, 1), step):
        end = min(start + window_size, n_tips)
        sub = items[start:end]
        if not sub:
            continue
        windows.append(make_aligned_seqs({k: v for k, v in sub}, moltype='protein'))

    # Ensure tail coverage if not exactly divisible and not already included
    if windows:
        last_names = set(windows[-1].names)
        if n_tips > window_size and len(last_names) < window_size:
            tail = items[-window_size:]
            if set(k for k, _ in tail) != last_names:
                windows.append(make_aligned_seqs({k: v for k, v in tail}, moltype='protein'))

    return windows


def iter_moving_window_alignment(alignment: Alignment,
                                 window_size: int,
                                 overlap: int):
    """
    Generator version of moving-window fanning by tips. Yields
    Alignment objects, each containing up to `window_size` tips with
    `overlap` tips overlapping with the previous window. All columns
    (sites) are preserved.

    Parameters
    ----------
    alignment : Alignment
        Source alignment to fan.
    window_size : int
        Number of tips (sequences) per sub-alignment.
    overlap : int
        Number of tips to overlap between consecutive windows.
    """
    if window_size <= overlap:
        raise ValueError("Window size must be greater than the overlap.")

    items = [(name, str(alignment.get_gapped_seq(name))) for name in alignment.names]
    n_tips = len(items)
    if n_tips == 0:
        return

    step = window_size - overlap
    # primary windows
    start = 0
    while start < n_tips:
        end = min(start + window_size, n_tips)
        sub = items[start:end]
        if not sub:
            break
        yield make_aligned_seqs({k: v for k, v in sub}, moltype='protein')
        if end == n_tips:
            break
        start += step
    # ensure tail coverage (last window of exact size if not already yielded)
    if n_tips > window_size:
        tail = items[-window_size:]
        tail_names = [k for k, _ in tail]
        try:
            last_names = list(alignment.names)  # fallback
        except Exception:
            last_names = []
        # we cannot inspect last yielded window names here; rely on
        # consumers tolerating a duplicate last window in rare edge cases
        # to guarantee coverage.
        # In typical use, the loop above already yielded the tail.


def iter_random_subalignment(alignment: Alignment,
                             sample_size: int,
                             n_samples: int,
                             *,
                             replace: bool = False,
                             rng: Optional[np.random.Generator] = None,
                             seed_fraction: float = 0.1,
                             seed_count: Optional[int] = None,
                             force_seed: bool = True):
    """Yield sub-alignments built by sampling tips uniformly at random.

    Parameters
    ----------
    alignment : Alignment
        Source alignment to sample from.
    sample_size : int
        Number of sequences (tips) per sampled sub-alignment.
    n_samples : int
        Total number of sub-alignments to generate.
    replace : bool, default=False
        Whether to sample with replacement. When False and ``sample_size``
        equals the number of tips, each sample simply returns the full
        alignment.
    rng : numpy.random.Generator, optional
        Random number generator to use. If ``None``, a default generator
        (``np.random.default_rng()``) is used.
    """
    names = list(alignment.names)
    n_tips = len(names)
    if n_tips == 0 or sample_size <= 0 or n_samples <= 0:
        return
    if sample_size > n_tips and not replace:
        raise ValueError("sample_size exceeds number of tips; enable replace=True to allow repeats.")

    rng = rng or np.random.default_rng()
    seq_map = {name: str(alignment.get_gapped_seq(name)) for name in names}

    if force_seed:
        if seed_count is None:
            seed_count = max(1, int(round(seed_fraction * n_tips)))
        else:
            seed_count = int(seed_count)
        seed_count = min(seed_count, sample_size, n_tips)
        seed_names = rng.choice(names, size=seed_count, replace=False)
        seed_set = set(seed_names)
        pool_no_seed = [n for n in names if n not in seed_set]
    else:
        seed_set = set()
        pool_no_seed = names

    for _ in range(n_samples):
        if force_seed and seed_set:
            needed = sample_size - len(seed_set)
            if needed <= 0:
                chosen = list(seed_set)
            else:
                if replace:
                    extra = rng.choice(names, size=needed, replace=True)
                else:
                    if needed >= len(pool_no_seed):
                        extra = pool_no_seed
                    else:
                        extra = rng.choice(pool_no_seed, size=needed, replace=False)
                chosen = list(seed_set) + list(extra)
        else:
            if replace:
                chosen = rng.choice(names, size=sample_size, replace=True)
            else:
                if sample_size >= n_tips:
                    chosen = names
                else:
                    chosen = rng.choice(names, size=sample_size, replace=False)
        sub = {name: seq_map[name] for name in chosen}
        yield make_aligned_seqs(sub, moltype='protein')

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

def check_full_hamming(landscape: 'FitnessLandscape',
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
