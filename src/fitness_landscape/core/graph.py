import numpy as np
import networkx as nx
from typing import List, Union, Literal, Tuple, Optional, Sequence
from .sequence import BaseNumpySequence, BinarySequence, sequence_distance, SoftSequence
from ..phylo.phylogenetic_asr import ASRConstructor
from ..phylo._sub_matrices import lg
import gudhi
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
from cogent3 import ArrayAlignment
from .._const import PROT_20
from ..utils import calculate_gapped_soft_score, 
from softalign.soft_alignment import align_soft_sequences
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import faiss


def _pack_binary(seqs: list[BaseNumpySequence]) -> np.ndarray:
    """
    Helper function to convert a list of `BaseNumpySequences` itno an
    int encoded array
    """
    
    # (n, L)
    arr = np.stack([s.to_array().astype(np.uint8) for s in seqs], axis=0)  
    if not np.isin(arr, [0, 1]).all():

        raise ValueError("Binary builder requires sequences with symbols {0,1}.")
    L = arr.shape[1]
    if L > 64:
        raise ValueError("Bit-pack assumes L <= 64.")
    
    # bit for each pos
    powers = (1 << np.arange(L, dtype=np.uint64))
    
    return (arr.astype(np.uint64) * powers).sum(axis=1, dtype=np.uint64)

def _build_hamming_csr_binary(sequences: list[BinarySequence]) -> csr_matrix:
    """
    Function to build undirected CSR adjacency for a binary Hamming
    graph using XOR neighbor generation.

    Parameters
    ----------
    sequences : List[BinarySequence]
        The input BinarySequence objects used to construct the
        Hamming graph. 
    
    Returns
    -------
    A : sp.csr_matrix
        Sparse adjacency matrix. 
    """
    # Guardrails
    if len(sequences) == 0:
        return csr_matrix((0, 0))
    
    n = len(sequences)
    bitstrings = _pack_binary(sequences)

    # infer L from used bits (safe if all positions vary at least once)
    max_bit = int(max(int(b).bit_length() for b in bitstrings))
    L = max(1, max_bit)

    index_of = {int(bs): i for i, bs in enumerate(bitstrings)}

    # worst-case capacity: n*L*2 (both directions)
    cap = n * L * 2
    rows = np.empty(cap, dtype=np.int32)
    cols = np.empty(cap, dtype=np.int32)
    
    # Lookup bit flipped sequeneces in hash map.
    k = 0
    for i, s in enumerate(bitstrings):
        s_int = int(s)
        for pos in range(L):
            t = s_int ^ (1 << pos)
            j = index_of.get(t)
            if j is None or i >= j:
                continue
            rows[k], cols[k] = i, j
            k += 1
            rows[k], cols[k] = j, i
            k += 1

    rows = rows[:k]; cols = cols[:k]
    # 1 weighted adjacency for unweighted.
    data = np.ones(k, dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A

def create_hamming_graph_binary(sequences: list[BinarySequence]) -> nx.Graph:
    """
    Function to build a undirected Hamming graph using efficiency bit
    wise (XOR) operations. 

    Parameters
    ----------
    sequences : List[BinarySequence]
        The input BinarySequence objects used to construct the
        Hamming graph. 

    Returns
    -------
    G : nx.Graph
        The undirected graph that can construct the `FitnessLandscape`
        class. 
    """
    A = _build_hamming_csr_binary(sequences)
    G = nx.from_scipy_sparse_array(A) 
    
    # attach node attributes for `FitnessLandscape` constructor.s
    for i, seq in enumerate(sequences):
        G.nodes[i]['sequence'] = seq
        
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
        G[u][v]['distance'] = 1
    
    return G

def _encode_multiallele(seqs: list[BaseNumpySequence]) -> tuple[np.ndarray, dict[str,int]]:
    """
    Helper function to map string symbols in the `BaseNumpySequence`
    alphabet to contiguous integers. 
    """

    # collect alphabet in order of first appearance to keep mapping stable
    seen = {}
    mats = []
    for s in seqs:
        arr = s.to_array()
        mats.append(arr)
        for sym in map(str, arr):
            if sym not in seen:
                seen[sym] = len(seen)
    mapping = seen
    int_mat = np.stack([[mapping[str(x)] for x in s.to_array()] for s in seqs], axis=0).astype(np.int32)
    return int_mat, mapping  # (n,L)

def _build_hamming_csr_multiallele_masked(sequences: list[BaseNumpySequence]) -> csr_matrix:
    """
    Function to build a sparse Hamming adjacency matrix using a
    radix-encoded (base B for B alleles) masking algorithm.

    Parameters
    ----------
    sequences : List[BaseNumpySequences] 
        List of input sequences. 
    
    Returns
    -------
    A : sp.csr_matrix
        The sparse Hamming adjacency matrix. 
    """

    # Guardrails
    if len(sequences) == 0:
        return csr_matrix((0, 0))

    X, _ = _encode_multiallele(sequences)  # (n,L) int32
    n, L = X.shape
    base = int(X.max()) + 1

    # base powers for radix encoding
    powers = (base ** np.arange(L, dtype=np.int64))  # [B^0, B^1, ..., B^(L-1)]
    
    # encode full keys
    key_full = (X * powers).sum(axis=1, dtype=np.int64)

    # storage (rough upper bound): ~ n*L*avg_degree/2*2 ~ n*L for sparse datasets
    rows = []
    cols = []

    for p in range(L):

        # masked key: remove digit at p
        masked = key_full - (X[:, p].astype(np.int64) * powers[p])
        order = np.argsort(masked, kind='stable')
        masked_sorted = masked[order]
        xp = X[:, p][order]
        # walk runs of identical masked key
        start = 0
        while start < n:
            end = start + 1
            while end < n and masked_sorted[end] == masked_sorted[start]:
                end += 1
            if end - start >= 2:
                block_idx = order[start:end]
                block_allele = xp[start:end]

                # group by allele value within the block
                # unique + inverse index
                ua, inv = np.unique(block_allele, return_inverse=True)
                
                for a_i in range(len(ua)):
                    src = block_idx[inv == a_i]
                    for a_j in range(a_i + 1, len(ua)):
                        dst = block_idx[inv == a_j]
                        if src.size and dst.size:
                            s_rep = np.repeat(src, dst.size)
                            d_tile = np.tile(dst, src.size)
                            rows.append(s_rep)
                            cols.append(d_tile)
                            rows.append(d_tile)   # symmetric
                            cols.append(s_rep)

            start = end

    if not rows:
        A = csr_matrix((np.zeros(0, dtype=np.float32), (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))),
                       shape=(n, n))
        return A

    rows = np.concatenate(rows).astype(np.int32)
    cols = np.concatenate(cols).astype(np.int32)

    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    
    # remove exact duplicates
    keep = np.ones_like(rows, dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    rows, cols = rows[keep], cols[keep]
    
    # 1 for unweighted.
    data = np.ones(rows.size, dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return A

def create_hamming_graph_multiallele(sequences: list[BaseNumpySequence]) -> nx.Graph:
    """
    Function to create a Hamming graph using B-radix encoded sequence
    masking to identify Hamming neighbors. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of input sequences to construct the graph from. 
    
    Returns
    -------
    G : nx.Graph
        The undirected graph with edge and node features accepted by
        the `FitnessLandscape` from graph constructor.
    """

    A = _build_hamming_csr_multiallele_masked(sequences)
    G = nx.from_scipy_sparse_array(A)
    for i, seq in enumerate(sequences):
        G.nodes[i]['sequence'] = seq
        
    # TODO: Update `weight`, `distance`, `similarity` logic.
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
        G[u][v]['distance'] = 1

    return G

# Main public method
def create_hamming_graph(sequences: List[BaseNumpySequence],
                         _backend: Literal['auto', 'binary_xor', 'masked'] = 'auto') -> nx.Graph:
    """
    Create a Hamming graph from sequences and fitness values. In a
    Hamming graph, nodes represent sequences and edges connect
    sequences that differ by exactly one position (Hamming
    distance = 1).

    Parameters
    ----------
    sequences : list of Sequence or array-like
        Sequences to connect.
    _backend : str, default=`aut`
        Backend to compute Hamming neighbors. 
        -`binary_xor`: applies binary XOR operation to find bit-encoded
        sequences that differ by precisely 1 bit in an indexed lookup
        table. Scales in O(n * L). Applies exlusively to the
        `BinarySequence` class. 
        - `masked` : applies a position p mask over radix (base B)
        enocoded sequences to find sequences that are identical outside
        of position p. Scales in O(L n log n)
        - `auto` : automatically chooses backend based on the sequence
        type.

    Returns
    -------
    networkx.Graph
        Hamming graph.
    """
    
    # Safety check all sequences are binary classes.
    is_binary = all(isinstance(s, BinarySequence) for s in sequences)

    if _backend == "auto":
        _backend = "binary_xor" if is_binary else "masked"

    if _backend == "binary_xor":
        if not is_binary:
            raise ValueError("backend='binary_xor' requires binary sequences {0,1}.")
        return create_hamming_graph_binary(sequences)
    elif _backend == "masked":
        return create_hamming_graph_multiallele(sequences)
    else:
        raise ValueError(f"Unknown `_backend`: {_backend}")


def _one_hot_matrix_amino(seqs: List[BaseNumpySequence]) -> np.ndarray:
    """
    Helper function to to convert a list of BaseNumpySequence objects
    into a one hot encded matrix. Assumes all sequences share the same
    alphabet.

    Parameters
    ----------
    seqs : List[BaseNumpySequence] 
        The sequences to convert to one-hot. 

    Returns
    -------
    X : np.ndarray
        The one hot encoded matrix.    
    """
    # assume all share the same alphabet order (e.g. PROT_20)
    A = list(seqs[0].alphabet)
    amap = {str(a): i for i, a in enumerate(A)}
    L = len(seqs[0])
    n = len(seqs)
    X = np.zeros((n, L * len(A)), dtype=np.float32)
    W = len(A)
    for r, s in enumerate(seqs):
        arr = s.to_array()
        for p, sym in enumerate(arr):
            X[r, p*W + amap[str(sym)]] = 1.0
    return X

def _find_knn_balltree(X : np.ndarray,
                       k : int,
                       tiebuffer : int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to find nearest neighbors by BallTree.

    Parameters
    ----------
    X : np.ndarray
        The encoded sequence array. 
    
    k : int 
        The number of neighbours to find.
        
    tiebuffer : int, defaut=1
        The tiebuffer for equidistant neighbors above k. 
    
    Returns
    -------
    dists, inds : np.ndarray
        Tuple of distances and indices.
    """
    n = X.shape[0]
    nn = NearestNeighbors(algorithm='auto', metric='hamming')
    nn.fit(X)
    kq = min(k + 1 + tiebuffer, n)
    dists, inds = nn.kneighbors(X, n_neighbors=kq, return_distance=True)
    return dists, inds

def _find_knn_faiss(X: np.ndarray,
                    k: int,
                    index_type: Literal['hnsw', 'flat', 'ivf'] = "hnsw",
                    metric: Literal['ip', 'l2'] = 'ip',
                    use_gpu: bool = False,
                    hnsw_M: int = 32,
                    include_self: bool = False,
                    tiebuffer : int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to find nearest neighbors by FAISS backend.

    Parameters
    ----------
    X : np.ndarray
        The encoded sequence array. 
    
    k : int 
        The number of neighbours to find.

    index_type : str, default=`hnsw`
        The faiss index type. Options are:
        - flat (exact) for small n. 
        - hnsw (approximate) for large n. 
        - ivf (approximate) for very large n.
    
    metric : str, default=`ip`
        The faiss metric to use specificall for hnsw. Options are:
        - `ip` : Inner product
        - `l2` : L2 norm 
    
    include_self : bool, default=`False`
        Boolean to include self in the neighbor list.
    
    use_gpu : bool, default=`False`
        Boolean to use FAISS GPU acceleration (application only to the
        flat index).
    
    hsnw_M : int, default = 32
        The hnsw dimesnion.
    
    include_self : bool, default=`False`
        Boolean to include self edges.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.
    
    Returns
    -------
    dists, inds : np.ndarray
        Tuple of distances and indices.

    """
    n, d = X.shape

    # Set the FAISS metric so easy conversion back to hamming distance.
    if metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError(f"Expected `faiss_metric` to be in [`ip`, `l2`, found {faiss_metric}]")

    # FAISS index
    if index_type == "flat":
        if metric == "ip":
            index = faiss.IndexFlatIP(d)
        else:  # 'l2'
            index = faiss.IndexFlatL2(d)
    
    elif index_type == "hnsw":
        # Catch error in setting `faiss_metrix`.
        try:
            index = faiss.IndexHNSWFlat(d, hnsw_M, faiss_metric)
        except TypeError:
            # Fallback: default is L2
            index = faiss.IndexHNSWFlat(d, hnsw_M)
            if metric != "l2":
                raise RuntimeError(
                    "IndexHNSWFlat in this FAISS build uses L2 only; set metric='l2' "
                    "or switch to 'flat'/'ivf' with METRIC_INNER_PRODUCT."
                )
            
    elif index_type == "ivf":
        nlist = max(256, int(np.sqrt(n)))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(X)
        index.nprobe = min(64, nlist)
    else:
        raise ValueError(f"Expected `index_type` to be in [`flat`, `hnsw`, `ivf`], found {index_type}")

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(X)
    
    # Include consideration for self edges and a tiebuffer.
    kq = k + (0 if include_self else 1) + tiebuffer
    dists, inds = index.search(X, kq)
    return dists, inds


def _create_knn_graph_balltree(sequences: List[BaseNumpySequence],
                               k: int,
                               tie_policy: Literal['all', 'min_index', 'random'] = 'all',
                               tiebuffer: int = 128,
                               seed: int = 42) -> nx.Graph:
    """
    Function to create an exact KNN using the scipy `BallTree`
    algorithm.

    Parameters
    -----------
    sequences : List[BaseNumpySequence]
        The list of input sequences. 
    k : int
        The number of neighbours to connect each sequence to. 
    
    tie_policy : str, default=`all`
        The tie policy for when there are more than k equidistant
        neighbors found. Options are:
        - `all` : all neighbors are kept and the graph becomes
        irregular.
        - `min_index` : The "first" connection is kept, the others are
        arbitrarily removed. The graph remains k regular. 
        - `random` : equidistant edges are kept at random.
    
    seed : int, default=42
        The random state seed.

    Returns
    -------
    nx.Graph
        The KNN graph. 
    """
    
    n = len(sequences)
    if n == 0:
        return nx.Graph()

    # integer-coded (n, L)
    X, _ = _encode_multiallele(sequences)  
    L = X.shape[1]

    dists, inds = _find_knn_balltree(X, k=k, tiebuffer=tiebuffer)

    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []

    for i in range(n):
        ids = inds[i]
        # convert fraction to Hamming count
        ds = (dists[i] * L)  

        # drop self
        keep = (ids != i)
        ids = ids[keep]
        ds  = ds[keep]

        if ids.size == 0:
            continue

        # stable sort by distance
        order = np.argsort(ds, kind='stable')
        ids, ds = ids[order], ds[order]

        if ids.size <= k:
            take = np.arange(ids.size)
        else:
            dk = ds[k-1]
            cand = np.nonzero(ds <= dk + 1e-9)[0]  # include all ties at kth distance
            if tie_policy == 'all':
                take = cand
            elif tie_policy == 'min_index':
                take = cand[:k]
            elif tie_policy == 'random':
                take = rng.choice(cand, size=min(k, cand.size), replace=False)
            else:
                raise ValueError(f"Unknown tie_policy: {tie_policy}")

        ids = ids[take]; ds = ds[take]
        rows.append(np.full(ids.size, i, dtype=np.int32))
        cols.append(ids.astype(np.int32))
        vals.append(ds.astype(np.float32))

    I = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
    J = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
    V = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

    # directed k-NN : symmetrize by UNION so degree >= k (and “all” can exceed)
    M = coo_matrix((V, (I, J)), shape=(n, n)).tocsr()
    U = M.maximum(M.T)

    G = nx.from_scipy_sparse_array(U, edge_attribute='distance')
    
    G.add_nodes_from(range(n))
    for i in range(n):
        G.nodes[i]['sequence'] = sequences[i]
    
    for u, v in G.edges():
        G[u][v]['weight'] = G[u][v]['distance']
    return G

def _create_knn_graph_faiss(sequences: List[BaseNumpySequence],
                            k: int,
                            *,
                            index_type: Literal['hnsw', 'flat', 'ivf'] = "hnsw",
                            metric: Literal['ip', 'l2'] = 'ip',
                            include_self: bool = False,
                            use_gpu: bool = False,
                            hnsw_M: int = 32,
                            tiebuffer : int = 128,
                            tie_policy: Literal['all', 'min_index', 'random'] = 'all',
                            seed: int = 42) -> nx.Graph:
    """
    Function to create an approximate nearest neighbour graph using
    FAISS indexing for efficient neighbour searching. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of input sequences to connect. 

    k : int 
        The number of neighbours to connect each node with. 

    index_type : str, default=`hnsw`
        The faiss index type. Options are:
        - flat (exact) for small n. 
        - hnsw (approximate) for large n. 
        - ivf (approximate) for very large n.
    
    metric : str, default=`ip`
        The faiss metric to use specificall for hnsw. Options are:
        - `ip` : Inner product
        - `l2` : L2 norm 
    
    include_self : bool, default=`False`
        Boolean to include self in the neighbor list.
    
    use_gpu : bool, default=`False`
        Boolean to use FAISS GPU acceleration (application only to the
        flat index).
    
    hsnw_M : int, default = 32
        The hnsw dimesnion.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.
    
    tie_policy : str, default=`all`
        The tie policy for when there are more than k equidistant
        neighbors found. Options are:
        - `all` : all neighbors are kept and the graph becomes
        irregular.
        - `min_index` : The "first" connection is kept, the others are
        arbitrarily removed. The graph remains k regular. 
        - `random` : equidistant edges are kept at random.
    
    seed : int, default=42
        The random state seed.

    Returns
    -------
    G : nx.Graph
        The constructed nearest neighbor graph. 
    """
    
    X = _one_hot_matrix_amino(sequences) # (n, L*20) one-hot
    n, d = X.shape
    L = len(sequences[0])

    D, I = _find_knn_faiss(X, k=k, index_type=index_type, metric=metric, use_gpu=use_gpu, hnsw_M=hnsw_M, tiebuffer=tiebuffer)

    # Convert metric distances to Hamming distances.
    if metric == "ip":
        dpos_all = (L - D).astype(np.float32)
    else: 
        dpos_all = (0.5 * D).astype(np.float32)

    G = nx.Graph()
    for i, s in enumerate(sequences):
        G.add_node(i, sequence=s)

    for i in range(n):
        ids = I[i]
        ds  = dpos_all[i]

        # filter invalids / -1 and optionally self
        valid = ids >= 0
        if not include_self:
            valid &= (ids != i)
        ids = ids[valid]
        ds  = ds[valid]

        if ids.size == 0:
            continue

        # stable sort by distance
        order = np.argsort(ds, kind="stable")
        ids, ds = ids[order], ds[order]

        # choose top-k with tie handling
        if ids.size <= k:
            take = np.arange(ids.size)
        else:
            dk = ds[k-1]
            cand = np.nonzero(ds <= dk)[0]
            if tie_policy in ("min_index"):
                take = cand[:k]
            elif tie_policy == "all":
                take = cand
            elif tie_policy == "random":
                if cand.size > k:
                    rng = np.random.default_rng(seed)
                    take = rng.choice(cand, size=k, replace=False)
                else:
                    take = cand
            else:
                raise ValueError(f"Unknown tie_policy: {tie_policy}")

        sel_ids = ids[take].astype(int)
        sel_ds  = ds[take].astype(float)

        # add edges; if an edge exists from the other direction, keep the min distance
        for j, dij in zip(sel_ids, sel_ds):
            if i == j:
                continue
            if G.has_edge(i, j):
                # keep the tighter (min) distance
                prev = G[i][j].get("distance", dij)
                if dij < prev:
                    G[i][j]["distance"] = dij
                    G[i][j]["weight"]   = dij
            else:
                G.add_edge(i, j, distance=dij, weight=dij)

    return G

def create_knn_graph(sequences: List[BaseNumpySequence],
                     k: int,
                     *,
                     backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                     index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                     faiss_metric: Literal['ip', 'l2'] = 'ip',
                     include_self: bool = False,
                     use_gpu: bool = False,
                     hnsw_M: int = 32,
                     tiebuffer: int = 128,
                     tie_policy: Literal['all', 'min_index', 'random'] = 'all',
                     seed : int = None) -> nx.Graph:
    """
    Function to create a k-nearest neighbor network graph from
    sequences, using an efficient backend algorithm. 

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The sequences to construct the graph from. 
    
    k : int
        The number of neighbors to connect each node to. 
    
    backend : str, default=`auto`
        The computational backend to use. Options are:
        -`faiss` : use a FAISS-based (sublinear) scalling (but
        approximate) backend to find neighbors. 
        - `balltree` : use the BallTree exact solver, which scales
        poorly with large dimension size. 
        - `auto` : Automatic backend based on dataset size.
    
    index_type : str, default=`hnsw`
        The FAISS indexing algorithm to use. Options are:
        - `hnsw` : hierarchical navigatible small worlds. Effective on
        large n. Approximate.
        - `flat` : Exact flat indexing.
        - `ivf` : inverted file indexing algorithm. Effective on very
        large n. Approximate.
    
    faiss_metric : str, default=`ip`
        The faiss metric. Options are:
        - `ip` : the inner produt.
        - `l2` : the L2 norm. 
        Use of `ip` guarantees distances are returned / stored as
        Hamming distances. 
    
    use_gpu : bool, default=`False`
        Boolean to use GPU on flat indexing. 
    
    hnsw_M : int, default=32
        The hnsw dimension size.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.
    
    tie_policy : str, default=`all`
        The tie policy for when there are more than k equidistant
        neighbors found. Options are:
        - `all` : all neighbors are kept and the graph becomes
        irregular.
        - `min_index` : The "first" connection is kept, the others are
        arbitrarily removed. The graph remains k regular. 
        - `random` : equidistant edges are kept at random.
    
    seed : int, default=42
        The random state seed. 
    Returns
    -------
    nx.Graph    
        The constructed KNN graph.
    """
    n = len(sequences)

    if backend == 'auto':
        backend = 'faiss' if n >= 5000 else 'balltree'

    if backend == 'faiss':
        return _create_knn_graph_faiss(
            sequences, k,
            index_type=index_type,
            metric=faiss_metric,
            include_self=include_self,
            use_gpu=use_gpu,
            hnsw_M=hnsw_M,
            tiebuffer=tiebuffer,
            tie_policy=tie_policy,
            seed=seed
        )
    elif backend == 'balltree':
        return _create_knn_graph_balltree(sequences,
                                          k, 
                                          tie_policy=tie_policy,
                                          tiebuffer=tiebuffer,
                                          seed=seed)
    else:
        raise ValueError(f"Unsupported backend {backend!r}. Expected `auto`, `faiss`, or `balltree`.")

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


def create_diffusion_emb_graph(sequences: List[BaseNumpySequence],
                               embeddings: np.ndarray = None,
                               k: int = 128,
                               tiebuffer: int = 0,
                               backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                               index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                               faiss_metric: Literal['ip', 'l2'] = 'ip',
                               include_self: bool = False,
                               use_gpu: bool = False,
                               hnsw_M: int = 32,
                               t: int = 5,
                               connectivity_threshold: float = 1e-4,
                               **kwargs) -> nx.Graph:
    """
    Function to construct a graph based on expected diffusion
    behaviour in a high-dimensional embedding space. Uses sparse
    operations neighbor finding with either BallTree or FAISS-based
    algorithms to accelerate the computation for very large graphs.

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

    backend : str, default=`auto`
        The computational backend to use. Options are:
        -`faiss` : use a FAISS-based (sublinear) scalling (but
        approximate) backend to find neighbors. 
        - `balltree` : use the BallTree exact solver, which scales
        poorly with large dimension size. 
        - `auto` : Automatic backend based on dataset size.
    
    index_type : str, default=`hnsw`
        The FAISS indexing algorithm to use. Options are:
        - `hnsw` : hierarchical navigatible small worlds. Effective on
        large n. Approximate.
        - `flat` : Exact flat indexing.
        - `ivf` : inverted file indexing algorithm. Effective on very
        large n. Approximate.
    
    faiss_metric : str, default=`ip`
        The faiss metric. Options are:
        - `ip` : the inner produt.
        - `l2` : the L2 norm. 
        Use of `ip` guarantees distances are returned / stored as
        Hamming distances. 
    
    use_gpu : bool, default=`False`
        Boolean to use GPU on flat indexing. 
    
    hnsw_M : int, default=32
        The hnsw dimension size.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.

    connectivity_threshold : float, default=`1e-04`
        The threshold the define discrete connectivity.

    Returns
    -------
    G : nx.graph
        The constructed graph with `BaseNumpySequence` features stored
        under `sequence`.
    """
    if embeddings is None:
        embeddings, _ = _encode_multiallele(sequences)
    
    k_for_scale = k
    if embeddings.shape[0] <= k_for_scale:
        k_for_scale = embeddings.shape[0] - 1
    
    # Use balltree algorithm (will fail as shape of embeddings >>>)
    if backend == 'balltree':
        distances, _ = _find_knn_balltree(embeddings, k, tiebuffer)
    
    # Use FAISS algorithm (approx or exact).
    elif backend == 'faiss':
        distances, _ = _find_knn_faiss(embeddings,
                                       k,
                                       index_type=index_type,
                                       metric=faiss_metric,
                                       use_gpu=use_gpu,
                                       hnsw_M=hnsw_M,
                                       tiebuffer=tiebuffer) 
                                    
    # Select backend algorithm based on size of embeddings.
    elif backend == 'auto':
        if embeddings.shape[0] < 5000:
            distances, _ = _find_knn_balltree(embeddings, k, tiebuffer)
        else:
            distances, _ = _find_knn_faiss(embeddings,
                                           k,
                                           index_type=index_type,
                                           metric=faiss_metric,
                                           use_gpu=use_gpu,
                                           hnsw_M=hnsw_M,
                                           tiebuffer=tiebuffer) 

    
    # The scale for each point is the distance to its k-th neighbor
    sigma = distances[:, k_for_scale]
    pos = sigma[sigma > 0]

    if pos.size == 0:
        median_sigma_sq = 1.0
    else:
        median_sigma_sq = float(np.median(pos))**2
        if not np.isfinite(median_sigma_sq) or median_sigma_sq <= 0:
            median_sigma_sq = 1.0

    gamma = 1.0 / (2 * median_sigma_sq)

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

def create_phylo_graph(sequences: Union[Path, ArrayAlignment],
                       replacement_matrix: List[str] = ['LG'],
                       model_fitting: bool = True) -> nx.DiGraph:
    """
    Factory function to create an undirected graph using phylogenetic
    inference and ancestral sequence reconstruction (with an 
    equilibrium amino acid replacement matrix).

    Parameters
    ----------
    alignment : Path or Alignment
        The alignment of extant sequences to use for ASR and
        phylogenetic infernece.
    
    replacement_matrix : List, default=[`LG`]
        List of replacement matrices to use for phylogenetic
        reconstruction. Must be an NQ non-equilibrium model.

    model_fitting : bool, default=`True`
        Whether to fit the ML model, using the model set defined in
        `replacement_matrix`.

    Returns
    -------
    G : nx.Graph
        The undirected graph output.
    """
    constructor = ASRConstructor(sequences,
                                 replacement_matrix = replacement_matrix,
                                 model_fitting = model_fitting)
    graph = constructor.construct_dag(graph_type='undirected')
    return graph

def create_evol_diffusion_graph(sequences: List[BaseNumpySequence],
                                             embeddings: np.ndarray,
                                             replacement_matrix: np.ndarray = lg,
                                             tiebuffer: int = 0,
                                             backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                                             index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                                             faiss_metric: Literal['ip', 'l2'] = 'ip',
                                             include_self: bool = False,
                                             use_gpu: bool = False,
                                             hnsw_M: int = 32,
                                             k: int = 50,
                                             t: int = 5,
                                             tau: float = 1.0,
                                             connectivity_threshold: float = 1e-4,
                                             **kwargs) -> nx.Graph:
    """
    Constructs a diffusion graph by scoring standard alignments with an
    symmetric equilibrium replacement matrix.

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of sequence in the landscape. 

    replacement_matrix : np.ndarray, default=lg
        The symmetric replacememnt matrix used to score symmetric
        distances.
    
    embeddings : np.ndarray
        Sequence embeddings indexed by the entry in `sequences`.
    
    k : int, default=50
        The number of neighbours to use for kNN pre-filtering.

    backend : str, default=`auto`
        The computational backend to use. Options are:
        -`faiss` : use a FAISS-based (sublinear) scalling (but
        approximate) backend to find neighbors. 
        - `balltree` : use the BallTree exact solver, which scales
        poorly with large dimension size. 
        - `auto` : Automatic backend based on dataset size.
    
    index_type : str, default=`hnsw`
        The FAISS indexing algorithm to use. Options are:
        - `hnsw` : hierarchical navigatible small worlds. Effective on
        large n. Approximate.
        - `flat` : Exact flat indexing.
        - `ivf` : inverted file indexing algorithm. Effective on very
        large n. Approximate.
    
    faiss_metric : str, default=`ip`
        The faiss metric. Options are:
        - `ip` : the inner produt.
        - `l2` : the L2 norm. 
        Use of `ip` guarantees distances are returned / stored as
        Hamming distances. 
    
    use_gpu : bool, default=`False`
        Boolean to use GPU on flat indexing. 
    
    hnsw_M : int, default=32
        The hnsw dimension size.
    
    tiebuffer : int, default=128
        The number of hits kept in buffer to eliminate ties.
    
    t : int, default=5
        The number of diffusion steps taken.
    
    tau : float, default=1.0
        The temperature parameter used to smooth the distance kernel.

    Returns
    -------
    nx.Graph
        The constructed graph.
    """
    if embeddings is None:
        embeddings, _ = _encode_multiallele(sequences)
        
     # Type check alphabet first
    for seq in sequences:
        if seq.alphabet != PROT_20:
            raise ValueError("Sequence alphabet must be PROT_20 for all entries.")

    n_sequences = len(sequences)
    if n_sequences == 0:
        return nx.Graph()

    if k > n_sequences - 1:
        k = n_sequences - 1

        # Use balltree algorithm (will fail as shape of embeddings >>>)
    if backend == 'balltree':
        _, neighbor_indices = _find_knn_balltree(embeddings, k, tiebuffer)
    
    # Use FAISS algorithm (approx or exact).
    elif backend == 'faiss':
        _, neighbor_indices = _find_knn_faiss(embeddings,
                                       k,
                                       index_type=index_type,
                                       metric=faiss_metric,
                                       use_gpu=use_gpu,
                                       hnsw_M=hnsw_M,
                                       tiebuffer=tiebuffer) 
                                    
    # Select backend algorithm based on size of embeddings.
    elif backend == 'auto':
        if embeddings.shape[0] < 5000:
            _, neighbor_indices = _find_knn_balltree(embeddings, k, tiebuffer)
        else:
            _, neighbor_indices = _find_knn_faiss(embeddings,
                                           k,
                                           index_type=index_type,
                                           metric=faiss_metric,
                                           use_gpu=use_gpu,
                                           hnsw_M=hnsw_M,
                                           tiebuffer=tiebuffer) 

    pairs_to_align = set()
    for i in range(n_sequences):
        for j_idx in neighbor_indices[i]:
            if i != j_idx:
                # Add pairs in a canonical order to avoid duplicates
                pair = tuple(sorted((i, j_idx)))
                pairs_to_align.add(pair)

    kernel_matrix = np.zeros((n_sequences, n_sequences))

    for i, j in pairs_to_align:
        seq_i = sequences[i]
        arr_i = seq_i.posterior if isinstance(seq_i, SoftSequence) else seq_i.to_one_hot()

        seq_j = sequences[j]
        arr_j = seq_j.posterior if isinstance(seq_j, SoftSequence) else seq_j.to_one_hot()

        alignment, _ = align_soft_sequences(sequences=[arr_i, arr_j], alphabet=PROT_20)

        score = calculate_gapped_soft_score(aligned_seq1=alignment[0],
                                            aligned_seq2=alignment[1],
                                            q=replacement_matrix)

        kernel_value = np.exp(score / tau)
        kernel_matrix[i, j] = kernel_value
        kernel_matrix[j, i] = kernel_value # Explicitly symmetrize

    np.fill_diagonal(kernel_matrix, 0)

    row_sums = kernel_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = kernel_matrix / row_sums

    diffused_matrix = np.linalg.matrix_power(transition_matrix, t)
    
    # Symmetrize the final diffused matrix to ensure the graph is undirected
    symmetric_diffused_matrix = (diffused_matrix + diffused_matrix.T) / 2

    graph = nx.Graph()
    graph.add_nodes_from(range(n_sequences))

    rows, cols = np.where(np.triu(symmetric_diffused_matrix > connectivity_threshold, k=1))

    for i, j in zip(rows, cols):
        graph.add_edge(i, j, weight=symmetric_diffused_matrix[i, j])

    for i, seq in enumerate(sequences):
        graph.nodes[i]['sequence'] = seq

    return graph

def pairwise_expected_hamming_from_aligned(aligned: Sequence[np.ndarray],
                                           *,
                                           gap_at: int = -1,
                                           return_norm: bool = True,
                                           block_cols: Optional[int] = None,
                                           eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute pairwise expected Hamming counts and effective lengths for
    N aligned soft sequences. Does not apply to hard sequences. 

    Parameters
    ----------
    aligned : Sequence[np.ndarray]
        List of aligned soft sequences, each of shape (L_aln, A+1)
        where last axis is A amino acids + gap.

    gap_at : int, default=-1
        Index of the gap channel in the last axis of aligned[i].
        If negative, counts from the end (-1 = last channel).
    
    return_norm : bool, default=True
        Whether to return normalized expected mismatch fraction in [0,1].
    
    block_cols : Optional[int], default=None
        If set, process columns in blocks of this size for memory efficiency.
        If None, process all columns at once.
    
    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.

    Returns
    -------
    exp_mut : np.ndarray
        Expected mismatch counts for each pair of sequences, shape (N, N).
    
    eff_len : np.ndarray
        Effective length of each pair of sequences, shape (N, N).
    
    dist : Optional[np.ndarray]
        Normalized expected mismatch fraction in [0,1] if return_norm=True,
        shape (N, N). Otherwise None.
    """
    # Stack to (N, L, C)
    P = np.asarray(aligned, dtype=np.float64)
    if P.ndim != 3:
        raise ValueError("aligned must stack to shape (N, L, A+1)")
    N, L, C = P.shape
    if C < 2:
        raise ValueError("last axis must have >=2 channels (>=1 AA + gap)")

    # Split AA vs gap
    gap_idx = gap_at if gap_at >= 0 else (C + gap_at)
    if not (0 <= gap_idx < C):
        raise ValueError("gap_at out of range for last axis")
    P_gap = P[..., gap_idx] # (N, L)
    P_aa  = np.delete(P, gap_idx, axis=2) # (N, L, A)

    # Probability each sequence is non-gap at column
    Wcol = (1.0 - P_gap) # (N, L)

    exp_mut = np.zeros((N, N), dtype=np.float64)
    eff_len = np.zeros((N, N), dtype=np.float64)

    B = block_cols or L
    for s in range(0, L, B):
        e = min(s + B, L)
        Pa = P_aa[:, s:e, :] # (N, b, A)
        W  = Wcol[:, s:e] # (N, b)s

        id_batch = np.einsum("nka,mka->nmk", Pa, Pa, optimize=True)

        # joint non-gap weight per column to (N,N,b)
        w_batch = np.einsum("nc,mc->nmc", W, W, optimize=True)

        # expected mismatches per column is w * (1 - id)
        exp_mut += np.sum(w_batch * (1.0 - id_batch), axis=2)
        eff_len += np.sum(w_batch, axis=2)

    if return_norm:
        dist = exp_mut / np.maximum(eff_len, eps)
        np.clip(dist, 0.0, 1.0, out=dist)
        return exp_mut, eff_len, dist
    else:
        return exp_mut, eff_len, None

def _ensure_gapped_last(arr: np.ndarray,
                        *,
                        eps: float = 1e-12) -> np.ndarray:
    """
    Helper function to ensure array is (L, A+1) with a final gap channel.
    If input is (L, A) (ungapped), append gap = 1 - sum(AA) (clipped).

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (L, A) or (L, A+1) where
        A is the number of amino acids (20 or 21 including gap).
    
    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.
    
    Returns
    -------
    np.ndarray
        Array of shape (L, A+1) with gap channel appended if needed.
    """
    if arr.ndim != 2:
        raise ValueError("sequence array must be 2-D (L, A or L, A+1)")
    L, C = arr.shape
    if C >= 21:
        return arr
    aa_sum = arr.sum(axis=1, keepdims=True)
    gap = np.clip(1.0 - aa_sum, 0.0, 1.0)
    return np.concatenate([arr, gap], axis=1)

def _expected_mutations_from_aligned(Au: np.ndarray,
                                     Av: np.ndarray,
                                     *,
                                     gap_at: int = -1,
                                     eps: float = 1e-12) -> Tuple[float, float, float]:
    """
    Helper function to compute the expected Hamming between two aligned
    soft sequences.

    Parameters
    ----------
    Au : np.ndarray
        Aligned soft sequence for u, shape (L, A+1) where A is the number of amino acids (20 or 21 including gap).
    
    Av : np.ndarray
        Aligned soft sequence for v, same shape as Au. shape must match Au.

    gap_at : int, default=-1
        Index of the gap channel in the last axis of Au/Av.
        If negative, counts from the end (-1 = last channel). Default behaviour is compatible with PROT_20.

    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.

    Returns
    -------
    Tuple[float, float, float]
        Expected mutation count, effective length, and normalized mismatch fraction.
    """
    Pu = _ensure_gapped_last(np.asarray(Au, float))
    Pv = _ensure_gapped_last(np.asarray(Av, float))

    if Pu.shape != Pv.shape:
        raise ValueError("Aligned arrays for a pair must have the same shape")

    L, C = Pu.shape
    gap_idx = gap_at if gap_at >= 0 else (C + gap_at)
    if not (0 <= gap_idx < C):
        raise ValueError("gap_at out of range")

    # split
    pu_gap = Pu[:, gap_idx] # (L,)
    pv_gap = Pv[:, gap_idx]
    pu_aa  = np.delete(Pu, gap_idx, axis=1) # (L, A)
    pv_aa  = np.delete(Pv, gap_idx, axis=1)

    w = (1. - pu_gap) * (1. - pv_gap) # (L,) joint non-gap weight
    ident = np.sum(pu_aa * pv_aa, axis=1) # (L,) identity per col
    mut = np.sum(w * (1. - ident))
    eff = np.sum(w)
    dist = float(mut / max(eff, eps))
    return float(mut), float(eff), dist

def compute_edge_mutations_star(G: nx.Graph | nx.DiGraph,
                                *,
                                alphabet: List = PROT_20,
                                center_key: str = "ungapped_arr", # expects (L, A) unaligned sequences.
                                chunk_size: Optional[int] = None,
                                eps: float = 1e-12) -> None:
    """
    Function to compute the expected Hamming distance for each edge
    in a graph via star subgraph computation. Sets edge attributes in place. 
    The complexity is approximately sum_u cost(align([u] + N_u_chunk)).

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The graph to compute expected Hamming distances for.
    
    alphabet : List, default=PROT_20
        The alphabet used for alignment. Should be PROT_20 or a PAML alphabet.
    
    center_key : str, default="ungapped_arr"
        The key in G.nodes that contains the unaligned sequence array for each node.
        This should be a soft sequence array of shape (L, A) where A is the number of amino acids.

    chunk_size : Optional[int], default=None
        If set, process neighbors in chunks of this size to save memory.
        If None, process all neighbors at once.
    
    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.
    """
    # record which undirected edges have been computed.
    done = set()

    set_w = {}
    set_d = {}
    set_s = {}

    nodes = list(G.nodes())
    for u in nodes:
        # Gather neighbors needing (u,v)
        pending_vs = []
        for v in G.neighbors(u):
            e = (u, v) if G.is_directed() else tuple(sorted((u, v)))
            if e in done:
                continue
            if center_key not in G.nodes[u] or center_key not in G.nodes[v]:
                continue
            pending_vs.append(v)

        if not pending_vs:
            continue

        # chunk helper
        def _chunks(lst, k):
            if k is None or k <= 0:
                yield lst
            else:
                for i in range(0, len(lst), k):
                    yield lst[i:i+k]

        Pu = G.nodes[u][center_key]
        for chunk in _chunks(pending_vs, chunk_size):
            seqs = [Pu] + [G.nodes[v][center_key] for v in chunk]

            # Do one MSA for u and its neighbors.
            # NOTE: Should always be PROT_20 (or a PAML alphabet).
            aligned, _ = align_soft_sequences(sequences=seqs, alphabet=alphabet)

            Au = np.asarray(aligned[0])
            for i, v in enumerate(chunk, start=1):
                Av = np.asarray(aligned[i])
                mut, eff, dist = _expected_mutations_from_aligned(Au, Av)
                e_u_v = (u, v) if G.is_directed() else tuple(sorted((u, v)))
                done.add(e_u_v)
                set_w[(u, v)] = float(mut)
                set_d[(u, v)] = float(dist)
                set_s[(u, v)] = float(-np.log(max(dist, eps)))

    if set_w:
        nx.set_edge_attributes(G, set_w, "weight")
    if set_d:
        nx.set_edge_attributes(G, set_d, "dist")
    if set_s:
        nx.set_edge_attributes(G, set_s, "sim")

def attach_expected_hamming_to_edges(G: nx.Graph | nx.DiGraph,
                                     aligned: Sequence[np.ndarray],
                                     node_order: Optional[Sequence] = None,
                                     *,
                                     gap_at: int = -1,
                                     eps: float = 1e-12) -> None:
    """
    Function to attach expected Hamming edge attributes to a graph from a 
    precomputed alignment of soft sequences. The expected Hamming distance
    is computed for each edge based on the aligned sequences.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        The graph to attach expected Hamming distances to.
    
    aligned : Sequence[np.ndarray]
        List of aligned soft sequences, each of shape (L_aln, A+1)
        where last axis is A amino acids + gap. Indices must match the node indices in G or the
        indices in `node_order`.
    
    node_order : Optional[Sequence], default=None
        If provided, specifies the order of nodes in G to match the aligned sequences.
    
    gap_at : int, default=-1
        Index of the gap channel in the last axis of aligned[i].
        If negative, counts from the end (-1 = last channel).
    
    eps : float, default=1e-12
        Small value to avoid division by zero in normalization.
    """
    if node_order is None:
        node_order = list(G.nodes())
    if len(node_order) != len(aligned):
        raise ValueError("node_order length must match number of aligned arrays")

    # compute global pairwise once
    exp_mut, eff_len, dist = pairwise_expected_hamming_from_aligned(
        aligned, gap_at=gap_at, return_norm=True
    )
    idx = {n: i for i, n in enumerate(node_order)}

    set_w = {}
    set_d = {}
    set_s = {}
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        w = float(exp_mut[i, j])
        d = float(dist[i, j])
        set_w[(u, v)] = w
        set_d[(u, v)] = d
        set_s[(u, v)] = float(-np.log(max(d, eps)))

    if set_w:
        nx.set_edge_attributes(G, set_w, "weight")
    if set_d:
        nx.set_edge_attributes(G, set_d, "dist")
    if set_s:
        nx.set_edge_attributes(G, set_s, "sim")