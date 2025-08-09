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