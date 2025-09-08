from __future__ import annotations
import ray
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
from .latent_alignment import RJMCMCAligner
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from ..utils import cosine_similarity_matrix
# FAISS is optional; we provide a NumPy fallback if unavailable.
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None
    _FAISS_AVAILABLE = False
from tqdm import tqdm
import os
import inspect


@ray.remote
def run_local_alignment_task(cluster_subgraphs: List[Union[nx.Graph, nx.DiGraph]],
                             aligner_params: Dict,
                             _local_cpu_chains: int) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[int, np.ndarray], Dict[int, list]]:
    """
    Executes the local RJMCMC alignment for a cluster of subgraphs.
    This is a Ray remote task to parallelize the local alignments.

    Parameters
    ----------
    cluster_subgraphs : List[nx.Graph, nx.DiGraph]
        A list of subgraphs representing a cluster of nodes.
    aligner_params : Dict
        Parameters for the RJMCMCAligner.
    _local_cpu_chains : int
        The number of chains to run in each parallel RJMCMC object.

    Returns
    -------
    Tuple[nx.Graph or nx.DiGraph, Dict[int, np.ndarray], Dict[int, list]]
        A tuple containing:
        - The blueprint graph for the local alignment.
        - A mapping of node indices to latent space indices.
        - The order of nodes as seen by the aligner.
    
    trace_E : List
        The local aligner energy trace over sampling steps. 
    
    trace_NL : List
        The local alginer number of latent nodes over sampling steps. 
    
    trace_edges : List
        The local aligner number of edges of sampling steps. 
    """
    
    # Make parameters safe for RJMCMC init.
    aligner_params = _filter_kwargs_for_init(RJMCMCAligner, dict(aligner_params))
    
    # Init and run chains in parallel.
    local_aligner = RJMCMCAligner(graphs=cluster_subgraphs, **aligner_params)
    local_aligner.sample(num_chains=_local_cpu_chains)
    
    blueprint = local_aligner.latent_blueprint_graph()
    node_mapping = local_aligner.get_node_to_latent_mapping()
    
    # Capture the order of nodes as the aligner sees them
    node_order = {i: list(g.nodes()) for i, g in enumerate(cluster_subgraphs)}
    
    return blueprint, node_mapping, node_order, local_aligner.trace_E, local_aligner.trace_NL, local_aligner.trace_edges, local_aligner._stored_L, local_aligner._stored_pi


def _filter_kwargs_for_init(cls, kwargs: Dict) -> Dict:
    """
    Drop kwargs not accepted by cls.__init__.
    """
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    # ignore `self`
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}

class HierarchicalRJMCMCAligner:
    """
    A hierarchical RJMCMC aligner that performs two-level graph alignment:
    1. Local alignments within clusters of nodes.
    2. Global meta-alignment across clusters.

    Attributes
    ----------
    graphs : List[nx.Graph, nx.DiGraph]
        A list of graphs to be aligned.
    aligner_params : dict
        Parameters for the RJMCMCAligner.
    Local_cluster_threshold : float, default=0.85
        Threshold for local cluster similarity.
    global_bridge_threshold : float, default=0.5
        Threshold for global bridge similarity.
    emb_key : str, default='emb_arr'
        The key in the graph nodes' data that containsthe embedding
        array.
    """

    def __init__(self,
                 graphs: List[Union[nx.Graph, nx.DiGraph]],
                 aligner_params: dict,
                 local_cluster_threshold: float = 0.85,
                 global_bridge_threshold: float = 0.5,
                 emb_key: str = 'emb_arr',
                 # Overlapping window controls
                 local_window_shifts: int = 0,
                 local_window_size: Optional[int] = None,
                 local_window_stride: Optional[int] = None,
                 _local_cpu_chains: int = (os.cpu_count()//10 if os.cpu_count()//10 > 1 else 1),
                 _meta_cpu_chains: int = os.cpu_count(),
                 _local_desc: str = "Local alignments",
                 _show_progress: bool = False,
                 _checkpoint_dir: str | None = None,
                 _checkpoint_interval: int = 300,
                 _resume_checkpoint: str | None = None
                 ) -> None:
        
        
        self.original_graphs = graphs
        # Filter params to what RJMCMCAligner.__init__ accepts (drops private flags like _show_progress)
        self.aligner_params = _filter_kwargs_for_init(RJMCMCAligner, dict(aligner_params))
        self.local_thresh = local_cluster_threshold
        self.global_thresh = global_bridge_threshold
        self.emb_key = emb_key
        # Overlap window params (allow overriding via aligner_params as well)
        _ap = dict(aligner_params) if isinstance(aligner_params, dict) else {}
        _shifts = local_window_shifts if local_window_shifts is not None else _ap.get('local_window_shifts', 0)
        _wsize = local_window_size if local_window_size is not None else _ap.get('local_window_size', None)
        _wstride = local_window_stride if local_window_stride is not None else _ap.get('local_window_stride', None)
        self.local_window_shifts = int(_shifts) if _shifts is not None else 0
        self.local_window_size = _wsize
        self.local_window_stride = _wstride
        self.K = len(graphs)
        self.directed = any(isinstance(g, nx.DiGraph) for g in graphs)
        
        # Parallel settings and reporting
        self._local_desc = _local_desc
        self._local_cpu_chains = _local_cpu_chains
        self._meta_cpu_chains = _meta_cpu_chains
        self._show_progress = _show_progress
        self._checkpoint_dir = _checkpoint_dir
        self._checkpoint_interval = _checkpoint_interval
        self._resume_checkpoint = _resume_checkpoint

        # Update aligner parameters with directed flag.
        if 'directed' not in self.aligner_params:
            self.aligner_params['directed'] = self.directed

        if not ray.is_initialized():
            ray.init()

        # Initialise local trace storing.
        self.local_energy_traces = {}
        self.local_nl_traces = {}
        self.local_edges_traces = {}

        # Initialise storage of posterior samples.
        self.local_posterior_L = []
        self.local_posterior_pi = []
        self.meta_posterior_L = []
        self.meta_posterior_pi = []

        self.full_posterior_L = []
        self.full_posterior_mappings = []

        # Initialise meta trace storing
        self.meta_energy_trace = []
        self.meta_nl_trace = []
        self.meta_edges_trace = []
        # Global slot union mapping built by overlap meta step
        self._slot_union_map: Dict[Tuple[int, int], int] = {}

    def run_alignment(self) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[int, np.ndarray]]:
        """
        Executes the full, two-level hierarchical alignment process.

        Returns
        -------
        Tuple[nx.Graph or nx.DiGraph, Dict[int, np.ndarray]]
            A tuple containing:
            - The final aligned graph.
            - A mapping of original graph nodes to latent space nodes.
        """
        # Local Alignments
        
        # Paralleliztion at level of both hierarchical orchestration and RJMCMC class (self._local_cpu_chains.)
        local_results = self._run_local_alignments(num_chains_per_task=self._local_cpu_chains)

        # Global Meta-Alignment
        # Parallelization at level of RJMCMC class and self._meta_cpu_chains.
        meta_blueprint, meta_mappings = self._run_global_meta_alignment(local_results)

        # Collect full posterior samples.
        self._reconstruct_and_store_full_posterior(local_results, meta_blueprint, meta_mappings)

        # Stitch results into the final format
        final_graph, final_mappings = self._stitch_results(local_results, meta_blueprint, meta_mappings)

        return final_graph, final_mappings

    def _create_clusters(self) -> List[Dict]:
        """
        Create local alignment groups. Two modes:
        - Overlap window mode (if local_window_shifts>0): build multiple shifted windows
          over a 1D ordering of all nodes to create overlapping groups.
        - Default: cosine-similarity graph + connected components (original behavior).

        Returns
        -------
        List[Dict]
            Each cluster dict has keys:
            - 'global_indices': indices into the flattened node list
            - 'node_backrefs': list of (graph_idx, node_id) for this cluster
        """
        # Flatten nodes and embeddings
        all_embeddings: List[np.ndarray] = []
        node_backrefs: List[Tuple[int, int]] = []
        for k, G in enumerate(self.original_graphs):
            for node_id, data in G.nodes(data=True):
                all_embeddings.append(np.asarray(data[self.emb_key], dtype=np.float32))
                node_backrefs.append((k, node_id))

        if not all_embeddings:
            return []

        E = np.vstack(all_embeddings).astype(np.float32)

        # Overlapping windows mode
        if self.local_window_shifts and self.local_window_shifts > 0:
            # Build a sparse kNN graph (cosine similarity), then compute MST and DFS order
            N, d = E.shape
            if N == 0:
                return []
            # Normalize for cosine
            E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            k_mst = min(30, N - 1) if N > 1 else 0
            if k_mst <= 0:
                order = np.arange(N)
            else:
                # Helper: top-k cosine neighbors using FAISS if available, else NumPy
                def _topk_cosine(E_norm_arr: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
                    if _FAISS_AVAILABLE:
                        index = faiss.IndexFlatIP(E_norm_arr.shape[1])
                        index.add(E_norm_arr)
                        return index.search(E_norm_arr, k + 1)
                    # NumPy fallback (O(N^2) memory; suitable for small N or environments without FAISS)
                    S = E_norm_arr @ E_norm_arr.T
                    # For each row, select top k+1 indices (including self)
                    Nn = S.shape[0]
                    kk = min(k + 1, Nn)
                    idx_part = np.argpartition(S, -kk, axis=1)[:, -kk:]
                    # Sort those top-k by similarity descending
                    row_indices = np.arange(Nn)[:, None]
                    part_vals = S[row_indices, idx_part]
                    order_desc = np.argsort(-part_vals, axis=1)
                    top_idx = idx_part[row_indices, order_desc]
                    top_sim = S[row_indices, top_idx]
                    return top_sim, top_idx

                sims, nbrs = _topk_cosine(E_norm, k_mst)
                # Build symmetric sparse distance matrix with weights = 1 - cosine
                rows, cols, data = [], [], []
                for i in range(N):
                    for c in range(1, k_mst + 1):
                        j = int(nbrs[i, c])
                        if j < 0 or j >= N or j == i:
                            continue
                        w = 1.0 - float(sims[i, c])
                        if w < 0:
                            w = 0.0
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([w, w])
                if rows:
                    from scipy import sparse as _sp
                    G_sparse = _sp.csr_matrix((data, (rows, cols)), shape=(N, N))
                    mst = minimum_spanning_tree(G_sparse)
                    T = (mst + mst.T).tocsr()
                    # Convert to NetworkX and DFS order; handle multiple components
                    Gnx = nx.Graph()
                    Gnx.add_nodes_from(range(N))
                    ti, tj = T.nonzero()
                    for a, b in zip(ti, tj):
                        if a < b:
                            Gnx.add_edge(int(a), int(b), weight=float(T[a, b]))
                    order_list = []
                    for comp in nx.connected_components(Gnx):
                        comp_nodes = list(comp)
                        # choose a root with highest degree in component
                        root = max(comp_nodes, key=lambda u: Gnx.degree(u)) if comp_nodes else None
                        if root is None:
                            continue
                        order_list.extend(list(nx.dfs_preorder_nodes(Gnx, source=root)))
                    order = np.array(order_list, dtype=int)
                    if order.size != N:
                        # Fallback: append any missing nodes
                        missing = np.setdiff1d(np.arange(N), order, assume_unique=False)
                        order = np.concatenate([order, missing])
                else:
                    order = np.arange(N)
            N = len(order)
            # Heuristic defaults
            w = int(self.local_window_size) if self.local_window_size else max(20, min(200, N // 20 if N >= 200 else max(10, N // 5)))
            s = int(self.local_window_stride) if self.local_window_stride else max(1, w // 2)
            S = max(1, int(self.local_window_shifts))

            clusters: List[Dict] = []
            for shift in range(S):
                offset = (shift * s) // S
                start = offset
                while start < N:
                    end = min(N, start + w)
                    if end - start <= 0:
                        break
                    idxs = order[start:end]
                    backrefs = [node_backrefs[i] for i in idxs]
                    clusters.append({
                        "global_indices": idxs.tolist(),
                        "node_backrefs": backrefs,
                    })
                    start += s

            # Deduplicate exact duplicate windows
            seen = set()
            uniq = []
            for c in clusters:
                key = tuple(sorted(c["global_indices"]))
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(c)
            return uniq

        # Default: similarity graph + connected components
        E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        num_nodes = E_norm.shape[0]
        k_neighbors = min(100, num_nodes)
        # Top-k neighbor search with FAISS if available, else NumPy fallback
        if _FAISS_AVAILABLE:
            d = E_norm.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(E_norm)
            similarities, indices = index.search(E_norm, k_neighbors)
        else:  # pragma: no cover
            S = E_norm @ E_norm.T
            kk = min(k_neighbors, num_nodes)
            idx_part = np.argpartition(S, -kk, axis=1)[:, -kk:]
            row_indices2 = np.arange(num_nodes)[:, None]
            part_vals2 = S[row_indices2, idx_part]
            order_desc2 = np.argsort(-part_vals2, axis=1)
            indices = idx_part[row_indices2, order_desc2]
            similarities = S[row_indices2, indices]
        row_indices = np.arange(num_nodes).repeat(k_neighbors)
        mask = (indices > -1) & (similarities >= self.local_thresh)
        rows = row_indices[mask.ravel()]
        cols = indices.ravel()[mask.ravel()]
        adjacency_matrix = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(num_nodes, num_nodes))
        adjacency_matrix_symmetric = adjacency_matrix + adjacency_matrix.T
        n_components, labels = connected_components(csgraph=adjacency_matrix_symmetric, directed=False, return_labels=True)
        comps: List[List[int]] = [[] for _ in range(n_components)]
        for i, lab in enumerate(labels):
            comps[lab].append(i)
        final_clusters = [{
            "global_indices": comp,
            "node_backrefs": [node_backrefs[i] for i in comp]
        } for comp in comps if comp]
        return final_clusters

    def _run_local_alignments(self,
                              num_chains_per_task: int=1) -> List[Dict]:
        """
        Method to run local RJMCMC alignments for each cluster of
        nodes.

        Returns
        -------
        List[Dict]
            A list of dictionaries, where each dictionary contains:
            - 'blueprint': The blueprint graph for the local alignment.
            - 'node_mapping': A mapping of node indices to latent space
            indices.
            - 'node_order': The order of nodes as seen by the aligner.
        """
        clusters = self._create_clusters()

        # Prepare storage aligned by cluster index for posterior samples
        self.local_posterior_L = [None] * len(clusters)
        self.local_posterior_pi = [None] * len(clusters)

        futures = []
        # Resume support: load existing results if provided
        completed: dict[int, Dict] = {}
        if self._resume_checkpoint:
            try:
                import pickle as _pickle
                with open(self._resume_checkpoint, 'rb') as f:
                    ck = _pickle.load(f)
                if isinstance(ck, dict) and 'local_results' in ck:
                    for idx, res in ck['local_results'].items():
                        if isinstance(idx, int):
                            completed[idx] = res
            except Exception:
                pass

        # keep a stable index for each cluster (preserve order on output)
        for cluster_idx, cluster_info in enumerate(clusters):
            if cluster_idx in completed:
                continue
            graph_constructor = nx.DiGraph if self.directed else nx.Graph
            subgraphs = [graph_constructor() for _ in range(self.K)]

            for k, node_id in cluster_info["node_backrefs"]:
                node_data = self.original_graphs[k].nodes[node_id]
                subgraphs[k].add_node(node_id, **node_data)

            for k in range(self.K):
                original_subgraph = self.original_graphs[k].subgraph(subgraphs[k].nodes())
                subgraphs[k].add_edges_from(original_subgraph.edges())

            params = self.aligner_params.copy()
            params['seed'] = int(np.random.randint(1e6))
            params['directed'] = self.directed
            params['num_chains'] = num_chains_per_task

            num_cpus_needed = 1 + num_chains_per_task
            obj_ref = run_local_alignment_task.options(num_cpus=num_cpus_needed).remote(subgraphs, params, self._local_cpu_chains)
            futures.append(obj_ref)

        # Prepare output container in input order
        results_in_order: List[Dict] = [dict(clusters[i]) for i in range(len(clusters))]
        # seed any resumed results into output container
        for idx, res in completed.items():
            results_in_order[idx].update(res)

        # Map ObjectRef : cluster_idx.
        ref_to_idx: Dict[ray.ObjectRef, int] = {ref: i for i, ref in enumerate(futures)}
        pending = set(futures)

        with tqdm(total=len(pending),
                desc=self._local_desc,
                disable=not self._show_progress) as pbar:
            while pending:
                done, pending = ray.wait(list(pending), num_returns=1, timeout=None)
                ref = done[0]
                idx = ref_to_idx[ref]
                try:
                    (blueprint, node_mapping, node_order,
                    trace_E, trace_NL, trace_edges,
                    stored_L, stored_pi) = ray.get(ref)
                except Exception as e:
                    raise RuntimeError(f"Local alignment failed for cluster {idx}") from e

                # Store into the stable slot
                results_in_order[idx]['blueprint'] = blueprint
                results_in_order[idx]['node_mapping'] = node_mapping
                results_in_order[idx]['node_order'] = node_order

                # posterior samples for local alignments (store by cluster index)
                self.local_posterior_L[idx] = stored_L
                self.local_posterior_pi[idx] = stored_pi

                # traces
                self.local_energy_traces[idx] = trace_E
                self.local_nl_traces[idx] = trace_NL
                self.local_edges_traces[idx] = trace_edges

                pbar.update(1)

                # checkpoint locals after each result
                if self._checkpoint_dir:
                    try:
                        import pickle as _pickle, time as _time
                        os.makedirs(self._checkpoint_dir, exist_ok=True)
                        path = os.path.join(self._checkpoint_dir, 'hier_local.ckpt.pkl')
                        local_pack = {}
                        for i, r in enumerate(results_in_order):
                            if 'blueprint' in r:
                                local_pack[i] = {
                                    'blueprint': r['blueprint'],
                                    'node_mapping': r.get('node_mapping'),
                                    'node_order': r.get('node_order'),
                                }
                        with open(path, 'wb') as f:
                            _pickle.dump({'local_results': local_pack, 'ts': _time.time()}, f)
                    except Exception:
                        pass

        return results_in_order

    def _run_global_meta_alignment(self,
                                   local_results: List[Dict]) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[int, np.ndarray]]:
        """
        Overlap-based meta-alignment using posterior probabilities from local RJMCMC.

        For each pair of overlapping local groups (windows), compute a slot co-assignment
        matrix from shared original nodes and match slots (Hungarian/greedy) to form
        equivalence classes (union-find). These define global latent slot IDs across all
        windows. No separate meta RJMCMC is performed in this mode.
        """
        graph_constructor = nx.DiGraph if self.directed else nx.Graph
        if not local_results:
            return graph_constructor(), {k: np.array([]) for k in range(self.K)}

        def _num_local_slots(res: Dict) -> int:
            bp = res.get('blueprint')
            if bp is not None:
                return bp.number_of_nodes()
            mx = 0
            for pm in res.get('node_mapping', {}).values():
                if pm is not None and pm.size > 0:
                    mx = max(mx, pm.shape[1])
            return mx

        # Quick lookup for (graph_idx,node_id)->(graph_idx,row) within each group
        node_to_localrow: List[Dict[Tuple[int, int], Tuple[int, int]]] = []
        for res in local_results:
            mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for gidx, order in res.get('node_order', {}).items():
                for row, nid in enumerate(order):
                    mapping[(gidx, nid)] = (gidx, row)
            node_to_localrow.append(mapping)

        # Union-find over (group_idx, local_slot)
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        def find(x: Tuple[int, int]) -> Tuple[int, int]:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a: Tuple[int, int], b: Tuple[int, int]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        try:
            from scipy.optimize import linear_sum_assignment
        except Exception:
            linear_sum_assignment = None

        N = len(local_results)
        for i in range(N):
            res_i = local_results[i]
            slots_i = _num_local_slots(res_i)
            if slots_i <= 0:
                continue
            set_i = set(res_i['node_backrefs'])
            for j in range(i + 1, N):
                res_j = local_results[j]
                slots_j = _num_local_slots(res_j)
                if slots_j <= 0:
                    continue
                set_j = set(res_j['node_backrefs'])
                overlap = list(set_i.intersection(set_j))
                if not overlap:
                    continue

                C = np.zeros((slots_i, slots_j), dtype=np.float64)
                total_mass = 0.0
                for (gidx, nid) in overlap:
                    gi, row_i = node_to_localrow[i].get((gidx, nid), (None, None))
                    gj, row_j = node_to_localrow[j].get((gidx, nid), (None, None))
                    if gi is None or gj is None:
                        continue
                    Pi = res_i['node_mapping'].get(gi)
                    Pj = res_j['node_mapping'].get(gj)
                    if Pi is None or Pj is None or Pi.size == 0 or Pj.size == 0:
                        continue
                    if row_i >= Pi.shape[0] or row_j >= Pj.shape[0]:
                        continue
                    p_i = Pi[row_i]
                    p_j = Pj[row_j]
                    if p_i.size != slots_i:
                        _pi = np.zeros(slots_i)
                        _pi[:min(slots_i, p_i.size)] = p_i[:min(slots_i, p_i.size)]
                        p_i = _pi
                    if p_j.size != slots_j:
                        _pj = np.zeros(slots_j)
                        _pj[:min(slots_j, p_j.size)] = p_j[:min(slots_j, p_j.size)]
                        p_j = _pj
                    C += np.outer(p_i, p_j)
                    total_mass += float(p_i.sum() * p_j.sum())

                if total_mass <= 0.0:
                    continue

                if linear_sum_assignment is not None:
                    m, n = C.shape
                    sz = max(m, n)
                    Cp = np.zeros((sz, sz), dtype=np.float64)
                    Cp[:m, :n] = C
                    cost = 1.0 - (Cp / max(total_mass, 1e-12))
                    r, c = linear_sum_assignment(cost)
                    pairs = [(ri, ci) for ri, ci in zip(r, c) if ri < m and ci < n]
                else:
                    pairs = []
                    used = set()
                    for ri in range(C.shape[0]):
                        cj = int(np.argmax(C[ri]))
                        if cj in used:
                            continue
                        used.add(cj)
                        pairs.append((ri, cj))

                for si, sj in pairs:
                    score = C[si, sj] / max(total_mass, 1e-12)
                    if score >= self.global_thresh:
                        union((i, si), (j, sj))

        # Assign global IDs; include isolated slots
        gid = 0
        slot_union_map: Dict[Tuple[int, int], int] = {}
        roots: Dict[Tuple[int, int], int] = {}
        for key in list(parent.keys()):
            r = find(key)
            if r not in roots:
                roots[r] = gid
                gid += 1
            slot_union_map[key] = roots[r]
        for i, res in enumerate(local_results):
            L = _num_local_slots(res)
            for s_idx in range(L):
                key = (i, s_idx)
                if key not in slot_union_map:
                    slot_union_map[key] = gid
                    gid += 1

        self._slot_union_map = slot_union_map

        # Return empty meta artefacts; stitching will use union map
        meta_blueprint = graph_constructor()
        meta_mappings = {k: np.array([]) for k in range(self.K)}
        return meta_blueprint, meta_mappings

    def _stitch_results(self,
                        local_results: List[Dict],
                        meta_blueprint: Union[nx.Graph, nx.DiGraph],
                        meta_mappings: Dict[int, np.ndarray],
                        sample_idx: Optional[int] = None) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[int, np.ndarray]]:
        """
        Method to stitch the local results and meta blueprint into a
        final graph and mappings. Can be used for both the posterior mean and for
        individual posterior samples.

        Parameters
        ----------
        local_results : List[Dict]
            A list of dictionaries containing the local alignment
            results.
        meta_blueprint : nx.Graph or nx.DiGraph
            The meta blueprint graph from the global alignment.
        meta_mappings : Dict[int, np.ndarray]
            A mapping of original graph nodes to latent space nodes
            from the global alignment.
        sample_idx : int, optional
            The index of the posterior sample to use. If None, the posterior mean is used.
        
        Returns
        -------
        Tuple[nx.Graph or nx.DiGraph, Dict[int, np.ndarray]]
            A tuple containing:
            - The final aligned graph.
            - A mapping of original graph nodes to latent space nodes.
        """
        final_graph = nx.DiGraph() if self.directed else nx.Graph()
        local_to_global_nodemap: Dict[Tuple[int, int], Union[str, int]] = {}
        use_union = hasattr(self, '_slot_union_map') and bool(getattr(self, '_slot_union_map'))

        # Compose local blueprints, relabeled to global IDs if available
        for i, result in enumerate(local_results):
            if sample_idx is not None and i < len(self.local_posterior_L):
                stored_list = self.local_posterior_L[i]
                if stored_list and len(stored_list) > 0:
                    s_idx = sample_idx % len(stored_list)
                    try:
                        local_bp_matrix = stored_list[s_idx]
                        local_bp = nx.from_numpy_array(local_bp_matrix, create_using=nx.DiGraph if self.directed else nx.Graph)
                    except Exception:
                        local_bp = result['blueprint']
                else:
                    local_bp = result['blueprint']
            else:
                local_bp = result['blueprint']

            node_rename_mapping: Dict[int, Union[str, int]] = {}
            for node in local_bp.nodes():
                if use_union:
                    gid = self._slot_union_map.get((i, int(node)))
                    name = f"g{gid}" if gid is not None else f"c{i}_n{node}"
                else:
                    name = f"c{i}_n{node}"
                node_rename_mapping[node] = name
                local_to_global_nodemap[(i, int(node))] = name

            relabeled_bp = nx.relabel_nodes(local_bp, node_rename_mapping, copy=True)
            final_graph = nx.compose(final_graph, relabeled_bp)

        # If union was not built, optionally add bridging edges based on meta blueprint
        if not use_union:
            for i, j in meta_blueprint.edges():
                nodes_i = local_results[i]["node_backrefs"]
                nodes_j = local_results[j]["node_backrefs"]
                embs_i = np.array([self.original_graphs[k].nodes[nid][self.emb_key] for k, nid in nodes_i])
                embs_j = np.array([self.original_graphs[k].nodes[nid][self.emb_key] for k, nid in nodes_j])
                sim_matrix = cosine_similarity_matrix(embs_i, embs_j)
                i_idx, j_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
                original_node_i_id = nodes_i[i_idx][1]
                original_node_j_id = nodes_j[j_idx][1]
                graph_idx_i = nodes_i[i_idx][0]
                graph_idx_j = nodes_j[j_idx][0]
                local_node_idx_i = local_results[i]['node_order'][graph_idx_i].index(original_node_i_id)
                local_node_idx_j = local_results[j]['node_order'][graph_idx_j].index(original_node_j_id)
                if sample_idx is not None:
                    use_sample = False
                    if i < len(self.local_posterior_pi) and j < len(self.local_posterior_pi):
                        Lpi_i = self.local_posterior_pi[i]
                        Lpi_j = self.local_posterior_pi[j]
                        if Lpi_i and Lpi_j and graph_idx_i in Lpi_i and graph_idx_j in Lpi_j:
                            li = Lpi_i[graph_idx_i]
                            lj = Lpi_j[graph_idx_j]
                            if li and lj and len(li) > 0 and len(lj) > 0:
                                s_idx = sample_idx % min(len(li), len(lj))
                                try:
                                    local_pi_i = li[s_idx]
                                    local_pi_j = lj[s_idx]
                                    local_latent_i = local_pi_i[local_node_idx_i]
                                    local_latent_j = local_pi_j[local_node_idx_j]
                                    use_sample = True
                                except Exception:
                                    use_sample = False
                    if not use_sample:
                        local_latent_i = np.argmax(local_results[i]['node_mapping'][graph_idx_i][local_node_idx_i, :])
                        local_latent_j = np.argmax(local_results[j]['node_mapping'][graph_idx_j][local_node_idx_j, :])
                else:
                    local_latent_i = np.argmax(local_results[i]['node_mapping'][graph_idx_i][local_node_idx_i, :])
                    local_latent_j = np.argmax(local_results[j]['node_mapping'][graph_idx_j][local_node_idx_j, :])

                global_node_i = local_to_global_nodemap.get((i, local_latent_i))
                global_node_j = local_to_global_nodemap.get((j, local_latent_j))
                if global_node_i and global_node_j and not final_graph.has_edge(global_node_i, global_node_j):
                    final_graph.add_edge(global_node_i, global_node_j)

        # Finalize node indices
        final_latent_node_order = list(final_graph.nodes())
        global_name_to_final_idx = {name: i for i, name in enumerate(final_latent_node_order)}
        original_node_to_row_idx = {
            k: {node: i for i, node in enumerate(g.nodes())}
            for k, g in enumerate(self.original_graphs)
        }

        final_mappings = {k: np.zeros((len(g.nodes()), len(final_latent_node_order)))
                        for k, g in enumerate(self.original_graphs)}
        contrib_counts = {k: np.zeros((len(g.nodes()),), dtype=np.float64) for k, g in enumerate(self.original_graphs)}

        # Scatter local probabilities into global columns (average and renormalize per row)
        for cluster_idx, result in enumerate(local_results):
            # Determine number of local latent slots
            num_local_latent = max([key[1] for key in local_to_global_nodemap if key[0] == cluster_idx] + [-1]) + 1
            # Map local latent -> master column index
            local_latent_to_master_col = np.full(num_local_latent, -1, dtype=int)
            for local_latent_idx in range(num_local_latent):
                global_name = local_to_global_nodemap.get((cluster_idx, local_latent_idx))
                if global_name:
                    local_latent_to_master_col[local_latent_idx] = global_name_to_final_idx[global_name]

            for graph_idx in range(len(self.original_graphs)):
                prob_matrix = result['node_mapping'].get(graph_idx)
                if prob_matrix is None or prob_matrix.size == 0:
                    continue
                local_to_master_row = np.array([
                    original_node_to_row_idx[graph_idx][original_node_id]
                    for original_node_id in result['node_order'][graph_idx]
                ])
                for r_local, r_master in enumerate(local_to_master_row):
                    row_probs = prob_matrix[r_local]
                    if row_probs.size != num_local_latent:
                        rp = np.zeros(num_local_latent)
                        rp[:min(num_local_latent, row_probs.size)] = row_probs[:min(num_local_latent, row_probs.size)]
                        row_probs = rp
                    nz = np.where(row_probs > 0)[0]
                    if nz.size == 0:
                        continue
                    for c_local in nz:
                        master_col = local_latent_to_master_col[c_local]
                        if master_col == -1:
                            continue
                        final_mappings[graph_idx][r_master, master_col] += row_probs[c_local]
                    contrib_counts[graph_idx][r_master] += 1.0

        for k in range(len(self.original_graphs)):
            cnt = contrib_counts[k]
            M = final_mappings[k]
            for r in range(M.shape[0]):
                if cnt[r] > 0:
                    M[r, :] /= cnt[r]
                s = M[r, :].sum()
                if s > 0:
                    M[r, :] /= s

        final_graph = nx.relabel_nodes(final_graph, global_name_to_final_idx, copy=True)

        return final_graph, final_mappings
    
    def _reconstruct_and_store_full_posterior(self,
                                              local_results,
                                              meta_blueprint,
                                              meta_mappings):
            """
            Helper method to reconstruct and store the full posterior samples
            of the final aligned graph and mappings by combining local and meta
            posterior samples.
            """
            num_samples = len(self.meta_posterior_L)
            if num_samples == 0:
                return # No samples to process

            for s_idx in range(num_samples):
                # Get the meta-graph for this specific sample
                meta_L_sample_matrix = self.meta_posterior_L[s_idx]
                meta_blueprint_sample = nx.from_numpy_array(
                    meta_L_sample_matrix, 
                    create_using=nx.DiGraph if self.directed else nx.Graph
                )

                # Use the stitching logic on this specific sample
                graph_sample, mappings_sample = self._stitch_results(
                    local_results,
                    meta_blueprint_sample,
                    # Pass the mean meta-mappings, _stitch_results will use sampled local pi
                    meta_mappings, 
                    sample_idx=s_idx)
                
                self.full_posterior_L.append(nx.to_numpy_array(graph_sample))
                self.full_posterior_mappings.append(mappings_sample)

    def latent_graph_from_local_posterior(self,
                                          posterior_prob_cutoff: float = 0.2) -> tuple[Union[nx.Graph, nx.DiGraph], Dict[int, np.ndarray]]:
        """
        Build a stitched latent graph at an arbitrary cutoff using only the
        stored LOCAL posterior samples (one posterior per overlapping window).

        This is useful in overlap-window mode where the meta stage does not
        produce global posterior samples. It reuses the previously computed
        union map (slot equivalence classes) and recomputes, per window, the
        local blueprint by averaging its stored adjacency samples and
        thresholding at the provided cutoff.

        Returns
        -------
        (graph, mappings): Tuple[nx.Graph|nx.DiGraph, Dict[int, np.ndarray]]
            The stitched latent graph and per-input-graph node->latent mapping.
        """
        # Recreate the cluster metadata (node_backrefs) deterministically
        clusters = self._create_clusters()
        graph_constructor = nx.DiGraph if self.directed else nx.Graph

        # Guard: require local posterior samples
        if not self.local_posterior_L or all((lst is None or len(lst) == 0) for lst in self.local_posterior_L):
            # Nothing to rebuild from; return empty graph and empty mappings
            empty = {k: np.array([]) for k in range(self.K)}
            return graph_constructor(), empty

        local_results: List[Dict] = []
        # Precompute original node -> row index for each graph
        orig_row_lut: Dict[int, Dict] = {
            k: {node: i for i, node in enumerate(g.nodes())}
            for k, g in enumerate(self.original_graphs)
        }

        for idx, cluster in enumerate(clusters):
            # 1) Build local blueprint by averaging posterior samples
            stored_L = self.local_posterior_L[idx]
            if stored_L is None or len(stored_L) == 0:
                bp = graph_constructor()  # empty
            else:
                max_nl = max(L.shape[0] for L in stored_L)
                tally = np.zeros((max_nl, max_nl), dtype=float)
                for L in stored_L:
                    nl = L.shape[0]
                    tally[:nl, :nl] += L
                Lavg = tally / max(1, len(stored_L))
                Lbin = (Lavg >= posterior_prob_cutoff).astype(int)
                bp = nx.from_numpy_array(Lbin, create_using=graph_constructor)

            # 2) Recompute node order for this cluster (match run_local_alignment_task)
            node_order: Dict[int, list] = {}
            for gidx in range(self.K):
                node_order[gidx] = [nid for (kk, nid) in cluster["node_backrefs"] if kk == gidx]

            # 3) Build node->latent probability mapping from stored permutations
            stored_pi = self.local_posterior_pi[idx]
            node_mapping: Dict[int, np.ndarray] = {}
            if stored_pi is None:
                stored_pi = [[] for _ in range(self.K)]

            # Determine max NL across samples for this cluster
            max_nl = 0
            for k in range(self.K):
                for pk in stored_pi[k]:
                    if pk is not None:
                        max_nl = max(max_nl, int(np.max(pk)) + 1 if pk.size else max_nl)
            # Fallback in case pk were empty
            if max_nl <= 0:
                max_nl = bp.number_of_nodes()

            for gidx in range(self.K):
                # Number of original nodes for this graph
                num_nodes = len(self.original_graphs[gidx].nodes())
                tally = np.zeros((num_nodes, max_nl), dtype=float)
                pis = stored_pi[gidx]
                if pis is None:
                    pis = []
                for pk in pis:
                    if pk is None or pk.size == 0:
                        continue
                    # pk is length equal to number of nodes in the subgraph; need to map to original rows
                    # Build a mapping from local row -> original node id from node_order
                    for local_row, latent_slot in enumerate(pk):
                        if latent_slot < 0:
                            continue
                        if local_row >= len(node_order[gidx]):
                            continue
                        orig_node = node_order[gidx][local_row]
                        orig_row = orig_row_lut[gidx].get(orig_node, None)
                        if orig_row is None:
                            continue
                        if latent_slot >= tally.shape[1]:
                            # expand columns if needed
                            extra = latent_slot + 1 - tally.shape[1]
                            tally = np.pad(tally, ((0, 0), (0, extra)), constant_values=0.0)
                        tally[orig_row, int(latent_slot)] += 1.0
                if len(pis) > 0:
                    tally /= float(len(pis))
                # Normalize each row to probabilistic mapping
                for r in range(tally.shape[0]):
                    s = tally[r].sum()
                    if s > 0:
                        tally[r] /= s
                node_mapping[gidx] = tally

            local_results.append({
                'blueprint': bp,
                'node_mapping': node_mapping,
                'node_order': node_order,
                'node_backrefs': cluster['node_backrefs'],
            })

        # Stitch using existing union map and return
        meta_bp = graph_constructor()
        meta_map = {k: np.array([]) for k in range(self.K)}
        final_graph, final_mappings = self._stitch_results(local_results, meta_bp, meta_map)
        return final_graph, final_mappings
