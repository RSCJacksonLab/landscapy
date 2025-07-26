from __future__ import annotations
import ray
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from .latent_alignment import RJMCMCAligner
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ..utils import cosine_similarity_matrix
import faiss


@ray.remote
def run_local_alignment_task(cluster_subgraphs: List[nx.Graph],
                             aligner_params: Dict) -> Tuple[nx.Graph, Dict[int, np.ndarray], Dict[int, list]]:
    """
    Executes the local RJMCMC alignment for a cluster of subgraphs.
    This is a Ray remote task to parallelize the local alignments.

    Parameters
    ----------
    cluster_subgraphs : List[nx.Graph]
        A list of subgraphs representing a cluster of nodes.
    aligner_params : Dict
        Parameters for the RJMCMCAligner.
    
    Returns
    -------
    Tuple[nx.Graph, Dict[int, np.ndarray], Dict[int, list]]
        A tuple containing:
        - The blueprint graph for the local alignment.
        - A mapping of node indices to latent space indices.
        - The order of nodes as seen by the aligner.
    """
    local_aligner = RJMCMCAligner(graphs=cluster_subgraphs, **aligner_params)
    local_aligner.sample()
    
    blueprint = local_aligner.latent_blueprint_graph()
    node_mapping = local_aligner.get_node_to_latent_mapping()
    
    # Capture the order of nodes as the aligner sees them
    node_order = {i: list(g.nodes()) for i, g in enumerate(cluster_subgraphs)}
    
    return blueprint, node_mapping, node_order


class HierarchicalRJMCMCAligner:
    """
    A hierarchical RJMCMC aligner that performs two-level graph alignment:
    1. Local alignments within clusters of nodes.
    2. Global meta-alignment across clusters.

    Attributes
    ----------
    graphs : List[nx.Graph]
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
                 graphs: List[nx.Graph],
                 aligner_params: dict,
                 local_cluster_threshold: float = 0.85,
                 global_bridge_threshold: float = 0.5,
                 emb_key: str = 'emb_arr') -> None:
        
        self.original_graphs = graphs
        self.aligner_params = aligner_params
        self.local_thresh = local_cluster_threshold
        self.global_thresh = global_bridge_threshold
        self.emb_key = emb_key
        self.K = len(graphs)

        if not ray.is_initialized():
            ray.init()

    def run_alignment(self) -> Tuple[nx.Graph, Dict[int, np.ndarray]]:
        """
        Executes the full, two-level hierarchical alignment process.

        Returns
        -------
        Tuple[nx.Graph, Dict[int, np.ndarray]]
            A tuple containing:
            - The final aligned graph.
            - A mapping of original graph nodes to latent space nodes.
        """
        # Local Alignments
        local_results = self._run_local_alignments()

        # Global Meta-Alignment
        meta_blueprint, meta_mappings = self._run_global_meta_alignment(local_results)

        # Stitch results into the final format
        final_graph, final_mappings = self._stitch_results(local_results, meta_blueprint, meta_mappings)

        return final_graph, final_mappings

    def _create_clusters(self) -> List[Dict]:
        """
        Method to create clusters of nodes based on cosine similarity
        of their embeddings across all graphs.

        Returns
        -------
        List[Dict]
            A list of clusters, where each cluster is a dictionary
            containing:
            - 'global_indices': Indices of nodes in the cluster.
            - 'node_backrefs': References to the original nodes in the
            graphs.
        """
        all_embeddings = []
        node_backrefs = [] 
        for k, G in enumerate(self.original_graphs):
            for node_id, data in G.nodes(data=True):
                all_embeddings.append(data[self.emb_key])
                node_backrefs.append((k, node_id))
        
        if not all_embeddings:
            return []
        
        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        all_embeddings /= np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        num_nodes, d = all_embeddings.shape

        index = faiss.IndexFlatIP(d)
        index.add(all_embeddings)

        k_neighbors = min(100, num_nodes)
        similarities, indices = index.search(all_embeddings, k_neighbors)
        row_indices = np.arange(num_nodes).repeat(k_neighbors)
        mask = (indices > -1) & (similarities >= self.local_thresh)
    
        rows = row_indices[mask.ravel()]
        cols = indices.ravel()[mask.ravel()] 
        
        adjacency_matrix = csr_matrix((np.ones_like(rows), (rows, cols)),
                                      shape=(num_nodes, num_nodes))
        adjacency_matrix_symmetric = adjacency_matrix + adjacency_matrix.T
        
        n_components, labels = connected_components(
            csgraph=adjacency_matrix_symmetric,
            directed=False,
            return_labels=True
        )

        clusters = [[] for _ in range(n_components)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        final_clusters = [
            {
                "global_indices": cluster_indices,
                "node_backrefs": [node_backrefs[i] for i in cluster_indices]
            }
            for cluster_indices in clusters if cluster_indices
        ]
        
        return final_clusters

    def _run_local_alignments(self) -> List[Dict]:
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
        
        futures = []
        for cluster_info in clusters:
            subgraphs = [nx.Graph() for _ in range(self.K)]
            for k, node_id in cluster_info["node_backrefs"]:
                node_data = self.original_graphs[k].nodes[node_id]
                subgraphs[k].add_node(node_id, **node_data)
            
            for k in range(self.K):
                original_subgraph = self.original_graphs[k].subgraph(subgraphs[k].nodes())
                subgraphs[k].add_edges_from(original_subgraph.edges())
            
            params = self.aligner_params.copy()
            params['seed'] = np.random.randint(1e6)
            
            futures.append(run_local_alignment_task.remote(subgraphs, params))
        
        ray_results = ray.get(futures)
        
        for i, (blueprint, node_mapping, node_order) in enumerate(ray_results):
            clusters[i]['blueprint'] = blueprint
            clusters[i]['node_mapping'] = node_mapping
            clusters[i]['node_order'] = node_order
        
        return clusters

    def _run_global_meta_alignment(self,
                                   local_results: List[Dict]) -> Tuple[nx.Graph, Dict[int, np.ndarray]]:
        """
        Method to run the global meta-alignment across clusters of
        nodes.

        Returns
        -------
        Tuple[nx.Graph, Dict[int, np.ndarray]]
            A tuple containing:
            - The meta blueprint graph for the global alignment.
            - A mapping of original graph nodes to latent space nodes.
        """

        if not local_results:
            
            # Return empty but correctly typed results
            return nx.Graph(), {k: np.array([]) for k in range(self.K)}

        meta_graphs = [nx.Graph() for _ in range(self.K)]
        
        for i, result in enumerate(local_results):
            cluster_embeddings = [self.original_graphs[k].nodes[node_id][self.emb_key] for k, node_id in result["node_backrefs"]]
            if not cluster_embeddings: continue
            
            mean_embedding = np.mean(cluster_embeddings, axis=0)
            graph_counts = [ref[0] for ref in result["node_backrefs"]]
            primary_graph_idx = max(set(graph_counts), key=graph_counts.count)

            meta_graphs[primary_graph_idx].add_node(i, **{self.emb_key: mean_embedding})
            
        num_clusters = len(local_results)
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                embs_i = [self.original_graphs[k].nodes[nid][self.emb_key] for k, nid in local_results[i]["node_backrefs"]]
                embs_j = [self.original_graphs[k].nodes[nid][self.emb_key] for k, nid in local_results[j]["node_backrefs"]]
                
                if not embs_i or not embs_j: continue
                
                sim_matrix = cosine_similarity_matrix(np.array(embs_i), np.array(embs_j))
                if np.max(sim_matrix) > self.global_thresh:
                    for k in range(self.K):
                         if meta_graphs[k].has_node(i) and meta_graphs[k].has_node(j):
                            meta_graphs[k].add_edge(i, j)

        total_meta_edges = sum(g.number_of_edges() for g in meta_graphs)
        if total_meta_edges == 0:

            # Create an empty blueprint graph but ensure it contains all the cluster nodes.
            meta_blueprint = nx.Graph()
            all_meta_nodes = set()
            for g in meta_graphs:
                all_meta_nodes.update(g.nodes())
            meta_blueprint.add_nodes_from(all_meta_nodes)
            
            # Create empty but correctly structured mapping dictionaries.
            meta_mappings = {k: np.empty((g.number_of_nodes(), 0)) for k, g in enumerate(meta_graphs)}
            return meta_blueprint, meta_mappings
        
        meta_aligner = RJMCMCAligner(graphs=meta_graphs, **self.aligner_params)
        meta_aligner.sample()

        return meta_aligner.latent_blueprint_graph(), meta_aligner.get_node_to_latent_mapping()

    def _stitch_results(self,
                        local_results: List[Dict],
                        meta_blueprint: nx.Graph,
                        meta_mappings: Dict[int, np.ndarray]) -> Tuple[nx.Graph, Dict[int, np.ndarray]]:
        """
        Method to stitch the local results and meta blueprint into a
        final graph and mappings.

        Parameters
        ----------
        local_results : List[Dict]
            A list of dictionaries containing the local alignment
            results.
        meta_blueprint : nx.Graph
            The meta blueprint graph from the global alignment.
        meta_mappings : Dict[int, np.ndarray]
            A mapping of original graph nodes to latent space nodes
            from the global alignment.
        
        Returns
        -------
        Tuple[nx.Graph, Dict[int, np.ndarray]]
            A tuple containing:
            - The final aligned graph.
            - A mapping of original graph nodes to latent space nodes.
        """
        final_graph = nx.Graph()
        local_to_global_nodemap = {}

        for i, result in enumerate(local_results):
            local_bp = result['blueprint']
            
            # Create a mapping for this cluster's nodes to unique global names
            node_rename_mapping = {node: f"c{i}_n{node}" for node in local_bp.nodes()}
            
            # Store this mapping for later use when adding bridge edges
            for local_node, global_name in node_rename_mapping.items():
                local_to_global_nodemap[(i, local_node)] = global_name
                
            # Create a new graph with the relabeled nodes
            relabeled_bp = nx.relabel_nodes(local_bp, node_rename_mapping, copy=True)
            
            # Compose the uniquely-named blueprint into the final graph
            final_graph = nx.compose(final_graph, relabeled_bp)



        for i, j in meta_blueprint.edges():
            nodes_i = local_results[i]["node_backrefs"]
            nodes_j = local_results[j]["node_backrefs"]
            embs_i = np.array([self.original_graphs[k].nodes[nid][self.emb_key] for k, nid in nodes_i])
            embs_j = np.array([self.original_graphs[k].nodes[nid][self.emb_key] for k, nid in nodes_j])

            sim_matrix = cosine_similarity_matrix(embs_i, embs_j)
            i_idx, j_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            
            # Find which local latent nodes these bridge nodes map to
            # This requires the node order from the local alignment
            original_node_i_id = nodes_i[i_idx][1]
            original_node_j_id = nodes_j[j_idx][1]
            
            graph_idx_i = nodes_i[i_idx][0]
            graph_idx_j = nodes_j[j_idx][0]
            
            # Find the local index for the original node ID
            local_node_idx_i = local_results[i]['node_order'][graph_idx_i].index(original_node_i_id)
            local_node_idx_j = local_results[j]['node_order'][graph_idx_j].index(original_node_j_id)
            
            local_latent_i = np.argmax(local_results[i]['node_mapping'][graph_idx_i][local_node_idx_i, :])
            local_latent_j = np.argmax(local_results[j]['node_mapping'][graph_idx_j][local_node_idx_j, :])

            global_node_i = local_to_global_nodemap.get((i, local_latent_i))
            global_node_j = local_to_global_nodemap.get((j, local_latent_j))

            if global_node_i and global_node_j and not final_graph.has_edge(global_node_i, global_node_j):
                final_graph.add_edge(global_node_i, global_node_j)

        final_latent_node_order = list(final_graph.nodes())
        global_name_to_final_idx = {name: i for i, name in enumerate(final_latent_node_order)}
        original_node_to_row_idx = {
            k: {node: i for i, node in enumerate(g.nodes())}
            for k, g in enumerate(self.original_graphs)
        }

        final_mappings = {k: np.zeros((len(g.nodes()), len(final_latent_node_order)))
                        for k, g in enumerate(self.original_graphs)}

        for cluster_idx, result in enumerate(local_results):
            num_local_latent = max([key[1] for key in local_to_global_nodemap if key[0] == cluster_idx] + [-1]) + 1
            local_latent_to_master_col = np.full(num_local_latent, -1, dtype=int)
            for local_latent_idx in range(num_local_latent):
                global_name = local_to_global_nodemap.get((cluster_idx, local_latent_idx))
                if global_name:
                    local_latent_to_master_col[local_latent_idx] = global_name_to_final_idx[global_name]
            for graph_idx, prob_matrix in result['node_mapping'].items():
                if prob_matrix.size == 0:
                    continue

                local_to_master_row = np.array([
                    original_node_to_row_idx[graph_idx][original_node_id]
                    for original_node_id in result['node_order'][graph_idx]
                ])

                local_rows, local_cols = prob_matrix.nonzero()
                probs = prob_matrix[local_rows, local_cols]

                master_rows = local_to_master_row[local_rows]
                master_cols = local_latent_to_master_col[local_cols]
                
                valid_mask = master_cols != -1
                if not np.any(valid_mask):
                    continue

                final_mappings[graph_idx][master_rows[valid_mask], master_cols[valid_mask]] = probs[valid_mask]
                
        return final_graph, final_mappings