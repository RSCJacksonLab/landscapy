from pathlib import Path
from typing import Union, Dict, List, Literal
import numpy as np
import networkx as nx
from cogent3 import load_aligned_seqs, ArrayAlignment, PhyloNode, load_tree, get_app
import piqtree
from piqtree import Model
from piqtree.model import AaModel
import math
from cogent3.util.table import Table
from .sequence import SoftSequence, BaseNumpySequence
from .._const import ALPHABET_21, PROT_20
from ..phylo._sub_matrices import nq_pfam
from sklearn.neighbors import NearestNeighbors
from ..phylo.phylogenetic_asr import ASRConstructor
from ..utils import calculate_gapped_soft_score
from .graph import (
    _find_knn_balltree,
    _find_knn_faiss,
    _encode_multiallele,
    attach_expected_hamming_to_edges,
    compute_edge_mutations_star
)

from ..embedding.particle_sampler import (
    EvolutionParticleSampler,
    SequenceGenerator,
    ParentSelector,
    TopPSampler,
    ESMEmbedder
)
from softalign.soft_alignment import align_soft_sequences
import ray

def create_phylo_digraph(sequences: Union[Path, ArrayAlignment],
                         replacement_matrix: List[str] = ['NQ.pfam'],
                         model_fitting: bool = True,
                         _log_progress: bool = False,
                         _nested_parallel: bool = False,
                         *,
                         _compute_hamming_edges: bool = True,
                         **kwargs) -> nx.DiGraph:
    """
    Factory function to create a Directed acyclic graph using
    phylogenetic inference and ancestral sequence reconstruction. 

    Parameters
    ----------
    alignment : Path or Alignment
        The alignment of extant sequences to use for ASR and
        phylogenetic infernece.
    
    replacement_matrix : List, default=[`NQ.pfam`]
        List of replacement matrices to use for phylogenetic
        reconstruction. Must be an NQ non-equilibrium model.

    model_fitting : bool, default=`True`
        Whether to fit the ML model, using the model set defined in
        `replacement_matrix`.

    Returns
    -------
    G : nx.DiDraph
        The Directed graph output.
    """
    if not all('NQ' in model for model in replacement_matrix):
        raise ValueError('Expected non-equilibrium model for digraph construction, found equilibrium model.')

    constructor = ASRConstructor(sequences,
                                 replacement_matrix = replacement_matrix,
                                 model_fitting = model_fitting,
                                 _log_progress=_log_progress)
    # Construct digraph with `graph_type` flag.
    digraph = constructor.construct_dag(graph_type='directed')
    
    # Attach edge attributes.
    if _compute_hamming_edges:
        compute_edge_mutations_star(G=digraph, _log_progress=_log_progress, _nested_parallel=_nested_parallel)
    return digraph

# Remote ray function for evol alignment.
@ray.remote
def _score_pair(i, j, seq_i, seq_j, tau, Q):

    Ai = seq_i.posterior if isinstance(seq_i, SoftSequence) else seq_i.to_one_hot()
    Aj = seq_j.posterior if isinstance(seq_j, SoftSequence) else seq_j.to_one_hot()
    # Ensure float inputs for stability in softalign
    Ai = np.ascontiguousarray(np.asarray(Ai, dtype=np.float64))
    Aj = np.ascontiguousarray(np.asarray(Aj, dtype=np.float64))
    _res = align_soft_sequences(sequences=[Ai, Aj], alphabet=PROT_20)
    aligned = _res[0] if isinstance(_res, tuple) else _res
    score = calculate_gapped_soft_score(aligned_seq1=aligned[0], aligned_seq2=aligned[1], q=Q)
    
    return i, j, float(np.exp(score / tau))

#TODO: Add emergence time masking.
def create_evol_diffusion_digraph(sequences: List[BaseNumpySequence],
                                             embeddings: np.ndarray = None,
                                             replacement_matrix: np.ndarray = nq_pfam,
                                             k: int = 50,
                                             tiebuffer: int = 0,
                                             backend: Literal['auto', 'faiss', 'balltree'] = 'auto',
                                             index_type: Literal['hnsw', 'flat', 'ivf'] = 'hnsw',
                                             faiss_metric: Literal['ip', 'l2'] = 'ip',
                                             include_self: bool = False,
                                             use_gpu: bool = False,
                                             hnsw_M: int = 32,
                                             t: int = 5,
                                             tau: float = 1.0,
                                             connectivity_threshold: float = 1e-4,
                                             cpus: int = 1,
                                             *,
                                             _compute_hamming_edges: bool = True,
                                             **kwargs) -> nx.DiGraph:
    """
    Constructs a diffusion graph by scoring standard alignments with an
    asymmetric non-equilibrium replacement matrix.

    Parameters
    ----------
    sequences : List[BaseNumpySequence]
        The list of sequence in the landscape. 
    
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
    
    cpus : int, default=1.0
        The number of CPUs to paralellise the kernel alignment and
        distance computation over.

    Returns
    -------
    nx.Graph or nx.DiGraph
        The constructed graph.
    """
    # Type check alphabet first
    for seq in sequences:
        if seq.alphabet != PROT_20:
            raise ValueError("Sequence alpbahet must be PROT_20 for all entries.")
    
    n_sequences = len(sequences)
    if n_sequences == 0:
        return nx.DiGraph()
    
    # Secure OHE embeddings if not provided otherwise.
    if embeddings is None:
        embeddings, _ = _encode_multiallele(sequences)

    # Find kNN in embedding space to identify candidate pairs
    # Should scale in O(N*k)
    
    # Update value of k if too large.
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
    
    pairs_to_align = []
    for i in range(n_sequences):
        for j_idx in neighbor_indices[i]:
            if i != j_idx:
                
                pair = (i, j_idx)
                pairs_to_align.append(pair)
    
    # Remove duplicate pairs before aligning
    pairs_to_align = sorted(list(set(pairs_to_align)))
    
    # Make kernel matrix
    kernel_matrix = np.zeros((n_sequences, n_sequences))
    
    # Init ray
    if not ray.is_initialized():
        ray.init()
    
    # Compute in parallel.
    refs = [_score_pair.options(num_cpus=cpus).remote(i, j, sequences[i], sequences[j], tau, replacement_matrix)
            for (i, j) in pairs_to_align]
    
    for i, j, kv in ray.get(refs):
        kernel_matrix[i, j] = kv

    # Proceed with diffusion and graph construction (same as before)
    np.fill_diagonal(kernel_matrix, 0)
    
    row_sums = kernel_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = kernel_matrix / row_sums
    
    # Compute diffusion steps
    diffused_matrix = np.linalg.matrix_power(transition_matrix, t)
    
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n_sequences))
    
    rows, cols = np.where(diffused_matrix > connectivity_threshold)
    
    for i, j in zip(rows, cols):
        if i != j:
            digraph.add_edge(i, j, kernel_weight=diffused_matrix[i, j])
        
    for i, seq in enumerate(sequences):
        digraph.nodes[i]['sequence'] = seq

    # Optionally compute expected Hamming distances if available
    if _compute_hamming_edges and all(
        hasattr(seq, "ungapped_arr") and getattr(seq, "ungapped_arr", None) is not None
        and getattr(seq, "alphabet", None) == PROT_20
        for seq in sequences
    ):
        compute_edge_mutations_star(G=digraph)
    
    return digraph

def create_particle_filter_digraph(sequences: List[BaseNumpySequence],
                                   n_samples: int,
                                   traj_length: int,
                                   batch_size: int,
                                   max_state_size: int,
                                   _emb_array_key: str = 'emb_array',
                                   temperature: float = 1.0,
                                   top_p: float = 0.9,
                                   *,
                                   _compute_hamming_edges: bool = True,
                                   **kwargs) -> nx.DiGraph:
    """
    Factory function to create a directed graph using a Gibbs sampling
    approach based on a protein language model.

    Parameters
    ----------
    seed_sequences : List[BaseNumpySequence]
        The initial sequences to start the simulation.
    n_samples : int
        The number of child sequences to generate from each parent.
    traj_length : int
        The number of steps in the evolutionary trajectory.
    batch_size : int
        The batch size for the sequence generator.
    max_state_size : int
        The maximum number of parent nodes to select at each step.
    hmm_file : str, optional
        Path to an HMM file to use for the sequence space attractor.
    embedding_attribute : str, default='representation'
        The node attribute key for embeddings.
    temperature : float, default=1.0
        The temperature for the Top-p sampler.
    top_p : float, default=0.9
        The top-p value for the Top-p sampler.

    Returns
    -------
    DirectedFitnessLandscape
        The constructed directed fitness landscape.
    """
    selector = ParentSelector(max_state_size=max_state_size)
    embedder = ESMEmbedder(model_name=kwargs.get('model_name', "facebook/esm2_t6_8M_UR50D"))
    sampler = TopPSampler(temperature=temperature, top_p=top_p)
    generator = SequenceGenerator(embedder=embedder,sampler=sampler, batch_size=batch_size)

    evolution_exp = EvolutionParticleSampler(generator=generator,
                                             selector=selector,
                                             n_samples=n_samples,
                                             traj_length=traj_length)

    # Convert BaseNumpySequence to strings for sampling.
    evolution_exp.initialize(seed_sequences=[seq.to_str() for seq in sequences])
    evolution_exp.run()
    digraph = evolution_exp.G
    
    # Optionally compute expected Hamming distances if available
    if _compute_hamming_edges and all(
        hasattr(seq, "ungapped_arr") and getattr(seq, "ungapped_arr", None) is not None
        and getattr(seq, "alphabet", None) == PROT_20
        for _, seq in digraph.nodes(data='sequence')
    ):
        compute_edge_mutations_star(G=digraph)
    
    return digraph

    # TODO: Evolutionary velocity connectivity
