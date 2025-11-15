from pathlib import Path
from typing import Union, Dict, List, Literal, Optional, Sequence, Iterator, Hashable, Mapping, Any
from itertools import count
import numpy as np
import networkx as nx
import logging
from cogent3 import load_aligned_seqs, load_tree, get_app
from cogent3.core.alignment import Alignment
try:
    from cogent3.core.tree import PhyloNode
except Exception:
    from cogent3 import PhyloNode
import piqtree
from .sequence import SoftSequence, BaseNumpySequence
from .._const import ALPHABET_21, PROT_20
from ..phylo._sub_matrices import nq_pfam
from sklearn.neighbors import NearestNeighbors
from ..phylo.phylogenetic_asr import ASRConstructor
from ..utils import (
    calculate_gapped_soft_score,
    sequence_to_text,
    string_to_sequence,
    hamming_distance_str,
    resolve_plm_embedder,
)
from .annotation import register_auto_annotation
from .graph import (
    _find_knn_balltree,
    _find_knn_faiss,
    _encode_multiallele,
    create_knn_graph,
    attach_expected_hamming_to_edges,
    compute_edge_mutations_star,
    _ensure_ray_initialized,
    _compute_stationary_distribution,
    _should_use_stationary_power,
)

from ..embedding.particle_sampler import (
    EvolutionParticleSampler,
    SequenceGenerator,
    ParentSelector,
    TopPSampler,
    ESMEmbedder as SoftSamplerESMEmbedder,
)
from ..embedding.beam_search import (
    PseudoLogLikelihoodScorer,
    InterpolationBeamSearch,
)
from softalign.soft_alignment import align_soft_sequences
import ray

def create_phylo_digraph(sequences: Union[Path, Alignment],
                         replacement_matrix: List[str] = ['NQ.pfam'],
                         model_fitting: bool = True,
                         _log_progress: bool = False,
                         _nested_parallel: bool = False,
                         reconstruct_ancestral_states: bool = True,
                         *,
                         _compute_hamming_edges: bool = True,
                         _lightweight_nodes: bool = False,
                         _hard_ancestors: bool = False,
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
    
    reconstruct_ancestral_states : bool, default=`True`
        Whether to reconstruct ancestral amino-acid states. If False,
        internal nodes are populated with lightweight placeholders so the
        resulting graph remains usable for topology-focused analyses.

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
                                 reconstruct_ancestral_states=reconstruct_ancestral_states,
                                 _log_progress=_log_progress)
    # Construct digraph with `graph_type` flag.
    digraph = constructor.construct_dag(graph_type='directed')

    # Optionally strip heavy arrays and collapse ancestors to hard sequences
    if _lightweight_nodes or _hard_ancestors:
        from .sequence import SoftSequence, BaseNumpySequence
        for node, data in list(digraph.nodes(data=True)):
            if _lightweight_nodes:
                data.pop('gapped_arr', None)
            if _hard_ancestors and isinstance(data.get('sequence'), SoftSequence):
                hard_str = ''.join(map(str, data['sequence'].to_array()))
                data['sequence'] = BaseNumpySequence.from_string(hard_str, alphabet=PROT_20, moltype='protein', sequence_id=str(node))
    
    # Attach edge attributes.
    if _compute_hamming_edges:
        compute_edge_mutations_star(G=digraph, _log_progress=_log_progress, _nested_parallel=_nested_parallel)

    role_records = {
        node: {"node_role": "extant" if digraph.out_degree(node) == 0 else "ancestral"}
        for node in digraph.nodes()
    }
    register_auto_annotation(
        digraph,
        "node_role",
        role_records,
        metadata={"description": "Phylogenetic node roles (ancestral vs extant)."},
    )
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
                                             t: Optional[Union[int, float]] = 5,
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
    
    t : int | float | None, default=5
        Diffusion power for the Markov transition matrix. When ``None``,
        ``0`` or ``np.inf`` the stationary distribution is used instead of
        an explicit power.
    
    tau : float, default=1.0
        The temperature parameter used to smooth the distance kernel.
    
    cpus : int, default=1
        Target number of worker CPUs for Ray alignment tasks. Each task
        consumes a single CPU.

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
    
    logger = logging.getLogger('fitness_landscape')
    try:
        num_cpus = int(cpus)
    except (TypeError, ValueError):
        raise ValueError("`cpus` must be an integer >= 1.") from None
    if num_cpus < 1:
        raise ValueError("`cpus` must be an integer >= 1.")

    # Init ray
    _ensure_ray_initialized(num_cpus, logger=logger)
    
    # Compute in parallel.
    refs = [_score_pair.options(num_cpus=1).remote(i, j, sequences[i], sequences[j], tau, replacement_matrix)
            for (i, j) in pairs_to_align]
    total_tasks = len(refs)
    logger.info('Submitted alignment tasks: %d', total_tasks)
    if total_tasks:
        pending = list(refs)
        completed = 0
        log_every = max(1, total_tasks // 20)
        while pending:
            num_returns = min(32, len(pending))
            ready, pending = ray.wait(pending, num_returns=num_returns)
            results = ray.get(ready)
            for i, j, kv in results:
                kernel_matrix[i, j] = kv
            completed += len(results)
            if completed == total_tasks or completed % log_every == 0:
                logger.info('Alignments progress: %d/%d (%.1f%%)', completed, total_tasks, (completed / total_tasks) * 100.0)

    # Proceed with diffusion and graph construction (same as before)
    np.fill_diagonal(kernel_matrix, 0)
    
    row_sums = kernel_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition_matrix = kernel_matrix / row_sums
    
    use_stationary = _should_use_stationary_power(t)

    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n_sequences))

    threshold = 1e-4 if connectivity_threshold is None else float(connectivity_threshold)

    if use_stationary:
        stationary = _compute_stationary_distribution(transition_matrix).astype(np.float64, copy=False)
        active_targets = np.where(stationary > threshold)[0]
        for src in range(n_sequences):
            for dst in active_targets:
                if src != dst:
                    digraph.add_edge(src, dst, kernel_weight=float(stationary[dst]))
    else:
        if isinstance(t, (np.floating, float)) and not float(t).is_integer():
            raise ValueError("`t` must be an integer when diffusion power is finite.")
        t_int = int(t) if t is not None else 0
        if t_int < 1:
            raise ValueError("`t` must be >= 1 when diffusion power is finite.")

        diffused_matrix = np.linalg.matrix_power(transition_matrix, t_int)

        rows, cols = np.where(diffused_matrix > threshold)

        for i, j in zip(rows, cols):
            if i != j:
                digraph.add_edge(i, j, kernel_weight=float(diffused_matrix[i, j]))
        
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
    embedder = SoftSamplerESMEmbedder(model_name=kwargs.get('model_name', "facebook/esm2_t6_8M_UR50D"))
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

def create_plm_interpolation_digraph(
    sequences: List[BaseNumpySequence],
    *,
    k: int = 12,
    backend: Literal["auto", "faiss", "balltree"] = "auto",
    index_type: Literal["hnsw", "flat", "ivf"] = "hnsw",
    faiss_metric: Literal["ip", "l2"] = "ip",
    include_self: bool = False,
    use_gpu: bool = False,
    hnsw_M: int = 32,
    tiebuffer: int = 128,
    tie_policy: Literal["all", "min_index", "random"] = "all",
    seed: int | None = None,
    gradient_threshold: float = 0.0,
    alpha_schedule: Sequence[float] = (0.25, 0.5, 0.75),
    beam_width: int = 6,
    distance_penalty: float = 0.5,
    max_beam_rounds: int = 20,
    max_children_per_parent: int | None = None,
    max_candidates_per_round: int | None = None,
    min_pll_gain: float = 1e-3,
    embedding_mode: Literal["auto", "hard", "soft"] = "auto",
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 16,
    device: str | None = None,
    embeddings: np.ndarray | None = None,
    _compute_hamming_edges: bool = True,
    **_,
) -> nx.DiGraph:
    """
    Construct a directed kNN graph whose edges follow the pseudo
    log-likelihood gradient implied by an ESM model, with optional
    Steiner nodes discovered via beam-search interpolation.
    """

    if not sequences:
        return nx.DiGraph()

    knn_graph = create_knn_graph(
        sequences,
        k,
        backend=backend,
        index_type=index_type,
        faiss_metric=faiss_metric,
        include_self=include_self,
        use_gpu=use_gpu,
        hnsw_M=hnsw_M,
        tiebuffer=tiebuffer,
        tie_policy=tie_policy,
        seed=seed,
        _compute_hamming_edges=False,
    )

    embedder, resolved_mode = resolve_plm_embedder(
        sequences,
        embedding_mode=embedding_mode,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    scorer = PseudoLogLikelihoodScorer(embedder, batch_size=batch_size)

    node_strings: Dict[Hashable, str] = {}
    for node, data in knn_graph.nodes(data=True):
        node_strings[node] = sequence_to_text(data["sequence"])

    pll_values = scorer.score([node_strings[n] for n in knn_graph.nodes()])
    pll_by_node = {node: pll for node, pll in zip(knn_graph.nodes(), pll_values)}

    directed = nx.DiGraph()
    for node, data in knn_graph.nodes(data=True):
        attrs = dict(data)
        attrs["pll"] = pll_by_node[node]
        attrs["embedding_mode"] = resolved_mode
        directed.add_node(node, **attrs)

    beam = InterpolationBeamSearch(
        scorer,
        beam_width=beam_width,
        distance_penalty=distance_penalty,
        max_rounds=max_beam_rounds,
        max_children_per_parent=max_children_per_parent,
        max_candidates_per_round=max_candidates_per_round,
        min_pll_gain=min_pll_gain,
    )
    steiner_counter = count()

    for u, v, edge_data in knn_graph.edges(data=True):
        seq_u = node_strings[u]
        seq_v = node_strings[v]
        pll_u = pll_by_node[u]
        pll_v = pll_by_node[v]
        base_attrs = dict(edge_data)

        if pll_v - pll_u >= gradient_threshold:
            _attach_directed_path(
                directed,
                beam,
                steiner_counter,
                source_node=u,
                target_node=v,
                source_seq=seq_u,
                target_seq=seq_v,
                source_pll=pll_u,
                target_pll=pll_v,
                alpha_schedule=alpha_schedule,
                base_edge_attrs=base_attrs,
            )

        if pll_u - pll_v >= gradient_threshold:
            _attach_directed_path(
                directed,
                beam,
                steiner_counter,
                source_node=v,
                target_node=u,
                source_seq=seq_v,
                target_seq=seq_u,
                source_pll=pll_v,
                target_pll=pll_u,
                alpha_schedule=alpha_schedule,
                base_edge_attrs=base_attrs,
            )

    if _compute_hamming_edges:
        try:
            compute_edge_mutations_star(G=directed)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to attach mutation annotations to PLM interpolation graph: %s", exc
            )

    role_records = {
        node: {"node_role": "steiner" if directed.nodes[node].get("steiner") else "terminal"}
        for node in directed.nodes()
    }
    register_auto_annotation(
        directed,
        "node_role",
        role_records,
        metadata={"description": "PLM interpolation node roles (terminal vs steiner)."},
    )

    return directed


def _attach_directed_path(
    graph: nx.DiGraph,
    beam: InterpolationBeamSearch,
    steiner_counter: Iterator[int],
    *,
    source_node,
    target_node,
    source_seq: str,
    target_seq: str,
    source_pll: float,
    target_pll: float,
    alpha_schedule: Sequence[float],
    base_edge_attrs: Dict,
) -> None:
    diff_positions = [i for i, (a, b) in enumerate(zip(source_seq, target_seq)) if a != b]
    if not diff_positions:
        return

    target_counts = []
    total_diffs = len(diff_positions)
    for alpha in alpha_schedule:
        count = int(np.round(alpha * total_diffs))
        count = max(1, min(total_diffs - 1, count))
        if count not in target_counts:
            target_counts.append(count)
    if not target_counts:
        target_counts = [total_diffs - 1]

    intermediates = beam.interpolate(
        source_seq,
        target_seq,
        target_counts=target_counts,
        diff_positions=diff_positions,
        start_pll=source_pll,
    )

    previous_node = source_node
    previous_seq = source_seq
    previous_pll = source_pll

    for state in intermediates:
        node_id = _next_steiner_node_id(graph, steiner_counter)
        seq_obj = string_to_sequence(state.sequence, sequence_id=node_id)
        graph.add_node(
            node_id,
            sequence=seq_obj,
            pll=state.pll,
            steiner=True,
            alpha=float(state.matches) / float(total_diffs),
        )
        _add_directed_edge(
            graph,
            previous_node,
            node_id,
            previous_seq,
            state.sequence,
            previous_pll,
            state.pll,
            base_edge_attrs,
        )
        previous_node = node_id
        previous_seq = state.sequence
        previous_pll = state.pll

    _add_directed_edge(
        graph,
        previous_node,
        target_node,
        previous_seq,
        target_seq,
        previous_pll,
        target_pll,
        base_edge_attrs,
    )


def _next_steiner_node_id(graph: nx.DiGraph, counter: Iterator[int]):
    while True:
        candidate = f"steiner_{next(counter)}"
        if candidate not in graph:
            return candidate


def _add_directed_edge(
    graph: nx.DiGraph,
    source,
    target,
    source_seq: str,
    target_seq: str,
    source_pll: float,
    target_pll: float,
    base_edge_attrs: Dict,
):
    attrs = dict(base_edge_attrs)
    distance = hamming_distance_str(source_seq, target_seq)
    attrs["distance"] = distance
    attrs["weight"] = distance
    attrs["knn_weight"] = distance
    delta = float(target_pll - source_pll)
    attrs["delta_pll"] = delta
    attrs["pll_delta"] = delta
    graph.add_edge(source, target, **attrs)
