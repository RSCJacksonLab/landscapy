import os

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationError,
    ConfigDict
)
from typing import (
    Union,
    List,
    Literal,
    Iterable,
    Dict,
    Any
)
import numpy as np
from ..core.landscape import (
    FitnessLandscape,
    DirectedFitnessLandscape
)
from ..core.graph import compute_edge_mutations_star
from ..core.sequence import (
    BaseNumpySequence,
    SoftSequence
)
from ..core.fitness import (
    NumericFitness,
    CategoricalFitness,
    ProbabilisticCategoricalFitness,
    BaseFitnessLayer
)
from ..graph_matching.latent_alignment import RJMCMCAligner
from ..graph_matching.hierarchical_alignment import HierarchicalRJMCMCAligner
import networkx as nx
from softalign.soft_alignment import align_soft_sequences
import ray
from pathlib import Path
from cogent3.core.alignment import Alignment
from ..utils import (
    PROT_20,
    alignment_to_base_numpy_sequences
)
import torch
import pickle


class NullAligner:
    """
    Lightweight, top-level placeholder aligner used when a Superscape
    is built from a single landscape and no hierarchical alignment is
    required. Exists at module scope so instances are picklable.
    """
    def __init__(self, *, directed: bool = False):
        self.full_posterior_L = []
        self.full_posterior_mappings = []
        self.directed = directed


class EmbNodeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    emb_arr: np.ndarray = Field(..., repr=False)

    @field_validator("emb_arr")
    @classmethod
    def _check_emb(cls, v):
        v = np.asarray(v)
        if v.ndim != 1:
            raise ValueError("emb_arr must be a 1-D array")
        return v

# Parallel landscape constructor private function.
# Parallel steps are spawned within the outer superscape loop.
@ray.remote(num_cpus=1)
def _create_landscape_task(
    constructor_class: Union[FitnessLandscape, DirectedFitnessLandscape],
    sequences: Union[Path, Alignment, List[BaseNumpySequence]],
    fitness_layers: Dict[str, BaseFitnessLayer] = None,
    _job_id: int | None = None,
    _total_jobs: int | None = None,
    _log_progress: bool = False,
    **kwargs: Any
) -> Union[FitnessLandscape, DirectedFitnessLandscape]:
    """
    A generalized Ray remote task that calls the `from_sequences` method
    of a specified landscape class.
    """
    import logging as _logging, time as _time, os as _os
    # Constrain intra-op threading in worker to avoid oversubscription/OOM
    _os.environ.setdefault('OMP_NUM_THREADS', '1')
    _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    _os.environ.setdefault('MKL_NUM_THREADS', '1')
    _os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    _logger = _logging.getLogger('fitness_landscape')
    if _log_progress:
        _logger.info('[job %s/%s] start', _job_id, _total_jobs)
    ts = _time.perf_counter()
    try:
        result = constructor_class.from_sequences(
            sequences=sequences,
            fitness_layers=fitness_layers,
            **kwargs
        )
        if _log_progress:
            _logger.info('[job %s/%s] complete in %.2fs', _job_id, _total_jobs, _time.perf_counter()-ts)
        return result
    except Exception as e:
        msg = f"[job {_job_id}/{_total_jobs}] construction failed: {e}"
        raise RuntimeError(msg) from e

class FitnessSuperscape:
    """
    FitnessSuperscape is a class that manages multiple fitness
    landscapes and aligns them into a common latent space using RJMCMC
    sampling.

    Attributes
    ----------
    landscapes : List[Union[FitnessLandscape, DirectedFitnessLandscape]]
        A list of fitness landscapes or graph-like objects to be aligned.
    posterior_prob_cutoff : float
        The cutoff for posterior probabilities when constructing the
        latent landscape.
    """

    def __init__(self,
                 
                 landscapes: List[Union[FitnessLandscape, DirectedFitnessLandscape]],
                 posterior_prob_cutoff: float = 0.1,
                 **sampler_kwargs) -> None:
        
        self.landscapes = landscapes
        self._landscape_graphs = self._extract_graphs(landscapes=self.landscapes)

        # Ensure all graphs have per-node embeddings. If missing, compute a
        # compact, length-invariant composition embedding from sequences.
        self._ensure_node_embeddings(self._landscape_graphs)

        # Validate data types in the graph.
        self._validate_embeddings(self._landscape_graphs)
        # Validate and set the common alphabet across all landscapes.
        self.alphabet = self._validate_and_set_alphabet(self.landscapes)

        # Fast-path: if there is only one landscape, skip hierarchical alignment
        # and treat the single graph as the latent graph with identity mapping.
        if len(self._landscape_graphs) == 1:
            G0 = self._landscape_graphs[0]
            # identity mapping from original nodes to latent nodes
            n0 = G0.number_of_nodes()
            import numpy as _np
            self.latent_graph = G0.copy()
            self._latent_mappings = {0: _np.eye(n0, dtype=float)}

            # Provide a minimal, picklable aligner-like object
            import networkx as _nx
            self._hierarchical_aligner = NullAligner(directed=isinstance(G0, _nx.DiGraph))
            # empty traces (alias both naming styles for consistency with multi-graph path)
            self.local_energy_traces = {}
            self.local_nl_traces = {}
            self.local_edges_traces = {}
            self.meta_energy_trace = []
            self.meta_nl_trace = []
            self.meta_edges_trace = []
            # public aliases used elsewhere when multiple landscapes are aligned
            self.local_trace_E = self.local_energy_traces
            self.local_trace_NL = self.local_nl_traces
            self.local_trace_edges = self.local_edges_traces
            self.meta_trace_E = self.meta_energy_trace
            self.meta_trace_NL = self.meta_nl_trace
            self.meta_trace_edges = self.meta_edges_trace

            # Canonical node order and back refs
            self._node_orders = [list(self.landscapes[0].graph.nodes())]
            self.back_reference = [(0, node_id) for node_id in self._node_orders[0]]

            # Construct latent landscape object and return
            self.latent_landscape = self.construct_latent_landscape(posterior_prob_cutoff=posterior_prob_cutoff)
            return

        # Run RJMCMC sampling using the hierarchical aligner (scales ~linearly).
        # Prepare hierarchical aligner controls (top-level kwargs) and pass the rest
        # through as RJMCMCAligner params via aligner_params.
        _sampler_kwargs = dict(sampler_kwargs)

        # Extract hierarchical kwargs (support both underscored and non-underscored forms)
        _hier_kwargs = {}
        def _pop_any(d, keys, default=None):
            for k in keys:
                if k in d:
                    return d.pop(k)
            return default

        # Sliding-window controls: default-enable sliding windows unless explicitly disabled
        _lw_shifts = _pop_any(_sampler_kwargs, [
            'local_window_shifts'
        ], None)
        _lw_size = _pop_any(_sampler_kwargs, [
            'local_window_size'
        ], None)
        _lw_stride = _pop_any(_sampler_kwargs, [
            'local_window_stride'
        ], None)
        # If user did not specify any windowing params, default to sliding windows enabled.
        if _lw_shifts is None and _lw_size is None and _lw_stride is None:
            _hier_kwargs['local_window_shifts'] = 1
        else:
            if _lw_shifts is not None:
                _hier_kwargs['local_window_shifts'] = _lw_shifts
            if _lw_size is not None:
                _hier_kwargs['local_window_size'] = _lw_size
            if _lw_stride is not None:
                _hier_kwargs['local_window_stride'] = _lw_stride

        # CPU chain controls
        _lc = _pop_any(_sampler_kwargs, ['_local_cpu_chains', 'local_cpu_chains'], None)
        _mc = _pop_any(_sampler_kwargs, ['_meta_cpu_chains', 'meta_cpu_chains'], None)
        if _lc is not None:
            _hier_kwargs['_local_cpu_chains'] = _lc
        if _mc is not None:
            _hier_kwargs['_meta_cpu_chains'] = _mc

        # Progress / checkpoint controls
        _show = _pop_any(_sampler_kwargs, ['_show_progress', 'show_progress'], None)
        _ckpt_dir = _pop_any(_sampler_kwargs, ['_checkpoint_dir', 'checkpoint_dir'], None)
        _ckpt_int = _pop_any(_sampler_kwargs, ['_checkpoint_interval', 'checkpoint_interval'], None)
        _ckpt_resume = _pop_any(_sampler_kwargs, ['_resume_checkpoint', 'resume_checkpoint'], None)
        if _show is not None:
            _hier_kwargs['_show_progress'] = _show
        if _ckpt_dir is not None:
            _hier_kwargs['_checkpoint_dir'] = _ckpt_dir
        if _ckpt_int is not None:
            _hier_kwargs['_checkpoint_interval'] = _ckpt_int
        if _ckpt_resume is not None:
            _hier_kwargs['_resume_checkpoint'] = _ckpt_resume

        self._hierarchical_aligner = HierarchicalRJMCMCAligner(
            graphs=self._landscape_graphs,
            aligner_params=_sampler_kwargs,
            **_hier_kwargs,
        )
        # The results are now stored directly, not the aligner object
        self.latent_graph, self._latent_mappings, = self._hierarchical_aligner.run_alignment()
        
        # Collect local traces (expose with consistent names)
        self.local_trace_E = self._hierarchical_aligner.local_energy_traces
        self.local_trace_NL = self._hierarchical_aligner.local_nl_traces
        self.local_trace_edges = self._hierarchical_aligner.local_edges_traces
        # Backward-friendly aliases
        self.local_energy_traces = self.local_trace_E
        self.local_nl_traces = self.local_trace_NL
        self.local_edges_traces = self.local_trace_edges
        
        # Collect meta traces (empty when using overlap-based meta stage)
        self.meta_trace_E = self._hierarchical_aligner.meta_energy_trace
        self.meta_trace_NL = self._hierarchical_aligner.meta_nl_trace
        self.meta_trace_edges = self._hierarchical_aligner.meta_edges_trace
        
        # Canonical node order.
        self._node_orders = [list(L.graph.nodes()) for L in self.landscapes]
        self.back_reference = [
            (k, node_id)
            for k, order in enumerate(self._node_orders)
            for node_id in order
        ]
        
        # Capture self attribute latent landscape for safety. Can be recomputed with diff posterior prob cutoffs.
        self.latent_landscape = self.construct_latent_landscape(posterior_prob_cutoff=posterior_prob_cutoff)

    @staticmethod
    def _ensure_node_embeddings(graphs: list[Union[nx.Graph, nx.DiGraph]], *, alphabet: List[str] = PROT_20) -> None:
        """
        Attach fallback node embeddings if 'emb_arr' is missing.

        Uses a simple composition vector over the provided alphabet (default PROT_20),
        computed as the per-position mean of a (L, A) soft/hard representation.
        This is length-invariant and robust for clustering.
        """
        A = len(alphabet)
        alpha_index = {str(a).upper(): i for i, a in enumerate(alphabet)}

        def comp_from_ungapped(arr: np.ndarray) -> np.ndarray:
            x = np.asarray(arr, dtype=np.float64)
            # Accept (L, A) or (L, A+1); drop gap if present
            if x.ndim != 2:
                return None
            C = x.shape[1]
            if C == A + 1:
                x = x[:, :A]
            if x.shape[1] != A:
                return None
            row_sum = x.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0.0] = 1.0
            x = x / row_sum
            v = x.mean(axis=0)
            s = float(v.sum())
            return (v / s) if s > 0 else np.full(A, 1.0 / A)

        def comp_from_sequence(seq) -> np.ndarray:
            # SoftSequence: use ungapped posterior
            if hasattr(seq, 'ungapped_arr'):
                v = comp_from_ungapped(seq.ungapped_arr)
                if v is not None:
                    return v
            # BaseNumpySequence: build frequency vector over alphabet
            arr = getattr(seq, 'to_array', lambda: None)()
            if arr is None:
                return np.full(A, 1.0 / A)
            counts = np.zeros(A, dtype=np.float64)
            total = 0
            for s in arr:
                j = alpha_index.get(str(s).upper(), None)
                if j is not None:
                    counts[j] += 1.0
                    total += 1
            if total <= 0:
                return np.full(A, 1.0 / A)
            return counts / total

        for G in graphs:
            for _, data in G.nodes(data=True):
                if 'emb_arr' in data and isinstance(data['emb_arr'], np.ndarray) and data['emb_arr'].ndim == 1:
                    continue
                # Try ungapped arrays first
                if 'ungapped_arr' in data:
                    emb = comp_from_ungapped(data['ungapped_arr'])
                    if emb is not None:
                        data['emb_arr'] = emb
                        continue
                # Fallback to sequence-based composition
                if 'sequence' in data:
                    data['emb_arr'] = comp_from_sequence(data['sequence'])

    def construct_latent_landscape(self,
                                   posterior_prob_cutoff: float = 0.2) -> Union[FitnessLandscape, DirectedFitnessLandscape]:
        """
        Constructs a latent landscape from the posterior samples using a
        specified probability cutoff for edge existence.

        This method can be called multiple times with different cutoffs to
        explore the latent graph at different confidence levels.

        Parameters
        ----------
        posterior_prob_cutoff : float, default=0.2
            The posterior probability threshold for an edge to be included
            in the latent graph.

        Returns
        -------
        Union[FitnessLandscape, DirectedFitnessLandscape]
            The constructed latent fitness landscape.
        """
        if not self._hierarchical_aligner.full_posterior_L:
            # Fallback: use the deterministic result returned by run_alignment()
            # This occurs when the meta-stage produced no edges / no posterior sampling.
            # Build the latent landscape directly from the stored latent_graph and mappings.
            graph = self.latent_graph
            mappings = self._latent_mappings
            return self._build_landscape_from_graph_and_mappings(graph, mappings)

        posterior_L = self._hierarchical_aligner.full_posterior_L
        posterior_mappings = self._hierarchical_aligner.full_posterior_mappings
        

        # Average the adjacency matrices from all posterior samples
        max_nl = max(l.shape[0] for l in posterior_L) if posterior_L else 0
        tally_matrix = np.zeros((max_nl, max_nl))
        for l_matrix in posterior_L:
            current_nl = l_matrix.shape[0]
            tally_matrix[:current_nl, :current_nl] += l_matrix
        
        L_avg = tally_matrix / len(posterior_L)
        
        # Apply the threshold to get the final adjacency matrix
        L_final = (L_avg >= posterior_prob_cutoff).astype(int)
        
        # Create the graph from the thresholded matrix
        GraphClass = nx.DiGraph if self._hierarchical_aligner.directed else nx.Graph
        graph = nx.from_numpy_array(L_final, create_using=GraphClass)

        # Build the landscape from this graph and the posterior mean mappings
        return self._build_landscape_from_graph_and_mappings(graph, self._latent_mappings)

    def _build_landscape_from_graph_and_mappings(self, 
                                                 graph: Union[nx.Graph, nx.DiGraph], 
                                                 mappings: Dict[int, np.ndarray]) -> Union[FitnessLandscape, DirectedFitnessLandscape]:
        """
        A helper method to construct a FitnessLandscape object from a given
        graph and a set of node-to-latent-space mappings.
        """
        num_total_nodes = sum(len(g.nodes()) for g in self._landscape_graphs)
        num_latent_nodes = graph.number_of_nodes()
        all_prob_maps = np.zeros((num_total_nodes, num_latent_nodes))
        
        current_row = 0
        for k in sorted(mappings.keys()):
            mapping_matrix = mappings[k]
            num_nodes_in_graph = mapping_matrix.shape[0]
            if num_nodes_in_graph > 0 and mapping_matrix.shape[1] == num_latent_nodes:
                 all_prob_maps[current_row : current_row + num_nodes_in_graph, :] = mapping_matrix
            current_row += num_nodes_in_graph

        all_ungapped_arrs = []
        for k, L in enumerate(self.landscapes):
            order = self._node_orders[k]
            for node_id in order:
                data = L.graph.nodes[node_id]
                # Prefer precomputed 'ungapped_arr' but derive on-the-fly from sequence if missing
                arr = data.get('ungapped_arr', None)
                if arr is None:
                    seq = data.get('sequence', None)
                    if seq is None:
                        raise ValueError(f"Node {node_id!r} missing 'sequence' for superscape.")
                    # SoftSequence and BaseNumpySequence expose a robust ungapped_arr
                    if hasattr(seq, 'ungapped_arr'):
                        arr = seq.ungapped_arr
                    else:
                        # As a last resort, convert any available gapped_arr by dropping gap and renormalising
                        gapped = data.get('gapped_arr', None)
                        if gapped is not None:
                            import numpy as _np
                            g = _np.asarray(gapped, dtype=_np.float64)
                            if g.ndim != 2 or g.shape[1] < 2:
                                raise ValueError(f"Node {node_id!r} has invalid gapped_arr shape {g.shape}")
                            aa = g[:, :-1]
                            rs = aa.sum(axis=1, keepdims=True)
                            rs[rs <= 0.0] = 1.0
                            arr = aa / rs
                        else:
                            # Fall back to one-hot
                            if hasattr(seq, 'to_one_hot'):
                                arr = seq.to_one_hot()
                            else:
                                raise ValueError(f"Node {node_id!r} lacks ungapped/gapped arrays and to_one_hot")
                all_ungapped_arrs.append(arr)

        all_lengths = [len(seq) for landscape in self.landscapes for seq in landscape.sequences]
        default_length = int(np.median(all_lengths)) if all_lengths else 1

        latent_sequences = []
        for latent_node_idx in range(num_latent_nodes):
            prob_col = all_prob_maps[:, latent_node_idx]
            contributor_indices = np.where(prob_col > 0)[0]
            observed_mappings = []
            for flat_idx in contributor_indices:
                graph_idx, node_id = self.back_reference[flat_idx]
                probability = prob_col[flat_idx]
                observed_mappings.append({"node_id": node_id, "probability": probability, "graph_index": graph_idx})

            observed_mappings.sort(key=lambda x: x['probability'], reverse=True)
            graph.nodes[latent_node_idx]['observed_node_mappings'] = observed_mappings

            if len(contributor_indices) == 0:
                uniform_probability = 1.0 / len(self.alphabet)
                uniform_posterior = np.full((default_length, len(self.alphabet)), uniform_probability)
                ambiguous_sequence = SoftSequence(uniform_posterior, alphabet=self.alphabet)
                latent_sequences.append(ambiguous_sequence)
                gapped_arr = np.zeros((default_length, len(self.alphabet) + 1))
                gapped_arr[:, :-1] = uniform_posterior
                graph.nodes[latent_node_idx]['gapped_arr'] = gapped_arr
                graph.nodes[latent_node_idx]['ungapped_arr'] = uniform_posterior
                continue

            ungapped_arrs_to_align = [np.ascontiguousarray(np.asarray(all_ungapped_arrs[i], dtype=np.float64)) for i in contributor_indices]
            contributor_probs = prob_col[contributor_indices]
            _res = align_soft_sequences(sequences=ungapped_arrs_to_align, alphabet=self.alphabet)
            aligned_arrays = _res[0] if isinstance(_res, tuple) else _res
            aligned_tensor = np.array(aligned_arrays)
            total_prob_for_node = np.sum(contributor_probs) + 1e-12
            weighted_sum_posterior = np.einsum('i,ija->ja', contributor_probs, aligned_tensor)
            final_posterior = weighted_sum_posterior / total_prob_for_node
            graph.nodes[latent_node_idx]['gapped_arr'] = final_posterior
            ungapped_arr = final_posterior[:, :-1]
            ungapped_arr = ungapped_arr / ungapped_arr.sum(axis=1, keepdims=True)
            graph.nodes[latent_node_idx]['ungapped_arr'] = ungapped_arr
            aa_posterior = final_posterior[:, :-1]
            gap_posterior = final_posterior[:, -1:]
            latent_sequences.append(SoftSequence(aa_posterior=aa_posterior, alphabet=self.alphabet, gap_posterior=gap_posterior))

        latent_fitness_layers = {}
        all_layer_names = set(name for l in self.landscapes for name in l.fitness_layers)

        for name in all_layer_names:
            first_layer = next(l.fitness_layers[name] for l in self.landscapes if name in l.fitness_layers)

            if first_layer.dtype == 'numeric':
                all_means = np.concatenate([l.view(name).to_scalar() for l in self.landscapes])
                total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
                weighted_sum = all_prob_maps.T @ all_means
                latent_means = (weighted_sum / total_prob_per_latent).tolist()
                latent_fitness_layers[name] = NumericFitness(name, [[m] for m in latent_means])

            elif first_layer.dtype == 'categorical':
                categories = first_layer.categories
                all_one_hot = np.concatenate([l.view(name).get_tensor().numpy() for l in self.landscapes])
                total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
                weighted_sum_of_one_hots = all_prob_maps.T @ all_one_hot
                latent_probabilities = weighted_sum_of_one_hots / total_prob_per_latent[:, np.newaxis]
                latent_fitness_layers[name] = ProbabilisticCategoricalFitness(name, latent_probabilities, categories)
        
        for i, seq in enumerate(latent_sequences):
            if i in graph.nodes:
                graph.nodes[i]['sequence'] = seq

        LandscapeClass = DirectedFitnessLandscape if isinstance(graph, nx.DiGraph) else FitnessLandscape
        landscape = LandscapeClass(sequences=latent_sequences, fitness_layers=latent_fitness_layers, graph=graph)
        
        if all(isinstance(seq, SoftSequence) and hasattr(seq, "ungapped_arr") and seq.alphabet==PROT_20 for _, seq in graph.nodes(data='sequence')):
            compute_edge_mutations_star(G=landscape.graph)
        
        return landscape
    def sample_latent_landscapes(self,
                                    n_samples: int,
                                    seed: int = None) -> List[FitnessLandscape]:
            """
            Samples from the posterior distribution of the latent landscape to create an
            ensemble of plausible landscapes.

            Parameters
            ----------
            n_samples : int
                The number of latent landscapes to sample.

            seed : int, optional
                Random seed for reproducibility. If `None`, a random seed is used.

            Returns
            -------
            List[FitnessLandscape]
                A list of FitnessLandscape objects, each representing a sample from the posterior.
            """
            if not hasattr(self, '_hierarchical_aligner') or not self._hierarchical_aligner.full_posterior_L:
                raise RuntimeError("The hierarchical alignment has not been run or did not store posterior samples. "
                                "Run `run_alignment()` on the HierarchicalRJMCMCAligner first.")

            posterior_L = self._hierarchical_aligner.full_posterior_L
            posterior_mappings = self._hierarchical_aligner.full_posterior_mappings
            num_available_samples = len(posterior_L)

            sampled_landscapes = []
            for _ in range(n_samples):

                # Randomly select a pre-computed sample index
                sample_idx = np.random.randint(num_available_samples)
                
                L_sample_matrix = posterior_L[sample_idx]
                mappings_sample = posterior_mappings[sample_idx]

                # Construct the graph for this sample
                graph_sample = nx.from_numpy_array(
                    L_sample_matrix,
                    create_using=nx.DiGraph if self._hierarchical_aligner.directed else nx.Graph
                )

                # Construct a full FitnessLandscape object from this sample
                landscape_sample = self._construct_landscape_from_sample(graph_sample, mappings_sample)
                sampled_landscapes.append(landscape_sample)

            return sampled_landscapes

    def _construct_landscape_from_sample(self, 
                                         graph_sample: Union[nx.Graph, nx.DiGraph], 
                                         mappings_sample: Dict[int, np.ndarray]) -> FitnessLandscape:
        """
        Helper function to build a FitnessLandscape object from a single posterior sample.
        This logic mirrors the `construct_latent_landscape` method but operates on a
        single probabilistic mapping instead of the posterior mean.
        """
        num_latent_nodes = graph_sample.number_of_nodes()
        all_prob_maps = np.vstack([mappings_sample[k] for k in sorted(mappings_sample.keys())])      
        
        all_ungapped_arrs = []
        for k, L in enumerate(self.landscapes):
            order = self._node_orders[k]
            for node_id in order:
                data = L.graph.nodes[node_id]
                all_ungapped_arrs.append(data['ungapped_arr'])

        all_lengths = [len(seq) for landscape in self.landscapes for seq in landscape.sequences]
        default_length = int(np.median(all_lengths)) if all_lengths else 1

        latent_sequences = []
        for latent_node_idx in range(num_latent_nodes):
            prob_col = all_prob_maps[:, latent_node_idx]
            contributor_indices = np.where(prob_col > 0)[0]
            
            if len(contributor_indices) == 0:

                # Handle cases where a latent node has no contributors in a sample
                uniform_probability = 1.0 / len(self.alphabet)
                uniform_posterior = np.full((default_length, len(self.alphabet)), uniform_probability)
                ambiguous_sequence = SoftSequence(uniform_posterior, alphabet=self.alphabet)
                latent_sequences.append(ambiguous_sequence)
                continue

            ungapped_arrs_to_align = [all_ungapped_arrs[i] for i in contributor_indices]
            contributor_probs = prob_col[contributor_indices]

            _res = align_soft_sequences(sequences=[np.ascontiguousarray(np.asarray(a, dtype=np.float64)) for a in ungapped_arrs_to_align], alphabet=self.alphabet)
            aligned_arrays = _res[0] if isinstance(_res, tuple) else _res
            aligned_tensor = np.array(aligned_arrays)
            
            total_prob_for_node = np.sum(contributor_probs) + 1e-12
            weighted_sum_posterior = np.einsum('i,ija->ja', contributor_probs, aligned_tensor)
            final_posterior = weighted_sum_posterior / total_prob_for_node
            
            aa_posterior = final_posterior[:, :-1]
            gap_posterior = final_posterior[:, -1:]

            latent_sequences.append(
                SoftSequence(
                    aa_posterior=aa_posterior,
                    alphabet=self.alphabet,
                    gap_posterior=gap_posterior
                )
            )
            
        latent_fitness_layers = {}
        all_layer_names = set(name for l in self.landscapes for name in l.fitness_layers)

        for name in all_layer_names:
            first_layer = next(l.fitness_layers[name] for l in self.landscapes if name in l.fitness_layers)
            
            if first_layer.dtype == 'numeric':

                all_means = np.concatenate([l.view(name).to_scalar() for l in self.landscapes])
                total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
                weighted_sum = all_prob_maps.T @ all_means
                latent_means = (weighted_sum / total_prob_per_latent).tolist()
                latent_fitness_layers[name] = NumericFitness(name, [[m] for m in latent_means])
            
            elif first_layer.dtype == 'categorical':

                categories = first_layer.categories
                all_one_hot = np.concatenate([l.view(name).get_tensor().numpy() for l in self.landscapes])
                total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
                weighted_sum_of_one_hots = all_prob_maps.T @ all_one_hot
                latent_probabilities = weighted_sum_of_one_hots / total_prob_per_latent[:, np.newaxis]
                latent_fitness_layers[name] = ProbabilisticCategoricalFitness(name, latent_probabilities, categories)

        for i, seq in enumerate(latent_sequences):
            if i in graph_sample.nodes:
                graph_sample.nodes[i]['sequence'] = seq

        LandscapeClass = DirectedFitnessLandscape if isinstance(graph_sample, nx.DiGraph) else FitnessLandscape
        return LandscapeClass(
            sequences=latent_sequences,
            fitness_layers=latent_fitness_layers,
            graph=graph_sample)

    @staticmethod
    def _validate_embeddings(graphs: list[Union[nx.Graph, nx.DiGraph]]) -> None:
        """
        Helper method to validate nodes have valid emb_arr attribute.

        Parameters
        ----------
        graphs : List
            List of nx.Graph or nx.DiGraph objects to be aligned.
        """
        for G in graphs:
            for node, data in G.nodes(data=True):
                try:
                    EmbNodeModel(**data) # will raise if missing/invalid
                except ValidationError as e:

                    raise ValueError(f"{node!r}: {e}") from None
    
    @staticmethod
    def _validate_and_set_alphabet(landscapes: List[Union[FitnessLandscape, DirectedFitnessLandscape]]) -> list:
        """
        Validates that all sequences across all landscapes share a
        common alphabet and returns it.

        Parameters
        ----------
        landscapes : List[FitnessLandscape, DirectedFitnessLandscape]
            The list of fitness landscapes to validate.

        Returns
        -------
        list
            The common alphabet.

        Raises
        ------
        ValueError
            If alphabets are inconsistent or no sequences are found.
        """
        
        combined_alphabet_set = set()

        # Create a generator for all sequences
        all_sequences_gen = (
            seq
            for landscape in landscapes
            if isinstance(landscape, FitnessLandscape) and landscape.sequences
            for seq in landscape.sequences
        )

        found_sequences = False
        for seq in all_sequences_gen:
            found_sequences = True
            combined_alphabet_set.update(seq.alphabet)

        if not found_sequences:
            raise ValueError("Could not determine alphabet: no sequences found in any of the provided landscapes.")

        return sorted(list(combined_alphabet_set))
                
    @staticmethod
    def _extract_graphs(landscapes: Iterable[Union[FitnessLandscape,
                                                   DirectedFitnessLandscape]]) -> list[Union[nx.Graph, nx.DiGraph]]:
        """
        Helper method to extract directed graphs from directed fitness
        landscapes.

        Parameters
        ----------
        landscapes : Iterable
            The list of FitenessLandscape or DirectedFitnessLandscapes.

        Returns
        -------
        out : list
            The list of nx.Graph or nx.DiGraph objects indexed matched
            to the landscapes.
        """
        return [obj.graph for obj in landscapes]

    # Delegate tensor methods to latent graph FitnessLandscape class.
    def to_graph_tensor(self) -> 'Data':
        """
        Exports the entire fitness landscape to a PyTorch Geometric
        Data object.

        This method converts the landscape's graph structure, node
        features (from embeddings or sequences), and all associated
        fitness layers into a format suitable for graph machine
        learning with PyTorch Geometric.

        Returns
        -------
        torch_geometric.data.Data
            A PyG Data object with the following attributes:
            - x: Node features (embeddings or one-hot encoded
            sequences).
            - edge_index: Graph connectivity in COO format.
            - edge_attr: Edge weights, if they exist.
            - Additional attributes corresponding to each fitness
            layer, named after the layer.
        """
        if not hasattr(self, 'latent_landscape'):
            raise RuntimeError("The latent landscape has not been constructed yet. "
                             "Run `construct_latent_landscape()` first.")
        
        return self.latent_landscape.to_graph_tensor()

    def to_sequence_tensors(self,
                            *,
                            sequence_idx: Union[List[int], int] = None,
                            sequence: Union[List[str], str] = None) -> List[Dict[str, Any]]:
        """
        Exports the sequences and their fitness layers as a list of
        dictionaries containing tensors. Supports indexing by sequence
        and by int.

        Parameters
        ----------
        sequence_idx : List or int, default=`None`
            Indices of sequences to export as tensors. If `None`, all
            sequences are exported.
        
        sequence : List of str, default=`None`
            Sequence to export as tensors. If `None`, all sequences
            are exported.

        Returns
        -------
        List[Dict[str, Any]]
            A list where each item is a dictionary representing a
            single sequence and its associated data. Each dictionary
            has the keys:
            - 'sequence_tensor': The one-hot encoded sequence or
            embedding.
            - 'fitness_tensors': A dictionary where keys are layer
            names and values are the corresponding fitness tensors
            for that sequence.
        """
        if not hasattr(self, 'latent_landscape'):
            raise RuntimeError("The latent landscape has not been constructed yet. "
                             "Run `construct_latent_landscape()` first.")
            
        return self.latent_landscape.to_sequence_tensors(
            sequence_idx=sequence_idx,
            sequence=sequence
        )
    
    def save(self, filepath: Path):
        """Saves the FitnessSuperscape object to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: Path):
        """Loads a FitnessSuperscape object from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def from_parallel_construction(cls,
                                   constructor_type: Literal['undirected', 'directed'],
                                   construction_jobs: List[Dict[str, Any]],
                                   posterior_prob_cutoff: float = 0.1,
                                   _show_progress: bool = True,
                                   _construct_checkpoint_dir: Union[str, Path, None] = None,
                                   _construct_checkpoint_interval: int = 300,
                                   _construct_resume_checkpoint: Union[str, Path, None] = None,
                                   _fresh_worker_per_job: bool = False,
                                   _parent_task_cpus: float = 1.0,
                                   _meta_cpu_chains: int | None = None,
                                   **sampler_kwargs: Any) -> "FitnessSuperscape":
        """
        A flexible factory method to create a FitnessSuperscape by
        constructing multiple landscapes of the same base type (either
        undirected or directed) in parallel using Ray.

        This method supports heterogeneous construction parameters,
        allowing construction of landscapes from different data sources
        and with different graph constructors within the same parallel
        run.

        Parameters
        ----------
        constructor_type : Literal['undirected', 'directed']
            Specifies the base type of landscapes to create for this
            entire run.
        construction_jobs : List[Dict[str, Any]]
            A list of dictionaries, each defining a single landscape to
            construct.
            
            Each dictionary must contain:
            - 'sequences': The input data (e.g., a Path, Alignment, 
            or List[BaseNumpySequence]).
            - 'graph_type' (for undirected) or 'digraph_type' (for
            directed).
            - Other keys are passed as kwargs to the constructor.
        posterior_prob_cutoff : float, default=0.1
            The cutoff for posterior probabilities in the latent
            landscape.
        **sampler_kwargs
            Keyword arguments for the RJMCMCAligner sampler.

        Returns
        -------
        FitnessSuperscape
            An instance containing the parallel-constructed landscapes.
        """
        if not ray.is_initialized():
            try:
                ray.init(object_spilling_directory="/tmp/ray_spill")
            except Exception:
                ray.init()

        landscape_class = (
            FitnessLandscape if constructor_type == 'undirected'
            else DirectedFitnessLandscape
        )

        # Optional checkpointing
        ckpt_path = None
        last_ckpt = 0.0
        if _construct_checkpoint_dir:
            ckpt_dir = Path(_construct_checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "superscape_construction.ckpt.pkl"

        # Resume support
        landscapes: List[Union[FitnessLandscape, DirectedFitnessLandscape, None]] = [None] * len(construction_jobs)
        remaining_idx = list(range(len(construction_jobs)))
        if _construct_resume_checkpoint:
            try:
                with open(_construct_resume_checkpoint, 'rb') as f:
                    state = pickle.load(f)
                if isinstance(state, dict) and 'landscapes' in state and len(state['landscapes']) == len(construction_jobs):
                    landscapes = state['landscapes']
                    remaining_idx = [i for i, x in enumerate(landscapes) if x is None]
            except Exception:
                pass

        # Prepare a submission queue; we submit at most `_meta_cpu_chains` tasks concurrently
        futures = []
        prepared_jobs = []
        for job in construction_jobs:
            if 'sequences' not in job:
                raise ValueError("Each job must have a `sequences` key.")
            elif 'graph_type' not in job and 'digraph_type' not in job:
                raise ValueError("Each job must have either `graph_type` or `digraph_type` key.")

            # Same base class to instantiate across all parallel runs.
            job['constructor_class'] = landscape_class
            
            # Only request GPUs if PLM embeddings are actually being computed.
            wants_plm = job.get("embedding_domain") == "plm"
            # detect embedding computation flags across constructors
            wants_compute = bool(job.get("_compute_phylo_embeddings", False) or job.get("_compute_embeddings", False))
            # evol_diffusion path computes embeddings internally if embedding_domain='plm'
            if not wants_compute and job.get("graph_type") == "evol_diffusion" and wants_plm:
                wants_compute = True
            num_gpus = 1 if (wants_plm and wants_compute) else 0
            # Avoid computing Hamming edge weights inside child constructors; they will be
            # recomputed on the latent graph after alignment.
            job.setdefault('_compute_hamming_edges', False)

            prepared_jobs.append((num_gpus, job))

        # Retrieve the results with progress logging
        import logging as _logging, time as _time
        _logger = _logging.getLogger('fitness_landscape')
        total = len(prepared_jobs)
        # Concurrency window size
        max_inflight = int(_meta_cpu_chains) if _meta_cpu_chains and _meta_cpu_chains > 0 else total
        # Submit initial window
        submit_order = list(range(len(prepared_jobs)))
        inflight: dict[Any, int] = {}
        done_count = 0
        t_last = _time.perf_counter()
        try:
            import psutil as _psutil  # optional
        except Exception:
            _psutil = None
        # Helper to submit next job if any remain and inflight below cap
        def _maybe_submit():
            nonlocal submit_order
            while submit_order and len(inflight) < max_inflight:
                jidx = submit_order.pop(0)
                num_gpus, job = prepared_jobs[jidx]
                _opts = {"num_gpus": num_gpus, "num_cpus": _parent_task_cpus}
                if _fresh_worker_per_job:
                    # Force worker isolation by using a unique runtime_env per task
                    import uuid as _uuid
                    _opts["runtime_env"] = {"env_vars": {"LANDSCAPY_FRESH_WORKER": str(_uuid.uuid4())}}
                ref = _create_landscape_task.options(**_opts).remote(**job)
                inflight[ref] = jidx

        _maybe_submit()
        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=30.0)
            now = _time.perf_counter()
            if done:
                ref = done[0]
                idx = inflight.pop(ref)
                try:
                    landscapes[idx] = ray.get(ref)
                except Exception as e:
                    raise
                done_count += 1
                if _show_progress:
                    _logger.info('parallel progress: %d/%d completed', done_count, total)
                # Submit next job to keep window full
                _maybe_submit()
                # checkpoint
                if ckpt_path and now - last_ckpt >= _construct_checkpoint_interval:
                    try:
                        with open(ckpt_path, 'wb') as f:
                            pickle.dump({
                                'landscapes': landscapes,
                                'jobs': construction_jobs,
                                'constructor_type': constructor_type,
                                'sampler_kwargs': sampler_kwargs,
                                'done_count': done_count,
                                'ts': now,
                            }, f)
                        last_ckpt = now
                        if _show_progress:
                            _logger.info('checkpoint written: %s', ckpt_path)
                    except Exception:
                        pass
            else:
                # heartbeat
                if _show_progress:
                    rss = ''
                    if _psutil is not None:
                        p = _psutil.Process()
                        rss_bytes = p.memory_info().rss
                        rss = f" rss={rss_bytes/1e9:.2f}GB"
                    _logger.info('parallel heartbeat: %d/%d completed%s', done_count, total, rss)

        # Initialize the FitnessSuperscape with the final list of landscapes
        return cls(
            landscapes=landscapes,
            posterior_prob_cutoff=posterior_prob_cutoff,
            _show_progress = _show_progress,
            **sampler_kwargs,
        )

    # TODO: shard with FAISS and retrieve subgraph with cosine match to
    # the query vector. Current method scales O(N^2) over exhaustive
    # graph alignment (even with anchoring): subgraphing will scale
    # linearly.

    @classmethod
    def from_streaming_construction(cls,
                                    constructor_type: Literal['undirected', 'directed'],
                                    construction_job_iter,
                                    posterior_prob_cutoff: float = 0.1,
                                    _show_progress: bool = True,
                                    _construct_checkpoint_dir: Union[str, Path, None] = None,
                                    _meta_cpu_chains: int | None = None,
                                    _fresh_worker_per_job: bool = False,
                                    _parent_task_cpus: float = 1.0,
                                    **sampler_kwargs: Any) -> "FitnessSuperscape":
        """
        Streaming variant of parallel construction. Consumes an iterator of
        job dictionaries and limits concurrency to `_meta_cpu_chains` to keep
        memory usage bounded. Useful for very large numbers of windows where
        materializing all jobs is expensive.
        """
        if not ray.is_initialized():
            try:
                ray.init(object_spilling_directory="/tmp/ray_spill")
            except Exception:
                ray.init()

        landscape_class = (
            FitnessLandscape if constructor_type == 'undirected'
            else DirectedFitnessLandscape
        )

        # Optional checkpointing directory
        ckpt_path = None
        if _construct_checkpoint_dir:
            ckpt_dir = Path(_construct_checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "superscape_construction.ckpt.pkl"

        # Submit jobs in a sliding window
        import logging as _logging, time as _time
        _logger = _logging.getLogger('fitness_landscape')
        max_inflight = int(_meta_cpu_chains) if _meta_cpu_chains and _meta_cpu_chains > 0 else (os.cpu_count() or 1)
        inflight: dict[Any, int] = {}
        landscapes: list[Union[FitnessLandscape, DirectedFitnessLandscape]] = []
        job_index = 0
        last_ckpt = 0.0

        pending_barrier = False

        def _submit_next(batch=1):
            nonlocal job_index, pending_barrier
            submitted = 0
            # Respect barrier: if set and there are inflight tasks, don't submit new ones
            if pending_barrier and inflight:
                return submitted
            while submitted < batch and len(inflight) < max_inflight:
                try:
                    job = next(construction_job_iter)
                except StopIteration:
                    return submitted
                # inject class and defaults
                job = dict(job)
                # Barrier handling: set flag and stop submitting until inflight drains
                if job.get('_barrier'):
                    pending_barrier = True
                    return submitted
                job['constructor_class'] = landscape_class
                job.setdefault('_compute_hamming_edges', False)
                wants_plm = job.get("embedding_domain") == "plm"
                wants_compute = bool(job.get("_compute_phylo_embeddings", False) or job.get("_compute_embeddings", False))
                if not wants_compute and job.get("graph_type") == "evol_diffusion" and wants_plm:
                    wants_compute = True
                num_gpus = 1 if (wants_plm and wants_compute) else 0
                # assign job id
                job.setdefault('_job_id', job_index + 1)
                job.setdefault('_total_jobs', None)
                _opts = {"num_gpus": num_gpus, "num_cpus": _parent_task_cpus}
                if _fresh_worker_per_job:
                    # Force worker isolation by using a unique runtime_env per task
                    import uuid as _uuid
                    _opts["runtime_env"] = {"env_vars": {"LANDSCAPY_FRESH_WORKER": str(_uuid.uuid4())}}
                ref = _create_landscape_task.options(**_opts).remote(**job)
                inflight[ref] = job_index
                job_index += 1
                submitted += 1
            return submitted

        # Prime submissions
        _submit_next(batch=max_inflight)

        try:
            import psutil as _psutil
        except Exception:
            _psutil = None

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1, timeout=30.0)
            now = _time.perf_counter()
            if done:
                ref = done[0]
                idx = inflight.pop(ref)
                try:
                    L = ray.get(ref)
                except Exception as e:
                    raise
                landscapes.append(L)
                if _show_progress:
                    _logger.info('stream progress: %d completed', len(landscapes))
                # If a barrier is pending and inflight is now empty, clear and continue submitting
                if pending_barrier and not inflight:
                    pending_barrier = False
                _submit_next(batch=1)
                # lightweight checkpoint
                if ckpt_path and now - last_ckpt >= 300:
                    try:
                        with open(ckpt_path, 'wb') as f:
                            pickle.dump({'landscapes': landscapes, 'done_count': len(landscapes), 'ts': now}, f)
                        last_ckpt = now
                        if _show_progress:
                            _logger.info('checkpoint written: %s', ckpt_path)
                    except Exception:
                        pass
            else:
                if _show_progress:
                    rss = ''
                    if _psutil is not None:
                        p = _psutil.Process()
                        rss_bytes = p.memory_info().rss
                        rss = f" rss={rss_bytes/1e9:.2f}GB"
                    _logger.info('stream heartbeat: %d completed%s', len(landscapes), rss)

        return cls(
            landscapes=landscapes,
            posterior_prob_cutoff=posterior_prob_cutoff,
            _show_progress=_show_progress,
            **sampler_kwargs,
        )
