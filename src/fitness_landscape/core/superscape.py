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
from cogent3 import ArrayAlignment
from ..utils import (
    PROT_20,
    alignment_to_base_numpy_sequences
)
import torch
import pickle


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
@ray.remote(num_gpus=1 if torch.cuda.is_available() else 0)
def _create_landscape_task(
    constructor_class: Union[FitnessLandscape, DirectedFitnessLandscape],
    sequences: Union[Path, ArrayAlignment, List[BaseNumpySequence]],
    fitness_layers: Dict[str, BaseFitnessLayer] = None,
    **kwargs: Any
) -> Union[FitnessLandscape, DirectedFitnessLandscape]:
    """
    A generalized Ray remote task that calls the `from_sequences` method
    of a specified landscape class.
    """
    return constructor_class.from_sequences(
        sequences=sequences,
        fitness_layers=fitness_layers,
        **kwargs
    )

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

        # Validate data types in the graph.
        self._validate_embeddings(self._landscape_graphs)
        # Validate and set the common alphabet across all landscapes.
        self.alphabet = self._validate_and_set_alphabet(self.landscapes)

        # Run RJMCMC sampling using the hierachical aligner (scales in linear time).
        # and not the RJMCMC aligner (scales in O(N^2K^2) time).
        self._hierarchical_aligner = HierarchicalRJMCMCAligner(
            graphs=self._landscape_graphs,
            aligner_params=sampler_kwargs
        )
        # The results are now stored directly, not the aligner object
        self.latent_graph, self._latent_mappings, _, _ = self._hierarchical_aligner.run_alignment()
        
        # Collect local traces 
        self.local_trace_E = self._hierarchical_aligner.local_energy_traces
        self.local_trace_NL = self._hierarchical_aligner.local_nl_traces
        self.local_trace_edges = self._hierarchical_aligner.local_edges_traces
        
        # Collect meta traces 
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
            raise RuntimeError("No posterior samples available to construct the landscape.")

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
                if 'ungapped_arr' not in data:
                    raise ValueError(f"Node {node_id!r} missing 'ungapped_arr' for superscape.")
                all_ungapped_arrs.append(data['ungapped_arr'])

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
                ambiguous_sequence = SoftSequence(uniform_posterior, self.alphabet)
                latent_sequences.append(ambiguous_sequence)
                gapped_arr = np.zeros((default_length, len(self.alphabet) + 1))
                gapped_arr[:, :-1] = uniform_posterior
                graph.nodes[latent_node_idx]['gapped_arr'] = gapped_arr
                graph.nodes[latent_node_idx]['ungapped_arr'] = uniform_posterior
                continue

            ungapped_arrs_to_align = [all_ungapped_arrs[i] for i in contributor_indices]
            contributor_probs = prob_col[contributor_indices]
            aligned_arrays, score = align_soft_sequences(sequences=ungapped_arrs_to_align, alphabet=self.alphabet)
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
                ambiguous_sequence = SoftSequence(uniform_posterior, self.alphabet)
                latent_sequences.append(ambiguous_sequence)
                continue

            ungapped_arrs_to_align = [all_ungapped_arrs[i] for i in contributor_indices]
            contributor_probs = prob_col[contributor_indices]

            aligned_arrays, _ = align_soft_sequences(sequences=ungapped_arrs_to_align, alphabet=self.alphabet)
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
            ray.init()

        landscape_class = (
            FitnessLandscape if constructor_type == 'undirected'
            else DirectedFitnessLandscape
        )

        futures = []
        for job in construction_jobs:
            if 'sequences' not in job:
                raise ValueError("Each job must have a `sequences` key.")
            elif 'graph_type' not in job and 'digraph_type' not in job:
                raise ValueError("Each job must have either `graph_type` or `digraph_type` key.")

            # Same base class to instantiate across all parallel runs.
            job['constructor_class'] = landscape_class
            futures.append(_create_landscape_task.remote(**job))

        # Retrieve the results
        landscapes = ray.get(futures)

        # Initialize the FitnessSuperscape with the final list of landscapes
        return cls(
            landscapes=landscapes,
            posterior_prob_cutoff=posterior_prob_cutoff,
            **sampler_kwargs
        )

    # TODO: shard with FAISS and retrieve subgraph with cosine match to
    # the query vector. Current method scales O(N^2) over exhaustive
    # graph alignment (even with anchoring): subgraphing will scale
    # linearly.
