from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from typing import Union, List, Literal, Iterable, Dict, Any
import numpy as np
from ..core.landscape import FitnessLandscape, DirectedFitnessLandscape
from ..core.sequence import BaseNumpySequence, SoftSequence
from ..core.fitness import NumericFitness, CategoricalFitness, ProbabilisticCategoricalFitness, BaseFitnessLayer
from ..graph_matching.latent_alignment import RJMCMCAligner
from ..graph_matching.hierarchical_alignment import HierarchicalRJMCMCAligner
import networkx as nx
from softalign.soft_alignment import align_soft_sequences
import ray
from pathlib import Path
from cogent3 import ArrayAlignment
from ..utils import alignment_to_base_numpy_sequences


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
@ray.remote
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
        hierarchical_aligner = HierarchicalRJMCMCAligner(
            graphs=self._landscape_graphs,
            aligner_params=sampler_kwargs
        )
        # The results are now stored directly, not the aligner object
        self.latent_graph, self._latent_mappings = hierarchical_aligner.run_alignment()
        
        # Collect local traces 
        self.local_trace_E = hierarchical_aligner.local_energy_traces
        self.local_trace_NL = hierarchical_aligner.local_nl_traces
        self.local_trace_edges = hierarchical_aligner.local_edges_traces
        
        # Collect meta traces 
        self.meta_trace_E = hierarchical_aligner.meta_energy_trace
        self.meta_trace_NL = hierarchical_aligner.meta_nl_trace
        self.meta_trace_edges = hierarchical_aligner.meta_edges_trace
        
        self.back_reference = [
            (k, node_id)
            for k, landscape in enumerate(self.landscapes)
            for node_id in landscape.graph.nodes()
        ]
        
    def construct_latent_landscape(self) :
        """
        Construct the latent graph from the posterior mapping.
        """
        # Can be directed or undirected - handle gracefully on return.
        num_total_nodes = sum(len(g.nodes()) for g in self._landscape_graphs)
        num_latent_nodes = self.latent_graph.number_of_nodes()
        
        all_prob_maps = np.zeros((num_total_nodes, num_latent_nodes))
        
        current_row = 0
        # Ensure process mappings in the correct graph order (0, 1, 2, ...)
        for k in sorted(self._latent_mappings.keys()):
            mapping_matrix = self._latent_mappings[k]
            num_nodes_in_graph = mapping_matrix.shape[0]
            # Handle cases where a graph might have no nodes mapping to the latent space
            if num_nodes_in_graph > 0:
                all_prob_maps[current_row : current_row + num_nodes_in_graph, :] = mapping_matrix
            current_row += num_nodes_in_graph

        # Collect all ungapped arrays from the nodes
        all_ungapped_arrs = [
            node_data['ungapped_arr']
            for landscape in self.landscapes
            for _, node_data in landscape.graph.nodes(data=True)
        ]

        # Collect distribution of sequence lengths for ambiguous soft sequences.
        all_lengths = [len(seq) for landscape in self.landscapes for seq in landscape.sequences]
        default_length = int(np.median(all_lengths)) if all_lengths else 1

        latent_sequences = []
        num_latent_nodes = self.latent_graph.number_of_nodes()

        for latent_node_idx in range(num_latent_nodes):
            # Get the column of probabilities for this specific latent node.
            prob_col = all_prob_maps[:, latent_node_idx]

            # Find indices of sequences that contribute to latent node.
            contributor_indices = np.where(prob_col > 0)[0]

            observed_mappings = []
            for flat_idx in contributor_indices:
                graph_idx, node_id = self.back_reference[flat_idx]
                probability = prob_col[flat_idx]
                observed_mappings.append({
                    "node_id": node_id,
                    "probability": probability,
                    "graph_index": graph_idx
                })

            observed_mappings.sort(key=lambda x: x['probability'], reverse=True)
            self.latent_graph.nodes[latent_node_idx]['observed_node_mappings'] = observed_mappings

            if len(contributor_indices) == 0:
                uniform_probability = 1.0 / len(self.alphabet)
                uniform_posterior = np.full(
                    (default_length, len(self.alphabet)),
                    uniform_probability)

                ambiguous_sequence = SoftSequence(
                    uniform_posterior,
                    self.alphabet
                )
                latent_sequences.append(ambiguous_sequence)
                # Also set the gapped_arr and ungapped_arr for the latent node
                gapped_arr = np.zeros((default_length, len(self.alphabet) + 1))
                gapped_arr[:, :-1] = uniform_posterior
                self.latent_graph.nodes[latent_node_idx]['gapped_arr'] = gapped_arr
                self.latent_graph.nodes[latent_node_idx]['ungapped_arr'] = uniform_posterior
                continue

            # Gather the specific ungapped arrays and their corresponding probabilities
            ungapped_arrs_to_align = [all_ungapped_arrs[i] for i in contributor_indices]
            contributor_probs = prob_col[contributor_indices]

            # Align the "ungapped_arr" attributes using softalign
            aligned_arrays, score = align_soft_sequences(
                sequences=ungapped_arrs_to_align,
                alphabet=self.alphabet
            )

            aligned_tensor = np.array(aligned_arrays)
            
            total_prob_for_node = np.sum(contributor_probs) + 1e-12
            
            weighted_sum_posterior = np.einsum(
                'i,ija->ja', contributor_probs, aligned_tensor
            )
            final_posterior = weighted_sum_posterior / total_prob_for_node

            # `final_posterior` is the gapped array
            self.latent_graph.nodes[latent_node_idx]['gapped_arr'] = final_posterior

            # Derive the ungapped array from the final_posterior
            ungapped_arr = final_posterior[:, :-1]
            # Renormalize
            ungapped_arr = ungapped_arr / ungapped_arr.sum(axis=1, keepdims=True)
            self.latent_graph.nodes[latent_node_idx]['ungapped_arr'] = ungapped_arr

            # Create SoftSequence for the latent landscape
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
                all_means = np.concatenate([
                    l.view(name).to_scalar() for l in self.landscapes
                ])
                total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
                weighted_sum = all_prob_maps.T @ all_means
                latent_means = (weighted_sum / total_prob_per_latent).tolist()
                latent_fitness_layers[name] = NumericFitness(name, [[m] for m in latent_means])

            elif first_layer.dtype == 'categorical':
                categories = first_layer.categories
                all_one_hot = np.concatenate([
                    l.view(name).get_tensor().numpy() for l in self.landscapes
                ])
                total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
                weighted_sum_of_one_hots = all_prob_maps.T @ all_one_hot
                latent_probabilities = weighted_sum_of_one_hots / total_prob_per_latent[:, np.newaxis]
                latent_fitness_layers[name] = ProbabilisticCategoricalFitness(name, latent_probabilities, categories)

        for i, seq in enumerate(latent_sequences):
            if i in self.latent_graph.nodes:
                self.latent_graph.nodes[i]['sequence'] = seq
        
        # Gracefully direct to correct landscape constructor.
        
        if isinstance(self.latent_graph, nx.Graph):
            self.latent_landscape = FitnessLandscape(
                sequences=latent_sequences,
                fitness_layers=latent_fitness_layers,
                graph=self.latent_graph)
        
        elif isinstance(self.latent_graph, nx.DiGraph):
            self.latent_landscape = DirectedFitnessLandscape(
                sequences=latent_sequences,
                fitness_layers=latent_fitness_layers,
                graph=self.latent_graph)
        
        else:
            raise ValueError(f"Expected latent graph to be nx.Graph or nx.DiGraph, found {type(self.latent_graph)}")


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
    
    def save(self, filepath: str):
        """Saves the FitnessSuperscape object to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
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
        **sampler_kwargs : Any
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