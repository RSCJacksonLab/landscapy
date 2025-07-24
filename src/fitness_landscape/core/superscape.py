from pydantic import BaseModel, Field, field_validator, ValidationError  
from typing import Union, List, Literal, Iterable
import numpy as np
from ..core.landscape import FitnessLandscape, _GraphLike, NodeModel
from ..core.sequence import BaseNumpySequence, SoftSequence
from ..graph_matching.latent_alignment import RJMCMCAligner
import networkx as nx
from softalign.soft_alignment import align_soft_sequences

class EmbNodeModel(BaseModel):
    emb_arr: np.ndarray = Field(..., repr=False)

    @field_validator("emb_arr")
    @classmethod
    def _check_emb(cls, v):
        v = np.asarray(v)
        if v.ndim != 1:
            raise ValueError("emb_arr must be a 1-D array")
        return v

class FitnessSuperscape:
    """
    """

    def __init__(self,
                 
                 landscapes: List[Union[FitnessLandscape, _GraphLike]],
                 posterior_prob_cutoff: float = 0.1,
                 **sampler_kwargs) -> None:
        
        self.landscapes = landscapes
        self._landscape_graphs = self._extract_graphs(landscapes=self.landscapes)

        # Validate data types in the graph.
        self._validate_embeddings(self._landscape_graphs)
        # Validate and set the common alphabet across all landscapes.
        self.alphabet = self._validate_and_set_alphabet(self.landscapes)

        # Run RJMCMC sampling
        self.graph_aligner = RJMCMCAligner(self._landscape_graphs,
                                            **sampler_kwargs)
        self.graph_aligner.sample()
        
        # Mappings for nodes to latent embeddings.
        self._posterior_mapping = self.graph_aligner.posterior_match_probabilities()
        self._latent_mappings = self.graph_aligner.get_node_to_latent_mapping()
        self._posterior_prob_cutoff = posterior_prob_cutoff
        
        
    def construct_latent_landscape(self) :
        """
        Construct the latent graph from the posterior mapping.
        """
        self.latent_graph = self.graph_aligner.latent_blueprint_graph(posterior_prob_cutoff=self._posterior_prob_cutoff)

        # Aggregate data into flat data structures.
        all_prob_maps = np.vstack(self._latent_mappings)
        all_fitnesses = np.concatenate([
            #TODO: Make sure using correct API.
            landscape.fitness_values for landscape in self.landscapes
        ])
        all_sequences = [
            seq for landscape in self.landscapes for seq in landscape.sequences
        ]

        # Back referece map for quick lookup.
        back_reference = [
            (k, node_id)
            for k, landscape in enumerate(self.landscapes)
            for node_id in landscape.nodes
        ]

        # Collect distribution of sequence lengths for ambiguous soft sequences.
        all_lengths = [len(seq) for seq in all_sequences]
        default_length = int(np.median(all_lengths)) if all_lengths else 1

        # Vectorised latent fitnesses.
        #TODO: update fitness to a new class and not a float.
        total_prob_per_latent = all_prob_maps.sum(axis=0) + 1e-12
        weighted_fitness_sum = all_prob_maps.T @ all_fitnesses
        latent_fitnesses_array = weighted_fitness_sum / total_prob_per_latent
        latent_sequences = []
        num_latent_nodes = self.latent_graph.number_of_nodes()

        for latent_node_idx in range(num_latent_nodes):
            # Get the column of probabilities for this specific latent node.
            prob_col = all_prob_maps[:, latent_node_idx]

            # Find indices of sequences that contribute to latent node.
            contributor_indices = np.where(prob_col > 0)[0]
            
            # 
            observed_mappings = []
            for flat_idx in contributor_indices:
                graph_idx, node_id = back_reference[flat_idx]
                probability = prob_col[flat_idx]
                # TODO: add node name.
                observed_mappings.append({
                    "node_id": node_id,
                    "probability": probability,
                    "graph_index": graph_idx
                })
            
            # Sort the mappings by probability
            observed_mappings.sort(key=lambda x: x['probability'], reverse=True)
            
            # Set the attribute on the corresponding node in the latent graph.
            self.latent_graph.nodes[latent_node_idx]['observed_node_mappings'] = observed_mappings            
            
            # If no nodes map here, create an ambiguous soft sequence.
            if len(contributor_indices) == 0:

                uniform_probability = 1.0 / len(self.alphabet)
                
                uniform_posterior = np.full(
                    (default_length, len(self.alphabet)),
                    uniform_probability)
                
                # Create ambiguous SoftSequence object and append it.
                ambiguous_sequence = SoftSequence(
                    uniform_posterior,
                    self.alphabet
                )
                latent_sequences.append(ambiguous_sequence)
                continue
        
            # Gather the specific sequences and their corresponding probabilities
            sequences_to_align = [all_sequences[i] for i in contributor_indices]
            contributor_probs = prob_col[contributor_indices]

            # Return a list of matrices of the same shape.
            # TODO: soft sequence alignment.
            aligned_matrices, score = self.align_soft_sequences(sequences_to_align)

            # Stack the aligned matrices into a 3D array.
            aligned_tensor = np.array(aligned_matrices)
            weighted_sum_posterior = np.einsum(
                'i,ija->ja', contributor_probs, aligned_tensor
            )
            final_posterior = weighted_sum_posterior / total_prob_per_latent[latent_node_idx]

            latent_sequences.append(
                SoftSequence(final_posterior, self.alphabet)
            )

        self.latent_landscape = FitnessLandscape(
            latent_sequences, latent_fitnesses_array
        )
        self.latent_landscape.graph = self.latent_graph


    @staticmethod
    def _validate_embeddings(graphs: list[nx.DiGraph]) -> None:
        """
        Helper method to validate nodes have valid emb_arr attribute.

        Parameters
        ----------
        graphs : List
            List of nx.DiGraph objects to be aligned.
        """
        for G in graphs:
            for node, data in G.nodes(data=True):
                try:
                    EmbNodeModel(**data)        # will raise if missing/invalid
                except ValidationError as e:
                    raise ValueError(f"{node!r}: {e}") from None
                
                # Node model missed if input is just graphs and not landscape.
                try: 
                    NodeModel(**data)  # will raise if missing/invalid
                except ValidationError as e:
                    raise ValueError(f"{node!r}: {e}") from None
    
    @staticmethod
    def _validate_and_set_alphabet(landscapes: List[FitnessLandscape]) -> list:
        """
        Validates that all sequences across all landscapes share a
        common alphabet and returns it.

        Parameters
        ----------
        landscapes : List[FitnessLandscape]
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
        # Avoids building a potentially large list in memory.
        all_alphabets_gen = (
            (i, j, seq.alphabet)
            for i, landscape in enumerate(landscapes)
            if isinstance(landscape, FitnessLandscape) and landscape.sequences
            for j, seq in enumerate(landscape.sequences)
        )

        try:
            # Get the first alphabet to use as the reference.
            _, _, reference_alphabet = next(all_alphabets_gen)
            reference_alphabet_set = set(reference_alphabet)
        except StopIteration:
            # If no sequences are found, raise an error.
            raise ValueError("Could not determine alphabet: no sequences found in any of the provided landscapes.")

        # Check all remaining alphabets in the generator against the reference.
        for i, j, current_alphabet in all_alphabets_gen:
            if set(current_alphabet) != reference_alphabet_set:
                raise ValueError(
                    f"Inconsistent alphabets found. "
                    f"Alphabet in landscape {i}, sequence {j} "
                    f"({set(current_alphabet)}) does not match the "
                    f"reference alphabet ({reference_alphabet_set})."
                )
        
        return reference_alphabet
                
    @staticmethod
    def _extract_graphs(landscapes: Iterable[Union[FitnessLandscape,
                                                   _GraphLike,
                                                   nx.DiGraph]]) -> list[nx.DiGraph]:
        """
        Helper method to extract directed graphs from directed fitness
        landscapes.

        Parameters
        ----------
        landscapes : Iterable
            The list of DirectedFitnessLandscapes, _GraphLike or
            nx.DiGraph objects.

        Returns
        -------
        out : list
            The list of nx.DiGraph objects indexed matched to the
            landscapes.
        """
        out = []
        for obj in landscapes:
            if isinstance(obj, FitnessLandscape):
                G = obj.graph
            elif isinstance(obj, nx.Graph):
                G = nx.Graph(obj)  # copy/upgrade
            elif isinstance(obj, _GraphLike):
                G = nx.Graph(obj) # last resort
            else:
                raise TypeError(f"Unsupported landscape/graph type: {type(obj)}")
            if not isinstance(G, nx.Graph):
                G = nx.Graph(G)
            out.append(G)
        return out
    
    
    # TODO: shard with FAISS and retrieve subgraph with cosine match to
    # the query vector. Current method scales O(N^2) over exhaustive
    # graph alignment (even with anchoring): subgraphing will scale
    # linearly.