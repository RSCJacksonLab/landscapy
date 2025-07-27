import numpy as np
import networkx as nx
from typing import List, Union, Dict, Any, Iterable, Literal,  Protocol, runtime_checkable, Hashable
from dataclasses import dataclass
from .sequence import BaseNumpySequence, make_sequence
from .graph import create_hamming_graph
from abc import ABC, abstractmethod
from .graph import create_knn_graph, create_hamming_graph
from ..embedding.soft_embedding import ESMEmbedder
from .fitness import BaseFitnessLayer
import inspect
from collections import defaultdict

class FitnessLandscape:
    """
    FitnessLandscape is a class that represents a fitness landscape
    constructed from a networkx graph. It allows for the analysis of
    fitness layers, sequences, and their relationships.

    Attributes
    ----------

    """
    def __init__(self,
                 sequences: List[BaseNumpySequence],
                 fitness_layers: Dict[str, BaseFitnessLayer],
                 *,
                 graph_type: Literal['hamming', 'knn'] = 'hamming',
                 graph: nx.Graph = None,
                 emb_nodes: bool = False,
                 res_emb_arr_key: str = 'residue_emb_arr',
                 emb_arr_key: str = 'emb_arr',

                 **kwargs) -> None:
        
        self.sequences = sequences
        self.fitness_layers = fitness_layers
        self.graph_type = graph_type if graph is None else 'precomputed'
        self.graph = graph
        self._res_emb_arr_key = res_emb_arr_key
        self._emb_arr_key = emb_arr_key

        # Internal mapping from sequence to its integer index for quick lookups
        self._records = {tuple(seq.to_array()): i for i, seq in enumerate(self.sequences)}

        # Set the initial active view for legacy method compatibility
        if self.fitness_layers:
            self._active_view_name = next(iter(self.fitness_layers.keys()))
        else:
            self._active_view_name = None

        # Build the graph if one wasn't provided
        if self.graph is None and self.graph_type:
            self.to_graph(**kwargs)

        # Optionally compute embeddings
        if emb_nodes:
            self.compute_node_embeddings(**kwargs)

    @property
    def active_layer(self) -> BaseFitnessLayer:
        """
        Dynamic property to get the active fitness layer.
        """
        if self._active_view_name is None:
            raise ValueError("No active fitness layer. Use .view(layer_name) to set one.")
        return self.fitness_layers[self._active_view_name]
    
    def view(self, name: str) -> BaseFitnessLayer:
        """
        Retrieves a fitness layer and sets it as the new active view.
        """
        if name not in self.fitness_layers:
            raise KeyError(f"Fitness layer '{name}' not found.")
        self._active_view_name = name
        return self.fitness_layers[name]
    
    def to_graph(self,
                 **kwargs) -> None:
        """
        Method to construct a networkx graph from the sequences and
        fitness layers. Symmetrical with the `from_graph` method.
        """
        if self.graph_type == 'hamming':
            self.graph = create_hamming_graph(self.sequences, **kwargs)
        elif self.graph_type == 'knn':
            self.graph = create_knn_graph(self.sequences, **kwargs)
        else:
            raise ValueError(f"Unsupported graph type for construction: {self.graph_type}")

        seq_to_node_map = {tuple(data['sequence'].to_array()): node_idx
                           for node_idx, data in self.graph.nodes(data=True)}

        for i, seq in enumerate(self.sequences):
            node_idx = seq_to_node_map.get(tuple(seq.to_array()))
            if node_idx is None: continue

            for name, layer in self.fitness_layers.items():
                attribute_name = f"fitness_{name}"
                # get_value() retrieves the native data (e.g., list of floats, or a string)
                self.graph.nodes[node_idx][attribute_name] = layer.get_value(i)
    
    @classmethod
    def from_graph(cls,
                   graph: nx.Graph,
                   layer_type_map: Dict[str, str] = None,
                   **kwargs) -> 'FitnessLandscape':
        """
        Creates a FitnessLandscape instance from a networkx graph.

        This method automatically detects fitness data stored in node 
        attributes that are prefixed with `fitness_`. It constructs
        the appropriate `FitnessLayer` objects for each detected
        attribute.

        Parameters
        ----------
        graph : nx.Graph
            The input graph. Nodes must have a 'sequence' attribute and
            none or more 'fitness_*' attributes.
        
        layer_type_map : Dict[str, str], optional
             A map to manually specify the type ('numeric' or
             'categorical') for ambiguous layers.

        Returns
        -------
        FitnessLandscape 
            A new landscape instance.
        """
        sequences = []
        # Use defaultdict to easily collect values for each layer
        raw_layer_data = defaultdict(list)
        
        # Ensure a consistent node order
        node_order = list(graph.nodes())

        for node in node_order:
            data = graph.nodes[node]
            if 'sequence' not in data:
                raise ValueError(f"Node {node} is missing the required 'sequence' attribute.")
            
            sequences.append(data['sequence'])
            
            for key, value in data.items():
                if key.startswith('fitness_'):
                    layer_name = key.replace('fitness_', '', 1)
                    raw_layer_data[layer_name].append(value)
        
        # Create FitnessLayer objects from the parsed data.
        fitness_layers = {}
        for name, values in raw_layer_data.items():
            
            # Infer the layer type based on the data
            inferred_type = 'numeric' if isinstance(values[0], (list, float, int)) else 'categorical'
            
            # Allow user to override inferred type
            layer_type = layer_type_map.get(name, inferred_type) if layer_type_map else inferred_type

            if layer_type == 'numeric':

                # Ensure all values are lists for NumericFitness
                numeric_values = [v if isinstance(v, list) else [v] for v in values]
                fitness_layers[name] = NumericFitness(name=name, values=numeric_values)
            
            elif layer_type == 'categorical':
                # For CategoricalFitness, we can infer all possible categories
                all_categories = sorted(list(set(values)))
                fitness_layers[name] = CategoricalFitness(name=name, values=values, categories=all_categories)
        
        if not fitness_layers:
            raise ValueError("No fitness data found in graph. Node attributes must be prefixed with 'fitness_'.")

        # Call the main constructor with the prepared data
        return cls(sequences, fitness_layers, graph_type=None, **kwargs)
    
    #TODO: Add to_graph_tensor() method.


    def compute_node_embeddings(self,
                                model_name: str = 'facebook/esm2_t6_8M_UR50D',
                                batch_size: int = 64) -> None:
        """
        Method to get node embeddings from soft sequence OHE. Inplace
        node attribute updates.

        Parameters
        ----------
        model_name : str, optional
            Name of the ESM model to use for embeddings. If not provided,
            defaults to 'esm2_t33_650M_UR50D'.
        batch_size : int, optional
            Batch size for embedding computation. If not provided,
            defaults to 64.
        """

        if not hasattr(self, 'emb_model'):
            
            self.emb_model = ESMEmbedder(model_name=model_name)

        ohe_arr = np.stack([node[1]['ungapped_arr'] for node in self.graph.nodes(data=True)])

        emb_arr = self.emb_model.embed_relaxed_seqs(relaxed_seqs=ohe_arr,
                                                batch_size=batch_size)
        
        # Iterate through nodes and update data attributes.
        for node_identifier, embedding_array in zip(self.graph.nodes(), emb_arr):
            # Update attribute in the node data.
            
            # Store residue-wise embeddings.
            self.graph.nodes[node_identifier][self._res_emb_arr_key] = embedding_array
            
            # Pool and store sequence-wise embeddings.
            pooled_array = np.mean(embedding_array, axis=0)
            self.graph.nodes[node_identifier][self._emb_arr_key] = pooled_array

    
    # Legacy methods for compatibility with old code.
    def get_fitness(self, sequence: BaseNumpySequence) -> float:
        """
        [Legacy] Method to retrieve the fitness of a sequence.

        Returns
        -------
        float
            Fitness value of the sequence. If the sequence is not
            found, returns the default value if provided, otherwise
            raises KeyError.
        """
        seq_index = self._records.get(tuple(sequence.to_array()))
        if seq_index is None:
            raise KeyError("Sequence not found in landscape.")
        return self.active_layer.to_scalar()[seq_index]

    def get_signal(self) -> np.ndarray:
        """
        [Legacy] Method to retrieve the graph signal vector.

        Returns
        -------
        np.ndarray
            Array of fitness values for each sequence in the landscape.
        """
        # Uses the new 'active_layer' property
        return self.active_layer.to_scalar()

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.get_fitness(self.sequences[idx])
    
    def __iter__(self):
        for seq in self.sequences:
            yield seq, self.get_fitness(seq)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_sequences={len(self.sequences)})"