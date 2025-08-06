import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import List, Union, Dict, Any, Iterable, Literal,  Protocol, runtime_checkable, Hashable, Union
from dataclasses import dataclass
from .sequence import BaseNumpySequence, make_sequence
from .graph import create_diffusion_emb_graph, create_hamming_graph, create_tda_graph, create_knn_graph
from .digraph import create_phylo_digraph, create_evol_diffusion_digraph, create_particle_filter_digraph
from .fitness import NumericFitness, CategoricalFitness, BaseFitnessLayer
from abc import ABC, abstractmethod
from ..utils import _compute_embeddings_from_sequences, alignment_to_base_numpy_sequences
import inspect
from collections import defaultdict
from cogent3 import ArrayAlignment, load_aligned_seqs
from pathlib import Path

from .._const import PROT_20


class FitnessLandscape:
    """
    FitnessLandscape is a class that represents a fitness landscape
    constructed from a networkx graph. It allows for the analysis of
    fitness layers, sequences, and their relationships.

    Attributes
    ---------- 
    sequences : List[BaseNumpySequence]
        The sequences in the fitness landscape

    graph : nx.Graph
        The instantianted graph.
    
    embeddings : np.ndarray, default=`None`
        The node embeddings.
    
    emb_arr_key.: str, default=`emb_arr`
        The keyword embeddings are stored under.
    """
    def __init__(self,
                 sequences: List[BaseNumpySequence],
                 graph: nx.Graph,
                 fitness_layers: Dict[str, BaseFitnessLayer] = None,
                 embeddings: np.ndarray = None,
                 emb_arr_key: str = 'emb_arr'):
        
        # Initialize Core Attributes with pre-computed objects
        self.sequences = sequences
        self.graph = graph
        self.fitness_layers = fitness_layers if fitness_layers is not None else {}
        self.embeddings = embeddings
        self._emb_arr_key = emb_arr_key

        # Finalize Setup and Annotate Graph
        self._annotate_graph_nodes_with_fitness()
        if self.embeddings is not None:
            self._annotate_graph_nodes_with_embeddings()

        self._records = {tuple(seq.to_array()): i for i, seq in enumerate(self.sequences)}
        self._active_view_name = next(iter(self.fitness_layers.keys())) if self.fitness_layers else None

    @classmethod
    def from_sequences(cls,
                       sequences: List[BaseNumpySequence],
                       fitness_layers: Dict[str, BaseFitnessLayer] = None,
                       graph_type: Literal['hamming', 'knn', 'tda', 'diffusion'] = 'hamming', #TODO: fix c-kNN BUG
                       embeddings: np.ndarray = None,
                       attach_embeddings: bool = True,
                       **kwargs) -> 'FitnessLandscape':
        """
        Primary factory method to create a FitnessLandscape from a list
        of sequences.

        This method orchestrates the computation of embeddings (if needed)
        and the construction of the graph based on the specified type.
        """
        embedding_based_graphs = {'knn', 'tda', 'diffusion'}

        # Secure Embeddings.
        if graph_type in embedding_based_graphs:
            if embeddings is None:

                model_name = kwargs.get('model_name', 'facebook/esm2_t6_8M_UR50D')
                batch_size = kwargs.get('batch_size', 64)
                embeddings = _compute_embeddings_from_sequences(
                    sequences,
                    model_name=model_name,
                    batch_size=batch_size
                )
        
        #  Build the Graph 
        graph_constructors = {
            'hamming': create_hamming_graph,
            'knn': create_knn_graph,
            'tda': create_tda_graph,
            'diffusion': create_diffusion_emb_graph
        }
        if graph_type not in graph_constructors:
            raise ValueError(f"Unsupported graph type for construction: {graph_type}")
        
        constructor_kwargs = kwargs
        if embeddings is not None:
            constructor_kwargs['embeddings'] = embeddings

        graph = graph_constructors[graph_type](sequences, **constructor_kwargs)

        #Instantiate the class using the simple __init__ 
        # The `attach_embeddings` flag controls embeddings in final graph.
        final_embeddings = embeddings if attach_embeddings else None
        
        return cls(sequences=sequences,
                   graph=graph,
                   fitness_layers=fitness_layers,
                   embeddings=final_embeddings,
                   emb_arr_key=kwargs.get('emb_arr_key', 'emb_arr')
                   )

    @classmethod
    def from_graph(cls,
                   graph: nx.Graph, **kwargs) -> 'FitnessLandscape':
        """
        Factory method to create a FitnessLandscape from an existing,
        annotated networkx graph.
        """

        sequences = []
        raw_layer_data = defaultdict(list)
        node_order = list(graph.nodes())

        for node in node_order:
            data = graph.nodes[node]
            if 'sequence' not in data:
                raise ValueError(f"Node {node} is missing 'sequence' attribute.")
            sequences.append(data['sequence'])
            
            for key, value in data.items():
                if key.startswith('fitness_'):
                    layer_name = key.replace('fitness_', '', 1)
                    raw_layer_data[layer_name].append(value)
        
        fitness_layers = {}
        for name, values in raw_layer_data.items():

            is_numeric = isinstance(values[0], (list, float, int, np.number))
            if is_numeric:
                numeric_values = [v if isinstance(v, list) else [v] for v in values]
                fitness_layers[name] = NumericFitness(name=name, values=numeric_values)
            else:
                all_categories = sorted(list(set(values)))
                fitness_layers[name] = CategoricalFitness(name=name, values=values, categories=all_categories)
        
        # Pop irrelevant keywords.
        kwargs.pop('graph_type', None)
        kwargs.pop('emb_nodes', None)
        
        # Call the simple constructor
        return cls(sequences=sequences,
                   graph=graph,
                   fitness_layers=fitness_layers,
                   **kwargs)
    
    # Annotation methods.
    def _annotate_graph_nodes_with_fitness(self):
        """
        Helper function to add all fitness layer data to graph nodes.
        """
        if not self.graph or not self.fitness_layers:
            return
            
        seq_to_node_map = {tuple(data['sequence'].to_array()): node_idx
                           for node_idx, data in self.graph.nodes(data=True)}

        for i, seq in enumerate(self.sequences):
            node_idx = seq_to_node_map.get(tuple(seq.to_array()))
            if node_idx is None: continue

            for name, layer in self.fitness_layers.items():
                attribute_name = f"fitness_{name}"
                self.graph.nodes[node_idx][attribute_name] = layer.get_value(i)

    def _annotate_graph_nodes_with_embeddings(self):
        """
        Helper to attach the stored embeddings to the graph nodes.
        """
        if self.graph is None or self.embeddings is None:
            return
        
        node_to_idx = {node: i for i, node in enumerate(self.graph.nodes())}
        attrs = {node: {self._emb_arr_key: self.embeddings[idx]} for node, idx in node_to_idx.items()}
        nx.set_node_attributes(self.graph, attrs)

    # Validation method.
    def _validate_data_against_graph(self,
                                     sequences: List[BaseNumpySequence],
                                     fitness_layers: Dict[str, BaseFitnessLayer]):
        """
        Method to validate the provided sequences and fitness layers
        against the current graph structure. This ensures that the
        sequences match the nodes in the graph and that the fitness
        layers are consistent with the node attributes.

        Parameters
        ----------
        sequences : List[BaseNumpySequence]
            List of sequences to validate against the graph.
        fitness_layers : Dict[str, BaseFitnessLayer]
            Dictionary of fitness layers to validate against the
            graph.

        Raises
        ------
        ValueError
            If there is a mismatch between the sequences and the graph
            nodes, or if the fitness layers do not match the attributes
            of the graph nodes.
        
        """
        if len(sequences) != self.graph.number_of_nodes():
            raise ValueError(
                f"Data inconsistency: The number of provided sequences ({len(sequences)}) "
                f"does not match the number of nodes in the graph ({self.graph.number_of_nodes()})."
            )

        graph_sequences = {
            node: tuple(data['sequence'].to_array())
            for node, data in self.graph.nodes(data=True)
        }
        provided_sequences = {i: tuple(s.to_array()) for i, s in enumerate(sequences)}

        if len(graph_sequences) != len(provided_sequences) or \
           set(graph_sequences.values()) != set(provided_sequences.values()):
            raise ValueError(
                "Data inconsistency: The set of provided sequences does not match "
                "the set of sequences stored in the graph nodes."
            )

        seq_to_node_map = {data['sequence']: node 
                           for node, data in self.graph.nodes(data=True)}

        for i, seq in enumerate(sequences):
            node_idx = seq_to_node_map.get(seq)
            if node_idx is None:

                continue

            graph_node_data = self.graph.nodes[node_idx]

            for layer_name, layer in fitness_layers.items():
                attribute_name = f"fitness_{layer_name}"
                
                if attribute_name not in graph_node_data:
                    raise ValueError(
                        f"Data inconsistency: Fitness layer '{layer_name}' exists in the "
                        f"provided dictionary but no corresponding '{attribute_name}' "
                        f"attribute was found on node {node_idx} in the graph."
                    )
                
                layer_value = layer.get_value(i)
                graph_value = graph_node_data[attribute_name]
                
                if layer_value != graph_value:
                    raise ValueError(
                        f"Data inconsistency for layer '{layer_name}' at sequence index {i} "
                        f"(node {node_idx}): The provided layer value ({layer_value}) does not "
                        f"match the graph attribute value ({graph_value})."
                    )

    @property
    def active_layer(self) -> BaseFitnessLayer:
        """
        Dynamic property to get the active fitness layer.
        """
        if self._active_view_name is None:
            raise ValueError("No active fitness layer. Use .view(layer_name) to set one.")
        return self.fitness_layers[self._active_view_name]
    

    #Fitness layer appending, modifying and viewing methods.

    def view(self, name: str) -> BaseFitnessLayer:
        """
        Retrieves a fitness layer and sets it as the new active view.
        """
        if name not in self.fitness_layers:
            raise KeyError(f"Fitness layer '{name}' not found.")
        self._active_view_name = name
        return self.fitness_layers[name]

    def attach(self,
               layer: BaseFitnessLayer) -> None:
        """
        Attaches a new fitness layer to the landscape. Fitness value
        indices must match the sequence indices.
        
        Parameters
        ----------
        layer : BaseFitnessLayer
            The initialised fitness layer to attach.    
            
        """
        # Validate the incoming layer
        if len(layer.to_scalar()) != len(self.sequences):
            raise ValueError(
                f"Cannot attach layer '{layer.name}': its length ({len(layer.to_scalar())}) "
                f"does not match the number of sequences in the landscape ({len(self.sequences)})."
            )
        
        layer_name = layer.name
        if layer_name in self.fitness_layers:
            raise ValueError(f"A layer with the name '{layer_name}' already exists.")

        # Add the layer to the dictionary
        self.fitness_layers[layer_name] = layer
        
        # If a graph exists, annotate its nodes
        if self.graph:
            seq_to_node_map = {tuple(data['sequence'].to_array()): node_idx
                               for node_idx, data in self.graph.nodes(data=True)}
            
            for i, seq in enumerate(self.sequences):
                node_idx = seq_to_node_map.get(tuple(seq.to_array()))
                if node_idx is not None:
                    attribute_name = f"fitness_{layer_name}"
                    self.graph.nodes[node_idx][attribute_name] = layer.get_value(i)

        # If this is the first layer being added, set it as the active view
        if self._active_view_name is None:
            self._active_view_name = layer_name

    def detach(self,
               layer_name: str):
        """
        Detaches a fitness layer from the landscape.

        layer_name : str
            The layer key to remove.
        """
        if layer_name not in self.fitness_layers:
            raise KeyError(f"Layer '{layer_name}' not found in the landscape.")

        # Remove the layer from the dictionary
        del self.fitness_layers[layer_name]

        # If a graph exists, remove the corresponding node attributes
        if self.graph:
            attribute_name = f"fitness_{layer_name}"
            for node_idx in self.graph.nodes():
                if attribute_name in self.graph.nodes[node_idx]:
                    del self.graph.nodes[node_idx][attribute_name]

        # If the detached layer was the active one, update the active view
        if self._active_view_name == layer_name:
            if self.fitness_layers:
                # Set the new active layer to the first available one
                self._active_view_name = next(iter(self.fitness_layers.keys()))
            else:
                # No layers left
                self._active_view_name = None


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
        if not self.graph: raise ValueError("Graph not constructed.")
        pyg_data = from_networkx(self.graph)
        if self.embeddings is not None:
            pyg_data.x = torch.tensor(self.embeddings, dtype=torch.float32)
        else:
            x_tensor = torch.tensor(np.array([s.to_one_hot() for s in self.sequences]), dtype=torch.float32)
            pyg_data.x = x_tensor.view(len(self.sequences), -1)
        for name, layer in self.fitness_layers.items():
            setattr(pyg_data, name, layer.get_tensor())
        pyg_data.num_nodes = self.graph.number_of_nodes()
        return pyg_data

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
        target_indices = []
        if sequence_idx is not None:
            target_indices = [sequence_idx] if isinstance(sequence_idx, int) else sequence_idx
        elif sequence is not None:
            sequence_list = [sequence] if isinstance(sequence, str) else sequence
            dtype = self.sequences[0].to_array().dtype
            for seq_str in sequence_list:
                seq_tuple = tuple(np.array(list(seq_str)).astype(dtype))
                idx = self._records.get(seq_tuple)
                if idx is not None: target_indices.append(idx)
                else: raise ValueError(f"Sequence '{seq_str}' not found.")
        else:
            target_indices = range(len(self.sequences))
        
        return [{'sequence_tensor': torch.tensor(self.sequences[i].to_one_hot(), dtype=torch.float32),
                 'fitness_tensors': {name: layer.get_tensor()[i] for name, layer in self.fitness_layers.items()}}
                for i in target_indices]


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
    

class DirectedFitnessLandscape(FitnessLandscape):
    """
    A fitness landscape represented by a directed graph, typically
    for phylogenetic or evolutionary trajectory data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.graph, nx.DiGraph):
            raise TypeError("DirectedFitnessLandscape requires a networkx.DiGraph object.")
    
    @classmethod
    def from_sequences(cls,
                       sequences: Union[List[BaseNumpySequence], ArrayAlignment, Path],
                       fitness_layers: Dict[str, BaseFitnessLayer] = None,
                       digraph_type: Literal['phylogenetic', 'diffusion_nq'] = 'phylogenetic',
                       embeddings: np.ndarray = None,
                       attach_embeddings: bool = True,
                       _compute_phylo_embeddings: bool = True,
                       **kwargs) -> 'DirectedFitnessLandscape':
        """
        Primary factory method to create a FitnessLandscape from a list
        of sequences.

        This method orchestrates the computation of embeddings (if needed)
        and the construction of the graph based on the specified type.
        """

        embedding_based_digraphs = {'diffusion_nq'}

        # Remove phylogenetic constructor for explicit typing.
        digraph_constructors = {
            'diffusion_nq': create_evol_diffusion_digraph,
            'diffusion_pll': None, # TODO: Directional diffusion on log-likelihood
            'particle_filter': create_particle_filter_digraph,
            }

        # Phylogenetic reconstruction requires specific types
        if digraph_type == 'phylogenetic':

            # Keep alignment for phylo and ASR.
            alignment = load_aligned_seqs(sequences) if isinstance(sequences, Path) else sequences

            embedding_kwargs = {}
            if 'model_name' in kwargs:
                embeddings_kwargs['model_name'] = kwargs.pop('model_name')
            if 'batch_size' in kwargs:
                embeddings_kwargs['batch_size'] = kwargs.pop('batch_size')
            if 'device' in kwargs:
                embeddings_kwargs['device'] = kwargs.pop('device')
            
            # Reconstruct phylogeny and ancestral states.
            digraph = create_phylo_digraph(alignment, **kwargs)
            
            # Collect sequences from the constructed graph (NOT the alignment).
            sequences = [node[1]['sequence'] for node in digraph.nodes(data=True)]

            # Logic to ensure embeddings are correctly secured for extant and ancestral sequences.
            if embeddings is not None:
                if embeddings.shape[0] != len(sequences):
                    raise ValueError(f"Embeddings expected embeddings shape {len(sequences)} in dim 0, found {embeddings.shape[0]}. Forgot ancestral sequences in precomputed embeddings?")
            
            elif _compute_phylo_embeddings:
                embeddings = _compute_embeddings_from_sequences(sequences, **embedding_kwargs)
        
            final_embeddings = embeddings if attach_embeddings else None
            
            return cls(sequences=sequences,
                   graph=digraph,
                   fitness_layers=fitness_layers,
                   embeddings=final_embeddings,
                   emb_arr_key=kwargs.get('emb_arr_key', 'emb_arr')
                   )

        # Non phylogenetic constructors where sequence typing is easy.
        # Secure Embeddings.
        if digraph_type in embedding_based_digraphs:
            if embeddings is None:

                model_name = kwargs.get('model_name', 'facebook/esm2_t6_8M_UR50D')
                batch_size = kwargs.get('batch_size', 64)
                embeddings = _compute_embeddings_from_sequences(
                    sequences,
                    model_name=model_name,
                    batch_size=batch_size
                )
    

        constructor_kwargs = kwargs
        if embeddings is not None:
            constructor_kwargs['embeddings'] = embeddings

        digraph = digraph_constructors[digraph_type](sequences, **constructor_kwargs)
            
        # Update final embeddings to attach to the graph.
        final_embeddings = embeddings if attach_embeddings else None

        return cls(sequences=sequences,
                   graph=digraph,
                   fitness_layers=fitness_layers,
                   embeddings=final_embeddings,
                   emb_arr_key=kwargs.get('emb_arr_key', 'emb_arr')
                   )

    @classmethod
    def from_graph(cls,
                   graph: nx.DiGraph, **kwargs) -> 'DirectedFitnessLandscape':
        """
        Factory method to create a FitnessLandscape from an existing,
        annotated networkx graph.
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Input graph must be a networkx.DiGraph.")

        # Undirected FitnessLandscape logic is correct, just different typing.
        return super(DirectedFitnessLandscape, cls).from_graph(graph, **kwargs)
    