import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import List, Union, Dict, Any, Iterable, Literal,  Protocol, runtime_checkable, Hashable, Union, Tuple, Mapping, Callable, Optional
from dataclasses import dataclass
from .sequence import BaseNumpySequence, make_sequence
from .graph import (
    create_diffusion_emb_graph,
    create_hamming_graph,
    create_tda_graph,
    create_knn_graph,
    _encode_multiallele,
    create_phylo_graph,
    create_evol_diffusion_graph,
)
from .digraph import create_phylo_digraph, create_evol_diffusion_digraph, create_particle_filter_digraph
from .fitness import NumericFitness, CategoricalFitness, BaseFitnessLayer, ProbabilisticCategoricalFitness
from abc import ABC, abstractmethod
from ..utils import _compute_embeddings_from_sequences, alignment_to_base_numpy_sequences
import inspect
from collections import defaultdict
from cogent3 import ArrayAlignment, load_aligned_seqs
from pathlib import Path
import warnings
from .._const import PROT_20

GraphCtor = Callable[..., nx.Graph]

@dataclass(frozen=True)
class _GraphRegistryItem:
    fn: GraphCtor
    needs_embeddings: bool

_GRAPH_REGISTRY: dict[str, _GraphRegistryItem] = {
    "hamming":   _GraphRegistryItem(create_hamming_graph, needs_embeddings=False),
    "knn":       _GraphRegistryItem(create_knn_graph, needs_embeddings=True),
    "tda":       _GraphRegistryItem(create_tda_graph, needs_embeddings=True),
    "diffusion": _GraphRegistryItem(create_diffusion_emb_graph, needs_embeddings=True),
    # phylogenetic handled separately (alignment/ASR path)
}
def _resolve_embeddings_for_graph(sequences: list[BaseNumpySequence],
                                  graph_type: str,
                                  embeddings: Optional[np.ndarray],
                                  embedding_domain: Literal['plm', 'ohe'],
                                  *,
                                  model_name: str,
                                  batch_size: int,
                                  device: Optional[str],) -> Tuple[Optional[np.ndarray], dict]:
    """
    Helper function to resolve embeddings for a graph type.

    Parameters
    ----------
    sequences : list[BaseNumpySequence]
        List of sequences to compute embeddings for.
    graph_type : str
        The type of graph to create (e.g., 'hamming', 'knn', 'tda',
        'diffusion').
    
    embeddings : Optional[np.ndarray]
        Pre-computed embeddings, if available. If `None`, embeddings
        will be computed.

    embedding_domain : Literal['plm', 'ohe']
        The domain of the embeddings. 'plm' for pre-trained language
        model embeddings, 'ohe' for one-hot encoded sequences.
    
    model_name : str
        The name of the pre-trained model to use for embeddings if
        `embedding_domain` is 'plm'.

    batch_size : int
        The batch size to use for computing embeddings if
        `embedding_domain` is 'plm'.

    device : Optional[str]
        The device to use for computing embeddings if
        `embedding_domain` is 'plm'. If `None`, defaults to the

    Returns 
    -------
    Tuple[Optional[np.ndarray], dict]
        Returns a tuple containing the embeddings and a dictionary
        with additional keyword arguments for the graph constructor.
    """
    reg = _GRAPH_REGISTRY.get(graph_type)
    if reg is None or not reg.needs_embeddings:
        return embeddings, {}

    if embeddings is not None:
        return embeddings, {"embeddings": embeddings}

    if embedding_domain == "plm":
        E = _compute_embeddings_from_sequences(
            sequences, model_name=model_name, batch_size=batch_size, device=device
        )
        return E, {"embeddings": E}

    if embedding_domain == "ohe":
        E, _ = _encode_multiallele(sequences)
        return E, {"embeddings": E}

    raise ValueError(f"embedding_domain must be 'plm' or 'ohe', got {embedding_domain!r}")

SeqKey = Union['BaseNumpySequence', str, Tuple]

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
        # Safe canonical node ordering.
        self._node_order = list(graph.nodes())  
        self._seq_to_nodes = self._build_seq_multimap()  # duplicate-safe
        self._nodes_by_index = {i: n for i, n in enumerate(self._node_order)}  # 0..N-1 -> node key
        self._annotate_graph_nodes_with_fitness()
        if self.embeddings is not None:
            self._annotate_graph_nodes_with_embeddings()
        self._records = self._build_sequence_index() 
        self._enforce_unique_sequences()
        
        if self.fitness_layers:
            self._active_view_name = (
                'default' if 'default' in self.fitness_layers
                else next(iter(self.fitness_layers.keys()))
            )
        else:
            self._active_view_name = None

    def _build_seq_multimap(self) -> Dict[Tuple, List]:
        """
        Helper function to map sequence array tuple. Safe for
        duplicates. 

        Returns
        -------
        mm : Dict
            The sequnce array to node value mapping.
        """
        mm: dict[tuple, list] = {}
        for n, data in self.graph.nodes(data=True):
            arr = tuple(data['sequence'].to_array())
            mm.setdefault(arr, []).append(n)
        return mm
    
    def _build_sequence_index(self) -> Dict[Tuple, int]:
        """
        Keep first occurrence index for fast get_fitness.
        """
        idx = {}
        for i, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            if key not in idx:
                idx[key] = i
        return idx
    
    def _index_map(self) -> Dict[Tuple, list[int]]:
        """
        Map sequence-array tuple -> [indices in self.sequences]
        (duplicate-safe).
        """
        m: dict[Tuple, list[int]] = {}
        for i, s in enumerate(self.sequences):
            key = tuple(s.to_array())
            m.setdefault(key, []).append(i)
        return m
    
    def _normalize_seq_key(self, k: SeqKey) -> Tuple:
        """
        Normalize a sequence-like key to a tuple of symbols (hashable).
        """
        if hasattr(k, "to_array"):
            return tuple(k.to_array())
        if isinstance(k, str):
            dtype = self.sequences[0].to_array().dtype
            return tuple(np.array(list(k)).astype(dtype))
        if isinstance(k, (tuple, list, np.ndarray)):
            return tuple(list(k))
        raise TypeError(f"Unsupported sequence key type: {type(k)}")
    
    # Annotation methods.
    def _annotate_graph_nodes_with_fitness(self):
        """
        Helper function to add all fitness layer data to graph nodes.
        """
        if not self.graph or not self.fitness_layers:
            return
            
        for name, layer in self.fitness_layers.items():
            # Raise error if sequences are missing labels
            layer._validate_length(len(self.sequences), name=f"during annotation ({name})")

        for i, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            nodes = self._seq_to_nodes.get(key, [])
            if not nodes:
                # Skip quietly, or raise error?
                continue
            
            for node in nodes:
                for name, layer in self.fitness_layers.items():
                    self.graph.nodes[node][f"fitness_{name}"] = layer.get_value(i)

    def _enforce_unique_sequences(self):
        """
        Helper function to enforce only unique sequences.
        """
        dupes = [k for k, v in self._seq_to_nodes.items() if len(v) > 1]
        if dupes:
            warnings.warn(f"Duplicate sequences detected for {len(dupes)} keys; "
                      f"downstream `attach()` policies will handle them.")
    
    def _annotate_graph_nodes_with_embeddings(self):
        """
        Helper to attach the stored embeddings to the graph nodes.
        """
        if self.graph is None or self.embeddings is None:
            return
        if self.embeddings.shape[0] != len(self._node_order):
            raise ValueError("Embeddings rows != number of graph nodes; cannot annotate safely.")
        attrs = {node: {self._emb_arr_key: self.embeddings[i]}
                for i, node in enumerate(self._node_order)}
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
            node_idx = seq_to_node_map.get(tuple(seq.to_array()))
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

    def view(self,
             name: str) -> BaseFitnessLayer:
        """
        Retrieves a fitness layer and sets it as the new active view.
        Main entry point for accessing fitness layers.

        Parameters
        ----------
        name : str
            The name of the fitness layer to retrieve.
        
        Returns
        -------
        BaseFitnessLayer
            The fitness layer corresponding to the provided name.
        """
        if name not in self.fitness_layers:
            raise KeyError(f"Fitness layer '{name}' not found.")
        self._active_view_name = name
        return self.fitness_layers[name]
    
    def add(self,
            **kwargs):
        """
        Convenience function to expedite fitness layer construction via
        the `attach` method.
        """
        if 'layer' in kwargs and kwargs['layer'] is not None:
            raise ValueError("`.add` builds from values; use `.attach(layer=...)` to attach a ready layer.")
        return self.attach(**kwargs)

    def attach(self,
            layer: BaseFitnessLayer | None = None,
            *,
            name: str = None,
            values = None,
            dtype: Literal['numeric','categorical'] = None,
            categories: list[str] = None,
            map_by: Literal['index','sequence'] = 'index',
            on_duplicates: Literal['error','first','all','aggregate'] = 'error',
            allow_missing: bool = False) -> None:
        """
        Method to attach a fitness layer to the landscape.
        
        Parameters
        ----------
        layer : BaseFitnessLayer, optional
            A pre-constructed fitness layer to attach. If provided,
            it overrides the other parameters (name, values, dtype,
            categories). If `None`, the other parameters must be
            provided.
        
        name : str, optional
            The name of the fitness layer to create. Required if
            `layer` is not provided.
        
        values : list, dict, or iterable, optional
            The values to use for the fitness layer. If `map_by` is
            'index', this should be a list of values aligned with the
            sequences. If `map_by` is 'sequence', this should be a
            mapping of sequence keys to values (e.g., dict or iterable
            of tuples). Required if `layer` is not provided.
        
        dtype : Literal['numeric', 'categorical'], optional
            The data type of the fitness layer. Must be 'numeric' or
            'categorical'. Required if `layer` is not provided.

        categories : list[str], optional
            The categories for a categorical fitness layer. Required if
            `dtype` is 'categorical' and `layer` is not provided.
        
        map_by : Literal['index', 'sequence'], default='index'
            How to map the `values` to sequences. If 'index', the
            `values` should be a list aligned with the sequences.
            If 'sequence', the `values` should be a mapping of
            sequence keys to values (e.g., dict or iterable of tuples).
        
        on_duplicates : Literal['error', 'first', 'all', 'aggregate'], default='error'
            How to handle duplicate sequences when mapping values.
            - 'error': Raise an error if duplicates are found.
            - 'first': Use the first value for duplicates.
            - 'all': Use the value for all duplicates.
            - 'aggregate': Merge values for duplicates (only for numeric).
        
        allow_missing : bool, default=False
            If `True`, allows sequences to not have a value assigned.
        """

        if layer is not None:
            if any(x is not None for x in (name, values, dtype, categories)):
                raise ValueError("Provide either `layer` or (name, values, dtype...), not both.")
            if len(layer.to_scalar()) != len(self.sequences):
                raise ValueError(
                    f"Cannot attach layer '{layer.name}': its length ({len(layer.to_scalar())}) "
                    f"does not match the number of sequences ({len(self.sequences)})."
                )
            layer_name = layer.name
            if layer_name in self.fitness_layers:
                raise ValueError(f"A layer with the name '{layer_name}' already exists.")
            self.fitness_layers[layer_name] = layer
            # annotate graph
            if self.graph:
                seq_to_node_map = {tuple(data['sequence'].to_array()): node_idx
                                for node_idx, data in self.graph.nodes(data=True)}
                for i, seq in enumerate(self.sequences):
                    node_idx = seq_to_node_map.get(tuple(seq.to_array()))
                    if node_idx is not None:
                        self.graph.nodes[node_idx][f"fitness_{layer_name}"] = layer.get_value(i)
            if self._active_view_name is None:
                self._active_view_name = layer_name
            return

        # Construct from values
        if name is None or values is None or dtype is None:
            raise ValueError("When not passing `layer`, you must provide name, values, and dtype.")

        n = len(self.sequences)

        # Resolve mapping by index
        if map_by == 'index':
            # Expect values to be a list aligned to sequences length
            if len(values) != n:
                raise ValueError(f"`values` length {len(values)} != number of sequences {n}")
            # Normalize to concrete layer
            if dtype == 'numeric':
                norm = [[v] if not isinstance(v, (list, tuple, np.ndarray)) else list(v) for v in values]
                new_layer = NumericFitness(name=name, values=norm)
            elif dtype == 'categorical':
                if categories is None:
                    categories = sorted(list(set(values)))
                new_layer = CategoricalFitness(name=name, values=list(values), categories=categories)
            else:
                raise ValueError("For probabilistic categories, use dtype='categorical' with `values`=probabilities and pass categories, or attach a ProbabilisticCategoricalFitness layer explicitly.")
            
            # Delegate to regular layer attachment.
            return self.attach(new_layer)

        # Resolve mapping by sequence key
        if map_by != 'sequence':
            raise ValueError(f"Unknown map_by: {map_by}")

        # Normalize `values` into a dict {Tuple(seq) -> value}
        if isinstance(values, Mapping):
            items = list(values.items())
        else:

            items = list(values)

        key_to_val = {self._normalize_seq_key(k): v for k, v in items}

        # Create a per-index container
        if dtype == 'numeric':
            idx_values: list[list[float]] = [[] for _ in range(n)]
        elif dtype == 'categorical':
            idx_values: list[Any] = [None] * n
        else:
            raise ValueError("dtype must be `numeric` or `categorical` here; pass a ready layer object otherwise.")

        # Build index map for duplicates.
        idx_map = self._index_map()

        # Private helper.
        def _apply_numeric(idx_list: list[int],
                           v):
            
            reps = v if isinstance(v, (list, tuple, np.ndarray)) else [float(v)]
            
            if on_duplicates == 'error' and len(idx_list) > 1:
                raise ValueError("Duplicate sequences found; set `on_duplicates` to `first`, `all`, or `aggregate`.")
            
            # Collect only first
            if on_duplicates == 'first':
                idx_values[idx_list[0]] = list(reps)
            
            # Collect all
            elif on_duplicates == 'all':
                for i in idx_list:
                    idx_values[i] = list(reps)
            
            # merge replicate lists across all matches
            elif on_duplicates == 'aggregate':
            
                merged = []
                for i in idx_list:
                    merged.extend(reps)
                for i in idx_list:
                    idx_values[i] = list(merged)
            else:
                raise ValueError(f"Unknown `on_duplicates` option: {on_duplicates}")

        # Private helper.
        def _apply_categorical(idx_list: list[int],
                               v):
            
            if on_duplicates == 'error' and len(idx_list) > 1:
                raise ValueError("Duplicate sequences found; set on_duplicates to 'first' or 'all'.")
            
            if on_duplicates == 'first':
                idx_values[idx_list[0]] = v
            
            elif on_duplicates == 'all':
                for i in idx_list:
                    idx_values[i] = v
            
            elif on_duplicates == 'aggregate':
                raise ValueError("on_duplicates='aggregate' is not supported for categorical.")
            
            else:
                raise ValueError(f"Unknown on_duplicates: {on_duplicates}")

        # Fill index containers
        seen = set()
        for key, v in key_to_val.items():
            idxs = idx_map.get(key, [])
            if not idxs:
                if allow_missing:
                    continue
                raise KeyError(f"Sequence {key} not found in landscape.")
            
            seen.add(key)
            
            if dtype == 'numeric':
                _apply_numeric(idxs, v)
            
            else:
                _apply_categorical(idxs, v)

        # If unfilled indices:
        if dtype == 'numeric':
            missing = [i for i, r in enumerate(idx_values) if len(r) == 0]
        
        else:
            missing = [i for i, r in enumerate(idx_values) if r is None]
        
        if missing and not allow_missing:
            raise ValueError(f"{len(missing)} sequences were not assigned a value. Use `allow_missing=True` to skip.")

        # Build the concrete layer
        if dtype == 'numeric':
            # For any unassigned (allow_missing=True), give NaN replicate so shape is valid
            idx_values = [r if r else [np.nan] for r in idx_values]
            new_layer = NumericFitness(name=name, values=idx_values)
        else:
            if categories is None:
                categories = sorted(list({v for v in idx_values if v is not None}))
            # Replace None with a placeholder category if allow_missing
            if allow_missing:
                if "__MISSING__" not in categories:
                    categories = categories + ["__MISSING__"]
                idx_values = [v if v is not None else "__MISSING__" for v in idx_values]
            new_layer = CategoricalFitness(name=name, values=idx_values, categories=categories)

        # Delegate to regular constructor.
        return self.attach(new_layer)

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

        for i, seq in enumerate(self.sequences):
            key = tuple(seq.to_array())
            for node in self._seq_to_nodes.get(key, []):
                self.graph.nodes[node].pop(attribute_name, None)

        # If the detached layer was the active one, update the active view
        if self._active_view_name == layer_name:
            if self.fitness_layers:
                # Set the new active layer to the first available one
                self._active_view_name = next(iter(self.fitness_layers.keys()))
            else:
                # No layers lefts
                self._active_view_name = None

    @property
    def active_layer_name(self) -> str | None:
        return getattr(self, "_active_view_name", None)

    def get_layer(self,
                  name: str,
                  *,
                  allow_active_default: bool = True):
        """
        Method to get return a layer. 

        Parameters
        ----------
        name : str
            The layer name. 
        
        allow_active_default : bool, default=`True`
            Boolean to include the active layer in be resolved by the
            method.
        
        Returns
        -------
        FitnessLayer
            The resolved fitness layer.
        """
        d = self.fitness_layers
        if name in d:
            return d[name]
        for lyr in d.values():
            if getattr(lyr, "name", None) == name:
                return lyr
        if allow_active_default and name == "default":
            active = self.active_layer_name
            if active and active in d:
                return d[active]

        raise KeyError(f"Layer '{name}' not found. Available keys: {list(d.keys())}; "
                       f"active={self.active_layer_name!r}")


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
    
    # Constructor methods.
    @classmethod
    def from_sequences(cls,
                       sequences: List[BaseNumpySequence],
                       fitness_layers: Dict[str, BaseFitnessLayer] = None,
                       graph_type: Literal['hamming', 'knn', 'tda', 'diffusion', 'evol_diffusion'] = 'hamming',
                       embeddings: np.ndarray = None,
                       embedding_domain: Literal['plm', 'ohe'] = 'ohe',
                       attach_embeddings: bool = True,
                       _compute_phylo_embeddings: bool = False,
                       **kwargs) -> 'FitnessLandscape':
        """
        Primary factory method to create a FitnessLandscape from a list
        of sequences.This method orchestrates the computation of
        embeddings (if needed) and the construction of the graph based
        on the specified type.

        Parameters
        ----------
        sequences : List[BaseNumpySequence]
            The sequences to build the landscape from. 
        
        fitess_layers : Dict[str, BaseFitnessLayer]
            The fitness landscape fitness layer(s).
        
        graph_type: str, default=`hamming`
            The graph constructor type. Options are:
            - `hamming` : The hamming graph where edges connect nodes
            with a single mutation difference. 
            - `knn` : k-nearest neighbor graph where each node is
            connected to it's k nearest neigbors in the embedding
            domain.
            - `tda` : topological data analysis where edges are
            computed by persistent homology in the embedding domain. 
            - `diffusion` : Edges are computed according to a diffusion
            process from a sequence evolution replacement matrix.
        
        embeddings : np.ndarray, default=`None`
            Pre-computed embeddings, indexed by sequence.
        
        embedding_domain : str, default=`ohe`
            The embedding domain for auto embedding construction.
            Options are:
            - `ohe`: one-hot encoded domain.
            - `plm`: protein language model embeddings. Embedding
            parameters must be provided in **kwargs. 
        
        attach_embeddings : bool, default=`True`
            Boolean to attach embeddings as node attributes.

        _compute_phylo_embeddings : bool, default=`False` 
            Boolean to compute embeddings for phylogenetic seqeucnes.
        
        Returns
        -------
        FitnessLandscape
            The constructed fitness landscape object.
        """
        embedding_based_graphs = {'tda', 'diffusion', 'evol_diffusion'}

        # pop kwargs that will break constructor_kwargs
        emb_arr_key = kwargs.pop('emb_arr_key', 'emb_arr')
        model_name = kwargs.pop('model_name', 'facebook/esm2_t6_8M_UR50D')
        batch_size = kwargs.pop('batch_size', 64)
        device = kwargs.pop('device', None)

        #  Build the Graph 
        graph_constructors = {
            'hamming': create_hamming_graph,
            'knn': create_knn_graph,
            'tda': create_tda_graph,
            'diffusion': create_diffusion_emb_graph,
            'evol_diffusion': create_evol_diffusion_graph,
        }

        # Phylogenetic reconstruction requires specific types
        if graph_type == 'phylogenetic':

            # Keep alignment for phylo and ASR.
            alignment = load_aligned_seqs(sequences, moltype='protein') if isinstance(sequences, Path) else sequences
            from ..utils import sanitize_alignment
            alignment = sanitize_alignment(alignment)

            # Reconstruct phylogeny and ancestral states.
            nested = bool(kwargs.pop('_nested_construction_parallel', False))
            graph = create_phylo_graph(alignment, _nested_parallel=nested, **kwargs)
            
            # Collect sequences from the constructed graph (NOT the alignment).
            sequences = [node[1]['sequence'] for node in graph.nodes(data=True)]

            # Logic to ensure embeddings are correctly secured for extant and ancestral sequences.
            if embeddings is not None:
                if embeddings.shape[0] != len(sequences):
                    raise ValueError(f"Embeddings expected embeddings shape {len(sequences)} in dim 0, found {embeddings.shape[0]}. Forgot ancestral sequences in precomputed embeddings?")
            
            elif _compute_phylo_embeddings:
                if embedding_domain == 'plm':
                    embeddings = _compute_embeddings_from_sequences(sequences,
                                                                    model_name=model_name,
                                                                    device=device,
                                                                    batch_size=batch_size)
                elif embedding_domain == 'ohe':
                    embeddings, _ = _encode_multiallele(sequences)
                # leave as computed
            else:
                # Allow proceeding without embeddings; Superscape can attach fallbacks later
                embeddings = None

            final_embeddings = embeddings if attach_embeddings else None
            
            return cls(sequences=sequences,
                   graph=graph,
                   fitness_layers=fitness_layers,
                   embeddings=final_embeddings,
                   emb_arr_key=emb_arr_key
                   )

        # Secure Embeddings.
        # How to balance PLM emb vs ohe?
        if graph_type in embedding_based_graphs:
            if embeddings is None:
                # Secure PLM embeddings
                if embedding_domain == 'plm':
                    model_name = model_name
                    batch_size = batch_size
                    embeddings = _compute_embeddings_from_sequences(
                        sequences,
                        model_name=model_name,
                        batch_size=batch_size
                    )
                # Secure OHE embeddings
                elif embedding_domain == 'ohe':
                    embeddings, _ = _encode_multiallele(sequences)
                
                else:
                    raise ValueError(f"Expected `embedding_domain` to be either `plm` or `ohe`, found {embedding_domain}")
        
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
                   emb_arr_key=emb_arr_key
                   )

    @classmethod
    def from_graph(cls,
                   graph: nx.Graph, **kwargs) -> 'FitnessLandscape':
        """
        Factory method to create a FitnessLandscape from an existing,
        annotated networkx graph.
        """

        node_list = list(graph.nodes())
        sequences = []
        raw_layer_data = defaultdict(list)

        for node in node_list:
            data = graph.nodes[node]
            if 'sequence' not in data:
                raise ValueError(f"Node {node} is missing 'sequence' attribute.")
            sequences.append(data['sequence'])
            for k, v in data.items():
                if k.startswith('fitness_'):
                    raw_layer_data[k[8:]].append(v)

        # length validation
        for name, values in raw_layer_data.items():
            if len(values) != len(node_list):
                raise ValueError(f"Layer '{name}' length {len(values)} != node count {len(node_list)}.")
        
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
    
    
    @classmethod
    def build(cls,
              sequences: list[BaseNumpySequence],
              *,
              graph: str | nx.Graph = "hamming",
              fitness_layers: dict[str, BaseFitnessLayer] | None = None,
              embeddings: np.ndarray | None = None,
              embedding_domain: Literal["plm", "ohe"] = "ohe",
              attach_embeddings: bool = True,
              emb_arr_key: str = "emb_arr",
              # PLM knobs (ignored for ohe/hamming)
              model_name: str = "facebook/esm2_t6_8M_UR50D",
              batch_size: int = 64,
              device: str | None = None,
              **graph_kwargs) -> "FitnessLandscape":
        """
        Constructor method for main entry to FitnessLandscape
        initialisaition. 

        Parameters
        ----------
        sequences : list[BaseNumpySequence]
            List of sequences to build the landscape from.
        
        graph : str or nx.Graph, default=`"hamming"`
            The graph type or an existing networkx graph. If a string,
            it should be one of the registered graph types.
        
        fitness_layers : dict[str, BaseFitnessLayer], optional
            Dictionary of fitness layers to attach to the landscape.
        
        embeddings : np.ndarray, optional
            Pre-computed embeddings for the sequences. If `None`, they
            will be computed based on the `embedding_domain`.
        
        embedding_domain : str, default=`"ohe"`
            The domain for embeddings. Options are:
            - `"plm"`: Protein language model embeddings.
            - `"ohe"`: One-hot encoded sequences.
        
        attach_embeddings : bool, default=`True`
            Whether to attach embeddings as node attributes in the graph.
        
        emb_arr_key : str, default=`"emb_arr"`
            The key under which embeddings will be stored in the graph
            nodes.
        
        model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
            The name of the model to use for PLM embeddings.
        
        batch_size : int, default=`64`
            Batch size for PLM embedding computation.
        
        device : str or None, default=`None`
            Device to use for PLM embedding computation (e.g., "cpu" or "cuda").
        
        graph_kwargs : dict
            Additional keyword arguments to pass to the graph constructor.
        
        Returns
        -------
        FitnessLandscape
            The constructed fitness landscape object.
        """
        if isinstance(graph, nx.Graph):
            # annotate & return
            G = graph
        else:
            gtype = str(graph)
            if gtype == "phylogenetic":
                raise ValueError("Use FitnessLandscape.from_alignment(...) for phylogenetic graphs.")
            reg = _GRAPH_REGISTRY.get(gtype)
            if reg is None:
                raise ValueError(f"Unknown graph type {gtype!r}. Options: {list(_GRAPH_REGISTRY)}")

            # resolve embeddings only if needed
            embeddings, extra = _resolve_embeddings_for_graph(
                sequences, gtype, embeddings, embedding_domain,
                model_name=model_name, batch_size=batch_size, device=device
            )
            ctor = reg.fn
            G = ctor(sequences, **graph_kwargs, **extra)

        # Attach embeddings to nodes if flagged
        final_embeddings = embeddings if attach_embeddings else None
        return cls(sequences=sequences,
                   graph=G,
                   fitness_layers=fitness_layers,
                   embeddings=final_embeddings,
                   emb_arr_key=emb_arr_key)

    @classmethod
    def from_alignment(cls,
                       alignment: ArrayAlignment | Path,
                       *,
                       fitness_layers: dict[str, BaseFitnessLayer] | None = None,
                       attach_embeddings: bool = True,
                       emb_arr_key: str = "emb_arr",
                       # PLM knobs for auto-embeddings on extant+ancestral
                       embedding_domain: Literal["plm", "ohe"] = "ohe",
                       model_name: str = "facebook/esm2_t6_8M_UR50D",
                       batch_size: int = 64,
                       device: str | None = None,
                       _compute_phylo_embeddings: bool = False,
                       **phylo_kwargs) -> "FitnessLandscape":
        """
        Constructor method to create a FitnessLandscape from an
        alignment or a path to an alignment file. Convenience wrapper
        around the phylogenetic graph constructor.

        Parameters
        ----------
        alignment : ArrayAlignment or Path
            The alignment object or path to an alignment file.
        
        fitness_layers : dict[str, BaseFitnessLayer], optional
            Dictionary of fitness layers to attach to the landscape.
        
        attach_embeddings : bool, default=`True`
            Whether to attach embeddings as node attributes in the graph.
        
        emb_arr_key : str, default=`"emb_arr"`
            The key under which embeddings will be stored in the graph
            nodes.
        
        embedding_domain : str, default=`"ohe"`
            The domain for embeddings. Options are:
            - `"plm"`: Protein language model embeddings.
            - `"ohe"`: One-hot encoded sequences.
        
        model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
            The name of the model to use for PLM embeddings.
        
        batch_size : int, default=`64`
            Batch size for PLM embedding computation.
        
        device : str or None, default=`None`
            Device to use for PLM embedding computation (e.g., "cpu" or "cuda").
        
        _compute_phylo_embeddings : bool, default=`False`
            Whether to compute embeddings for phylogenetic sequences.
        
        phylo_kwargs : dict
            Additional keyword arguments to pass to the phylogenetic graph constructor.
        
        Returns
        -------
        FitnessLandscape
            The constructed fitness landscape object.
        """

        aln = load_aligned_seqs(alignment) if isinstance(alignment, Path) else alignment
        G = create_phylo_graph(aln, **phylo_kwargs)
        seqs = [data["sequence"] for _, data in G.nodes(data=True)]

        E = None
        if _compute_phylo_embeddings:
            if embedding_domain == "plm":
                E = _compute_embeddings_from_sequences(seqs, model_name=model_name, batch_size=batch_size, device=device)
            elif embedding_domain == "ohe":
                E, _ = _encode_multiallele(seqs)
            else:
                raise ValueError(f"embedding_domain must be 'plm' or 'ohe', got {embedding_domain!r}")

        return cls(sequences=seqs,
                   graph=G,
                   fitness_layers=fitness_layers,
                   embeddings=(E if attach_embeddings else None),
                   emb_arr_key=emb_arr_key)

    @classmethod
    def from_graph_annotated(cls, graph: nx.Graph, **kwargs) -> "FitnessLandscape":
        """
        Thin alias around  existing `from_graph` for parity with other APIs.
        """
        return cls.from_graph(graph, **kwargs)

    
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
    
    # Custom constructor methods.
    @classmethod
    def from_sequences(cls,
                       sequences: Union[List[BaseNumpySequence], ArrayAlignment, Path],
                       fitness_layers: Dict[str, BaseFitnessLayer] = None,
                       digraph_type: Literal['phylogenetic', 'diffusion_nq', 'particle_filter'] = 'phylogenetic',
                       embeddings: np.ndarray = None,
                       embedding_domain: Literal['plm', 'ohe'] = 'ohe',
                       attach_embeddings: bool = True,
                       _compute_phylo_embeddings: bool = False,
                       **kwargs) -> 'DirectedFitnessLandscape':
        """
        Primary factory method to create a DirectedFitnessLandscape
        from a list of sequences.This method orchestrates the
        computation of embeddings (if needed) and the construction of
        the digraph based on the specified type.

        Parameters
        ----------
        sequences : List[BaseNumpySequence]
            The sequences to build the landscape from. 
        
        fitess_layers : Dict[str, BaseFitnessLayer]
            The fitness landscape fitness layer(s).
        
        digraph_type: str, default=`phylogenetic`
            The digraph constructor type. Options are:
            - `phylogenetic` : Reconstruction of ancestral state soft
            sequences and a directed phylogenetic topology using a 
            non-equilibrium non-time reversible transition matrix.
            - `diffusion_nq` : Edges are computed according to a diffusion
            process from an asymmetric sequence evolution replacement
            matrix.
            - `particle_filter` : Nodes and edges are computed by
            forward sampling logits on a protein language model.
        
        embeddings : np.ndarray, default=`None`
            Pre-computed embeddings, indexed by sequence.
        
        embedding_domain : str, default=`ohe`
            The embedding domain for auto embedding construction.
            Options are:
            - `ohe`: one-hot encoded domain.
            - `plm`: protein language model embeddings. Embedding
            parameters must be provided in **kwargs. 
        
        attach_embeddings : bool, default=`True`
            Boolean to attach embeddings as node attributes.

        _compute_phylo_embeddings : bool, default=`False` 
            Boolean to compute embeddings for phylogenetic seqeucnes.
        
        Returns
        -------
        DirectedFitnessLandscape
            The constructed directed fitness landscape object.
        """

        # Allow alias 'digraph' to match other entry points
        if 'digraph' in kwargs:
            digraph_type = str(kwargs.pop('digraph'))

        embedding_based_digraphs = {'diffusion_nq'}

        # pop kwargs that will break constructor_kwargs
        emb_arr_key=kwargs.pop('emb_arr_key', 'emb_arr')
        model_name = kwargs.pop('model_name', 'facebook/esm2_t6_8M_UR50D')
        batch_size = kwargs.pop('batch_size', 64)
        device = kwargs.pop('device', None)

        # Remove phylogenetic constructor for explicit typing.
        digraph_constructors = {
            'diffusion_nq': create_evol_diffusion_digraph,
            'particle_filter': create_particle_filter_digraph,
            }

        # Phylogenetic reconstruction requires specific types
        if digraph_type == 'phylogenetic':

            # Keep alignment for phylo and ASR.
            alignment = load_aligned_seqs(sequences, moltype='protein') if isinstance(sequences, Path) else sequences
            from ..utils import sanitize_alignment
            alignment = sanitize_alignment(alignment)

            # Reconstruct phylogeny and ancestral states.
            nested = bool(kwargs.pop('_nested_construction_parallel', False))
            digraph = create_phylo_digraph(alignment, _nested_parallel=nested, **kwargs)
            
            # Collect sequences from the constructed graph (NOT the alignment).
            sequences = [node[1]['sequence'] for node in digraph.nodes(data=True)]

            # Logic to ensure embeddings are correctly secured for extant and ancestral sequences.
            if embeddings is not None:
                if embeddings.shape[0] != len(sequences):
                    raise ValueError(f"Embeddings expected embeddings shape {len(sequences)} in dim 0, found {embeddings.shape[0]}. Forgot ancestral sequences in precomputed embeddings?")
            
            elif _compute_phylo_embeddings:
                if embedding_domain == 'plm':
                    embeddings = _compute_embeddings_from_sequences(sequences,
                                                                    model_name=model_name,
                                                                    device=device,
                                                                    batch_size=batch_size)
                elif embedding_domain == 'ohe':
                    embeddings, _ = _encode_multiallele(sequences)
                else:
                    raise ValueError(f"embedding_domain must be 'plm' or 'ohe', got {embedding_domain!r}")
            else:
                # Allow proceeding without embeddings; Superscape can attach fallbacks later
                embeddings = None
        
            final_embeddings = embeddings if attach_embeddings else None
            
            return cls(sequences=sequences,
                   graph=digraph,
                   fitness_layers=fitness_layers,
                   embeddings=final_embeddings,
                   emb_arr_key=emb_arr_key
                   )

        # Non phylogenetic constructors where sequence typing is easy.
        # Secure Embeddings.
        if digraph_type in embedding_based_digraphs:
            if embeddings is None:
                # Secure PLM embeddings
                if embedding_domain == 'plm':
                    model_name = model_name
                    batch_size = batch_size
                    embeddings = _compute_embeddings_from_sequences(
                        sequences,
                        model_name=model_name,
                        batch_size=batch_size
                    )
                # Secure OHE embeddings
                elif embedding_domain == 'ohe':
                    embeddings, _ = _encode_multiallele(sequences)
                
                else:
                    raise ValueError(f"Expected `embedding_domain` to be either `plm` or `ohe`, found {embedding_domain}")
        
        if digraph_type not in digraph_constructors:
            raise ValueError(f"Unsupported graph type for construction: {digraph_type}")

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
                   emb_arr_key=emb_arr_key 
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
    


    @classmethod
    def build(cls,
              sequences: list[BaseNumpySequence] | ArrayAlignment | Path,
              *,
              digraph: str | nx.DiGraph = "phylogenetic",
              fitness_layers: dict[str, BaseFitnessLayer] | None = None,
              embeddings: np.ndarray | None = None,
              embedding_domain: Literal["plm", "ohe"] = "ohe",
              attach_embeddings: bool = True,
              emb_arr_key: str = "emb_arr",
              model_name: str = "facebook/esm2_t6_8M_UR50D",
              batch_size: int = 64,
              device: str | None = None,
              _compute_phylo_embeddings: bool = False,
              **kwargs) -> "DirectedFitnessLandscape":
        
        """
        Constructor method for main entry to DirectedFitnessLandscape.

        Parameters
        ----------
        sequences : list[BaseNumpySequence] | ArrayAlignment | Path
            List of sequences or an alignment to build the landscape from.

        digraph : str or nx.DiGraph, default=`"phylogenetic"`
            The directed graph type or an existing networkx directed graph.
            If a string, it should be one of the registered digraph types.

        fitness_layers : dict[str, BaseFitnessLayer], optional
            Dictionary of fitness layers to attach to the landscape.

        embeddings : np.ndarray, optional
            Pre-computed embeddings for the sequences. If `None`, they
            will be computed based on the `embedding_domain`.

        embedding_domain : str, default=`"ohe"`
            The domain for embeddings. Options are:
            - `"plm"`: Protein language model embeddings.
            - `"ohe"`: One-hot encoded sequences.

        attach_embeddings : bool, default=`True`
            Whether to attach embeddings as node attributes in the graph.

        emb_arr_key : str, default=`"emb_arr"`
            The key under which embeddings will be stored in the graph nodes.

        model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
            The name of the model to use for PLM embeddings.

        batch_size : int, default=`64`
            Batch size for PLM embedding computation.

        device : str or None, default=`None`
            Device to use for PLM embedding computation (e.g., "cpu" or "cuda").

        _compute_phylo_embeddings : bool, default=`False`
            Whether to compute embeddings for phylogenetic sequences.

        kwargs : dict
            Additional keyword arguments to pass to the digraph constructor.

        Returns
        -------
        DirectedFitnessLandscape
            The constructed directed fitness landscape object.
        """

        if isinstance(digraph, nx.DiGraph):
            DG = digraph
            seqs = [data["sequence"] for _, data in DG.nodes(data=True)]
            E = embeddings
            # attach optional embeddings
            final_E = E if attach_embeddings else None
            return cls(sequences=seqs, graph=DG, fitness_layers=fitness_layers,
                       embeddings=final_E, emb_arr_key=emb_arr_key)

        dtype = str(digraph)
        if dtype == "phylogenetic":
            aln = load_aligned_seqs(sequences) if isinstance(sequences, Path) else sequences
            DG = create_phylo_digraph(aln, **kwargs)
            seqs = [data["sequence"] for _, data in DG.nodes(data=True)]

            E = None
            if _compute_phylo_embeddings:
                if embedding_domain == "plm":
                    E = _compute_embeddings_from_sequences(seqs, model_name=model_name, batch_size=batch_size, device=device)
                elif embedding_domain == "ohe":
                    E, _ = _encode_multiallele(seqs)
                else:
                    raise ValueError(f"embedding_domain must be 'plm' or 'ohe', got {embedding_domain!r}")

            return cls(sequences=seqs, graph=DG, fitness_layers=fitness_layers,
                       embeddings=(E if attach_embeddings else None), emb_arr_key=emb_arr_key)

        # embedding-based directed constructors 
        ctor_map = {
            "diffusion_nq": create_evol_diffusion_digraph,
            "particle_filter": create_particle_filter_digraph,
        }
        if dtype not in ctor_map:
            raise ValueError(f"Unknown digraph type {dtype!r}. Options: {list(ctor_map)}")

        seqs = sequences if not isinstance(sequences, (Path, ArrayAlignment)) else alignment_to_base_numpy_sequences(sequences)
        # resolve embeddings if needed 
        E, extra = _resolve_embeddings_for_graph(
            seqs, "diffusion", embeddings, embedding_domain,
            model_name=model_name, batch_size=batch_size, device=device
        )
        DG = ctor_map[dtype](seqs, **kwargs, **extra)
        return cls(sequences=seqs, graph=DG, fitness_layers=fitness_layers,
                   embeddings=(E if attach_embeddings else None), emb_arr_key=emb_arr_key)
    
def read_csv_landscape(path: str | Path,
                       *,
                       sequence_col: str = "sequence",
                       id_col: str | None = None,
                       alphabet: Iterable | None = None,
                       moltype: str | None = None,
                       graph: str | nx.Graph = "hamming",
                       
                       # layer parsing
                       numeric_layers: list[str] | None = None, # e.g., ["fitness", "score"]
                       replicate_prefixes: dict[str, list[str]] | None = None, # {"fitness": ["fitness.rep1","fitness.rep2"]}
                       categorical_layers: list[str] | None = None, # e.g., ["label"]
                       probabilistic_specs: dict[str, list[str]] | None = None, # {"label": ["label=A","label=B","label=C"]}
                       
                       # embeddings for graph if needed
                       embeddings: np.ndarray | None = None,
                       embedding_domain: Literal["plm", "ohe"] = "ohe",
                       attach_embeddings: bool = True,
                       emb_arr_key: str = "emb_arr",
                       model_name: str = "facebook/esm2_t6_8M_UR50D",
                       batch_size: int = 64,
                       device: str | None = None) -> FitnessLandscape:
    """
    Function to initialise a FitnessLandscape from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing the landscape data.
    
    sequence_col : str, default=`"sequence"`
        The column name in the CSV that contains the sequences.
    
    id_col : str or None, default=`None`
        Optional column name for sequence IDs (not used in landscape).
    
    alphabet : Iterable, optional
        The alphabet to use for sequence encoding. If None, defaults to
        the standard alphabet for the specified moltype.
    
    moltype : str, optional
        The molecular type of the sequences (e.g., "protein", "dna").
    
    graph : str or nx.Graph, default=`"hamming"`
        The graph type or an existing networkx graph to use.
    
    numeric_layers : list[str] | None, optional
        List of numeric layer names to parse from the CSV.
    
    replicate_prefixes : dict[str, list[str]] | None, optional
        Dictionary mapping layer names to lists of replicate column names.
    
    categorical_layers : list[str] | None, optional
        List of categorical layer names to parse from the CSV.
    
    probabilistic_specs : dict[str, list[str]] | None, optional
        Dictionary mapping layer names to lists of probabilistic column names.
    
    embeddings : np.ndarray | None, optional
        Pre-computed embeddings for the sequences. If None, they will be computed.
    
    embedding_domain : Literal["plm", "ohe"], default=`"ohe"`
        The domain for embeddings. Options are:
        - `"plm"`: Protein language model embeddings.
        - `"ohe"`: One-hot encoded sequences.
    
    attach_embeddings : bool, default=`True`
        Whether to attach embeddings as node attributes in the graph.
    
    emb_arr_key : str, default=`"emb_arr"`
        The key under which embeddings will be stored in the graph nodes.
    
    model_name : str, default=`"facebook/esm2_t6_8M_UR50D"`
        The name of the model to use for PLM embeddings.
    
    batch_size : int, default=`64`
        Batch size for PLM embedding computation.
    
    device : str or None, default=`None`
        Device to use for PLM embedding computation (e.g., "cpu" or "cuda").
    
    Returns
    -------
    FitnessLandscape
        The constructed fitness landscape object.
    """

    df = pd.read_csv(path)

    if sequence_col not in df.columns:
        raise ValueError(f"sequence_col '{sequence_col}' not found in CSV columns {list(df.columns)}")

    # build sequences
    seqs = [make_sequence(s, alphabet=alphabet, moltype=moltype) for s in df[sequence_col].tolist()]

    layers: dict[str, BaseFitnessLayer] = {}

    # numeric scalar columns
    if numeric_layers:
        for name in numeric_layers:
            if name not in df.columns:
                raise ValueError(f"Numeric layer column '{name}' not found")
            layers[name] = NumericFitness.from_scalars(name, df[name].to_numpy())

    # numeric replicate groups
    if replicate_prefixes:
        for name, cols in replicate_prefixes.items():
            for c in cols:
                if c not in df.columns:
                    raise ValueError(f"Replicate column '{c}' for layer '{name}' not found")
            reps = df[cols].to_numpy(dtype=float)  # shape (N, R)
            # convert rows to list[list]
            rep_lists = [row[~np.isnan(row)].tolist() if np.isnan(row).any() else row.tolist()
                         for row in reps]
            layers[name] = NumericFitness.from_replicates(name, rep_lists)

    # categorical single-column layers
    if categorical_layers:
        for name in categorical_layers:
            if name not in df.columns:
                raise ValueError(f"Categorical layer column '{name}' not found")
            vals = df[name].astype(str).tolist()
            layers[name] = CategoricalFitness.from_values(name, vals)

    # probabilistic layers (wide)
    if probabilistic_specs:
        for name, cols in probabilistic_specs.items():
            for c in cols:
                if c not in df.columns:
                    raise ValueError(f"Probabilistic column '{c}' for layer '{name}' not found")
            P = df[cols].to_numpy(dtype=float)
            cats = [c.split("=", 1)[1] if "=" in c else c for c in cols]
            layers[name] = ProbabilisticCategoricalFitness.from_probabilities(name, P, categories=cats)

    # build graph qnd landscape (using the unified builder).
    L = FitnessLandscape.build(
        sequences=seqs,
        graph=graph,
        fitness_layers=layers if layers else None,
        embeddings=embeddings,
        embedding_domain=embedding_domain,
        attach_embeddings=attach_embeddings,
        emb_arr_key=emb_arr_key,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    return L

def to_csv_landscape(L: FitnessLandscape,
                     path: str | Path,
                     *,
                     sequence_col: str = "sequence",
                     include_layers: bool = True) -> None:
    
    """
    Function to write a FitnessLandscape to a CSV file.
    
    Parameters
    ----------
    L : FitnessLandscape
        The fitness landscape object to write to CSV.
    
    path : str or Path
        The path where the CSV file will be saved.
    
    sequence_col : str, default=`"sequence"`
        The column name for sequences in the output CSV.
    
    include_layers : bool, default=`True`
        Whether to include fitness layers in the output CSV.
    """
    rows = []
    for i, s in enumerate(L.sequences):
        row = {sequence_col: s.to_str()}
        if include_layers:
            for name, layer in L.fitness_layers.items():
                if layer.dtype == "numeric":
                    row[name] = float(layer.to_scalar()[i])
                elif layer.dtype == "categorical":
                    row[name] = layer.get_value(i)
            # (extend to probabilistic; emit wide columns).
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
