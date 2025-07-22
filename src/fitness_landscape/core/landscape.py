import numpy as np
import networkx as nx
from typing import List, Union, Dict, Any, Iterable, Literal,  Protocol, runtime_checkable, Hashable
from dataclasses import dataclass
from .sequence import BaseNumpySequence, make_sequence
from .graph import create_hamming_graph
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from .graph import create_knn_graph, create_hamming_graph
from ..embedding.soft_embedding import ESMEmbedder
import inspect


class NodeModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sequence: BaseNumpySequence
    fitness: Union[float, None] = np.nan
    gapped_arr: np.ndarray = Field(..., repr=False)
    ungapped_arr: np.ndarray = Field(..., repr=False)

    @field_validator("gapped_arr")
    @classmethod
    def _check_gap(cls, v):
        if v.ndim != 2 or v.shape[1] != 21:
            raise ValueError("gapped_arr must be (L,21)")
        return v

    @field_validator("ungapped_arr")
    @classmethod
    def _check_ungap(cls, v):
        if v.ndim != 2 or v.shape[1] != 20:
            raise ValueError("ungapped_arr must be (L,20)")
        return v

SeqKey = tuple[str, ...]

@runtime_checkable
class _GraphLike(Protocol):
    def nodes(
        self, data: bool = ...
    ) -> Iterable[tuple[Hashable, dict]]: ...

@dataclass(slots=True)
class _Record:
    sequence: BaseNumpySequence
    fitness: float
    gapped_arr: np.ndarray | None = None
    ungapped_arr: np.ndarray | None = None

class BaseGraphLandscape(ABC):
    """
    Abstract base class for directed and undirected fitness landscapes.
    Implements standard class methods.

    Attributes
    ----------
    graph : _GraphLike
        The graph representation of the fitness landscape.
    sequences : list[BaseNumpySequence]
        List of sequences in the landscape.
    _records : Dict[SeqKey, _Record]
        Dictionary mapping sequence keys to records containing sequence and fitness data.
    graph_type : Literal['hamming', 'knn']
        Type of graph used in the landscape (e.g., 'hamming', 'knn').
    """
    graph: _GraphLike
    sequences: list[BaseNumpySequence]
    _records: Dict[SeqKey, _Record] 
    graph_type: Literal['hamming', 'knn'] # TODO: Add other graph types? 

    def __init__(self) -> None:
        self.sequences = []
        self._records = {}
        self.graph_type = None,
        
        self._res_emb_arr_key: str = 'residue_emb_arr'
        self._emb_arr_key: str = 'emb_arr'

    def get_fitness(self,
                    sequence,
                    *,
                    default: Union[float, None] = None) -> float:
        """
        Method to retrieve the fitness of a sequence.

        Returns
        -------
        float
            Fitness value of the sequence. If the sequence is not
            found, returns the default value if provided, otherwise
            raises KeyError.
        """

        key = tuple(make_sequence(sequence).to_array())
        try:
            return self._records[key].fitness
        except KeyError:
            if default is None:
                raise
            return default

    def get_signal(self) -> np.ndarray:
        """
        Method to retrieve the graph signal vector.

        Returns
        -------
        np.ndarray
            Array of fitness values for each sequence in the landscape.
        """
        return np.fromiter(
            (rec.fitness for rec in self._records.values()), float, len(self._records)
        )
    
    def _init_from_pairs(self,
                         seqs: List[BaseNumpySequence],
                         fits: Union[List, np.ndarray]) -> None:
        
        """
        Method to initialize the landscape from pairs of sequences and
        fitness values.

        Parameters
        ----------
        seqs : List[BaseNumpySequence]
            List of sequences to initialize the landscape with.
        fits : Union[List, np.ndarray]
            List or array of fitness values corresponding to the sequences.
        """

        for seq, fit in zip(seqs, fits):
            s = make_sequence(seq)
            self.sequences.append(s)
            self._records[tuple(s.to_array())] = _Record(sequence=s, fitness=float(fit))

    def _init_from_graph(self, graph: _GraphLike) -> None:
        self.graph = graph
        for node, data in graph.nodes(data=True):
            try:
                model = NodeModel(**data)
            except ValidationError as err:
                raise ValueError(f"Node {node!r}: {err}") from None

            seq = make_sequence(model.sequence)
            rec = _Record(
                sequence=seq,
                fitness=float(model.fitness),
                gapped_arr=getattr(model, "gapped_arr", None),
                ungapped_arr=getattr(model, "ungapped_arr", None),
            )
            self.sequences.append(seq)
            self._records[tuple(seq.to_array())] = rec


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

    @abstractmethod
    def _to_graph(self):
        """
        Abstract graph constructor method.
        """
        
        pass

    @staticmethod
    def _split_kwargs(callable_a: Any,
                      callable_b: Any,
                      kwargs: Dict[str, Any]) -> tuple[dict, dict]:
        """
        Method to split kwargs between two callables based on their
        signatures. Raises TypeError if a kwarg is ambiguous (i.e., valid
        for both callables).

        Parameters
        ----------
        callable_a : Any
            First callable to check kwargs against.
        callable_b : Any
            Second callable to check kwargs against.
        kwargs : Dict[str, Any]
            Dictionary of keyword arguments to split.
        
        Returns
        -------
        tuple[dict, dict]
            Two dictionaries containing kwargs for each callable.
            If a kwarg is ambiguous (valid for both callables), raises
            TypeError.
        """

        sig_a = inspect.signature(callable_a)
        sig_b = inspect.signature(callable_b)
        a_names = set(sig_a.parameters) - {"self"}
        b_names = set(sig_b.parameters) - {"self"}

        kw_a, kw_b = {}, {}
        for k, v in kwargs.items():
            if k in a_names and k in b_names:
                raise TypeError(f"Ambiguous kwarg '{k}' valid for both functions")
            if k in a_names:
                kw_a[k] = v
            elif k in b_names:
                kw_b[k] = v

        return kw_a, kw_b

    @classmethod
    def from_graph(cls,
                   graph: _GraphLike,
                   **kwargs):
        
        """
        Class method to create a landscape from a graph.

        Parameters
        ----------
        graph : _GraphLike
            The graph representation of the fitness landscape.
        **kwargs
            Additional keyword arguments to pass to the constructor.
        """

        return cls(graph=graph, **kwargs)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.get_fitness(self.sequences[idx])
    
    def __iter__(self):
        for seq in self.sequences:
            yield seq, self.get_fitness(seq)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_sequences={len(self.sequences)})"

    

class FitnessLandscape(BaseGraphLandscape):
    """
    Base class for fitness landscapes.
    
    Attributes
    ----------
    sequences : array-like, default=`None`
        Sequences represented as arrays of elements.
    fitness_values : array-like, default=`None`
        Fitness values corresponding to sequences.
    graph : networkx.Graph, default=`None`
        NetworkX graph representation of the landscape.
    graph_type : str, optional
        Type of graph to create if sequences are provided ('hamming', 'knn', 'custom').
    """
    
    def __init__(self,
                 graph: nx.Graph = None,
                 sequences: List[BaseNumpySequence] = None,
                 fitness_values: np.ndarray = None,
                 *,
                 graph_type: Literal['hamming'] = 'hamming',
                 emb_nodes: bool = False,
                 **kwargs) -> None:
        
        super().__init__()
        
        self.graph = None
        self.graph_type = graph_type
        
        if sequences is not None and fitness_values is not None:
            self._init_from_pairs(sequences, fitness_values)
        elif graph is not None:
            self._init_from_graph(graph)
        else:
            raise ValueError("Either sequences and fitness_values or graph must be provided")
        
        # Split kwargs
        build_kwargs, emb_kwargs = self._split_kwargs(callable_a=self._to_graph,
                                                      callable_b=self.compute_node_embeddings,
                                                      kwargs=kwargs)
        # Create graph if not provided
        if self.graph is None and graph_type is not None:
            self._to_graph(graph_type=graph_type, **build_kwargs)

        # Compute nodes
        if emb_nodes:
            self.compute_node_embeddings(**emb_kwargs)
                
    def _to_graph(self,
                 **kwargs) -> nx.Graph:
        """
        Method to convert fitness landscape to a network graph.
        
        Parameters
        ----------
        graph_type : str
            Type of graph to create. 'hamming': Connect sequences that
            differ by exactly one position. 'knn': Connect each
            sequence to its k nearest neighbors.
        **kwargs
            Additional parameters for graph creation.
            
        Returns
        -------
        networkx.Graph
            Graph representation of the landscape.
        """

        #Get fitness values for graph construction.
        fitness_values = self.get_signal()

        creation_kwargs = kwargs.copy()
        creation_kwargs.pop('graph_type', None)

        if self.graph_type == 'hamming':
            self.graph = create_hamming_graph(self.sequences, fitness_values, **creation_kwargs)

        elif self.graph_type == 'knn':
            self.graph = create_knn_graph(self.sequences, fitness_values, **creation_kwargs)

        #TODO: Other graph types can be added here.

        else:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")
    