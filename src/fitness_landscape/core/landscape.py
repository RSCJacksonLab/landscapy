import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal,  Protocol, runtime_checkable, Hashable
from dataclasses import dataclass
from .sequence import BaseNumpySequence, make_sequence
from .graph import create_hamming_graph
import scipy.linalg as la
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from .graph import create_knn_graph, create_hamming_graph
from .digraph import ASRLandscapeConstructor
from cogent3 import make_aligned_seqs
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
        """
        return np.fromiter(
            (rec.fitness for rec in self._records.values()), float, len(self._records)
        )
    
    def _init_from_pairs(self,
                         seqs,
                         fits):
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
                                model_name: str = None,
                                batch_size: int = None) -> None:
        """
        Method to get node embeddings from soft sequence OHE. Inplace
        node attribute updates.
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
    

class DirectedFitnessLandscape(BaseGraphLandscape):
    """
    Base class for directed fitness landscapes. Supports causal,
    non-stationary relationship between nodes.
    
    Attributes
    ----------
    graph : networkx.DiGraph
        NetworkX directed graph representation of the landscape.

    fitness_values : array-like, default=`None`
        Fitness values corresponding to sequences.
    """
    
    def __init__(self,
                 digraph: nx.DiGraph = None,
                 sequences: List[BaseNumpySequence] = None,
                 fitness_values: np.ndarray = None,
                 *,
                 laplacian: Union[Literal['directed'], None] = None,
                 emb_nodes: bool = False,
                 graph_type: Literal['phylogenetic_directed'] = 'phylogenetic_directed',
                 **kwargs) -> None:
        

        super().__init__()
        
        self.sequences = []
        self.fitness_values = {}
        self.graph = None
        self.graph_type = graph_type
        
        if sequences is not None and fitness_values is not None:
            self._init_from_pairs(sequences, fitness_values)
        elif digraph is not None:
            self._init_from_graph(digraph)
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

    def compute_directed_laplacian(self,
                                   teleport_dampened: bool = True,
                                   epsilon: float = 1e-8) -> None:
        """
        Method to compute the symmetrical Laplacian by converting the
        weighted inner product space of the transition matrix into a
        conventional Euclidean inner product space. If the graph is
        acycic, the transition matrix must be dampened with node
        teleporation to ensure a postivie semi-definite Laplacian.

        Parameters
        ----------
        teleport_dampened : bool, default=`True`
            Boolean to connect all nodes with a low 'teleportation'
            probability (epsilon). If `teleport_dampened` is `False`,
            the DiGraph must be strongly connected, otherwise the
            laplacian will not be positive semi-definite.
        
        epsilon : float, default=`1e-8`
            The teleportation probability.
        """

        # Constructs the matrix phi, which
        # is a diagonal matrix of the pi vector then uses phi to convert the
        # pi-weighted transition matrix P with a weighted inner product
        # space, \langle f, g \rangle_\pi = \sum_i \pi_i f_i g_i, into a
        # standard euclidean inner product space,
        # \langle f, g \rangle = \sum_i f_i g_i. The root of phi is used to
        # convert the weighted inner product space of P to a standard
        # euclidean inner product space as the norm of a weighted inner
        # product space is
        # \|f\|\pi^2 = \langle f, f\rangle\pi = \sum_i \pi_i f_i^2. Thus,
        # \Phi^{1/2} is the natural operation for the transformation of
        # f{\prime} = \Phi^{1/2} f, where f{\prime} is the euclidean
        # transformed inner product space.

        # NOTE: this only works if the graph is strongly connected.


        def _compute_transition_matrix(digraph: nx.DiGraph) -> Tuple[np.ndarray, Dict]:
            """
            Method to compute the transition matrix P from a
            directed graph. Computed from the successors of a node.
            Self-loops are added to terminal sinks in the directed graph
            such that rows sum to 1.

            Parameters
            ----------
            digraph : nx.DiGraph
                The landscape DAG.
            
            Returns
            -------
            P : np.ndarray
                the row-stohcastic transition matrix. 
        
            node_index : Dict
                The node : index dictionary.
            """
            
            nodes = list(digraph.nodes())
            n = len(nodes)
            node_index = {node: i for i, node in enumerate(nodes)}
            P = np.zeros((n, n))
            for node in nodes:
                i = node_index[node]
                successors = list(digraph.successors(node))
                if len(successors) == 0:
                    
                    # For sinks, add a self-loop so the row sums to 1.
                    P[i, i] = 1.0
                else:
                    for succ in successors:
                        j = node_index[succ]
                        P[i, j] = 1.0 / len(successors)
            
            return P, node_index

        def _compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
            """
            Function to compute the stationary distrbution of a
            row-stochastic transition matrix, where the stationary
            distribution is defined as pi^T P = pi^T. By this definition,
            these are the eigenvectors that correspond to an eigenvalue
            of 1.

            Parameters
            ----------
            P : np.ndarray
                The transition matrix.
            
            Returns
            -------
            pi : np.ndarray
                The stationary distribution of the transition matrix.
            """

            if not hasattr(self, 'transition_matrix'):
                self._compute_transition_matrix()

            # Spectral decomposition of P^T.
            vals, vecs = la.eig(self.transition_matrix.T)

            # Find the index of the eigenvalue closest to 1.
            idx = np.argmin(np.abs(vals - 1))
            pi = np.real(vecs[:, idx])
            
            # Ensure non-negativity and normalize.
            pi = np.abs(pi)
            pi = pi / np.sum(pi)
            return pi

        ####

        digraph = self.graph
        P, node_index = _compute_transition_matrix(digraph)
        self._node_index = node_index
        
        # Ensure strong connectivity if teleport dampened.
        if teleport_dampened:
            n = P.shape[0]
            Q = np.ones((n, n)) / n  # uniform teleportation matrix
            P = (1 - epsilon) * P + epsilon * Q
        self._directed_transition_matrix = P

        pi = _compute_stationary_distribution(P=P)
        self._directed_stationary_distributipn = pi

        n = P.shape[0]
        
        # Form the diagonal matrix phi and its square-root and inverse square-root.
        sqrt_phi = np.diag(np.sqrt(pi))
        inv_sqrt_phi = np.diag(1 / np.sqrt(pi))
        
        # Compute the two transformed operators.
        T1 = sqrt_phi @ P @ inv_sqrt_phi
        T2 = inv_sqrt_phi @ P.T @ sqrt_phi
        
        # Average them to enforce symmetry.
        S = 0.5 * (T1 + T2)
        
        # The directed Laplacian.
        L = np.eye(n) - S
        
        return L
    
    def _alignment_from_sequences(self,
                                  *,
                                  moltype="protein"):
        """

        """
        if hasattr(self, "graph") and self.graph is not None:
            data = {
                str(node): str(d["sequence"])              # Sequence objects stringify fine
                for node, d in self.graph.nodes(data=True)
                if "sequence" in d
            }
        else:
            data = {f"seq_{i}": str(seq) for i, seq in enumerate(self.sequences)}

        return make_aligned_seqs(data=data, moltype=moltype)

    def _to_graph(self, **kwargs):

        # build Alignment on the fly
        aln = self._alignment_from_sequences(moltype=kwargs.get("moltype", "protein"))

        constructor = ASRLandscapeConstructor(
            alignment=aln,
            _reconstruct_phylogeny=True,
            _reconstruct_ancestral_states=True,
            **kwargs,
        )
        self.graph = constructor.construct_dag()
        self._init_from_graph(self.graph)
        