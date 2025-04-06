import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from .sequence import Sequence, BinarySequence, MultialleleSequence, sequence_distance
from .graph import create_hamming_graph

class FitnessLandscape:
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
                 sequences: np.ndarray = None,
                 fitness_values: np.ndarray = None,
                 graph: nx.Graph = None,
                 graph_type: Literal['hamming'] = 'hamming',
                 **kwargs) -> None:
        
        self.sequences = []
        self.fitness_values = {}
        self.graph = None
        self.graph_type = graph_type
        
        if sequences is not None and fitness_values is not None:
            self._init_from_sequences(sequences, fitness_values)
        elif graph is not None:
            self._init_from_graph(graph)
        else:
            raise ValueError("Either sequences and fitness_values or graph must be provided")
        
        # Create graph if not provided
        if self.graph is None and graph_type is not None:
            self.to_graph(graph_type=graph_type, **kwargs)
    
    def _init_from_sequences(self,
                             sequences,
                             fitness_values) -> None:
        """
        Initialise fitness landscape from sequences and fitness
        values. 

        Parameters
        ----------
        sequences : np.ndarray
            Sequences in the fitness landscape.
        
        fitness_values : np.ndarray
            fitness values indexed matched to the sequences.
        
        """
        # Convert sequences to Sequence objects if they aren't already
        self.sequences = []
        for seq in sequences:
            if isinstance(seq, Sequence):
                self.sequences.append(seq)
            else:
                self.sequences.append(Sequence(seq))
        
        # Create mapping from sequences to fitness values
        self.fitness_values = {}
        for seq, fitness in zip(self.sequences, fitness_values):
            # Use tuple representation as dictionary key
            seq_tuple = tuple(seq.to_array())
            self.fitness_values[seq_tuple] = float(fitness)
    
    def _init_from_graph(self,
                         graph: nx.Graph):
        """
        Initialise fitness landscape class from networkX graph. 

        Parameters
        ----------
        graph : nx.Graph
            The initialised network Graph fitness landscape.
        """
        self.graph = graph
        
        # Extract sequences and fitness values from graph
        self.sequences = []
        self.fitness_values = {}
        
        for node, data in graph.nodes(data=True):
            if 'sequence' in data:
                seq = data['sequence']
                if not isinstance(seq, Sequence):
                    seq = Sequence(seq)
                self.sequences.append(seq)
                
                # Use tuple representation as dictionary key
                seq_tuple = tuple(seq.to_array())
                
                if 'fitness' in data:
                    self.fitness_values[seq_tuple] = float(data['fitness'])
    
    def get_fitness(self,
                    sequence: Union[Sequence, np.ndarray]) -> float:
        """
        Method to return fitness value for a sequence.
        
        Parameters
        ----------
        sequence : Sequence or array-like
            Sequence to retrieve fitness for.
            
        Returns
        -------
        float
            Fitness value.
        """
        if not isinstance(sequence, Sequence):
            sequence = Sequence(sequence)
        
        seq_tuple = tuple(sequence.to_array())
        
        if seq_tuple in self.fitness_values:
            return self.fitness_values[seq_tuple]
        else:
            raise KeyError(f"Sequence {sequence} not found in fitness landscape")
    
    def set_fitness(self,
                    sequence: Union[Sequence, np.ndarray],
                    fitness: float) -> None:
        """
        Method to set fitness value for a sequence.
        
        Parameters
        ----------
        sequence : Sequence or array-like
            Sequence to set fitness for.
        fitness : float
            Fitness value.
        """
        if not isinstance(sequence, Sequence):
            sequence = Sequence(sequence)
        
        # Add to sequences if not already present
        if sequence not in self.sequences:
            self.sequences.append(sequence)
        
        # Update fitness value
        seq_tuple = tuple(sequence.to_array())
        self.fitness_values[seq_tuple] = float(fitness)
        
        # Update graph if it exists
        if self.graph is not None:
            # Find node corresponding to sequence
            for node, data in self.graph.nodes(data=True):
                if 'sequence' in data and np.array_equal(data['sequence'], sequence.to_array()):
                    self.graph.nodes[node]['fitness'] = float(fitness)
                    break
    
    def to_graph(self,
                 graph_type: Literal['hamming', 'knn'],
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
        if graph_type == 'hamming':
            
            self.graph = create_hamming_graph(self.sequences, list(self.fitness_values.values()), **kwargs)
        elif graph_type == 'knn':
            from .graph import create_knn_graph
            self.graph = create_knn_graph(self.sequences, list(self.fitness_values.values()), **kwargs)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
        return self.graph
    
    @classmethod
    def from_graph(cls,
                   graph: nx.Graph,
                   **kwargs):
        """
        Create landscape from graph representation.
        
        Parameters
        ----------
        graph : networkx.Graph
            Graph representation of the landscape.
            
        Returns
        -------
        FitnessLandscape
            Fitness landscape.
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
