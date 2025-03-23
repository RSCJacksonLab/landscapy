"""
Fitness landscape representations and operations.

This module provides classes and functions for representing and manipulating
fitness landscapes, which map sequences to fitness values.
"""

import numpy as np
import networkx as nx
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from .sequence import Sequence, BinarySequence, MultialleleSequence, sequence_distance


class FitnessLandscape:
    """
    Base class for fitness landscapes.
    
    Parameters
    ----------
    sequences : array-like or None
        Sequences represented as arrays of elements.
    fitness_values : array-like or None
        Fitness values corresponding to sequences.
    graph : networkx.Graph or None
        NetworkX graph representation of the landscape.
    graph_type : str, optional
        Type of graph to create if sequences are provided ('hamming', 'knn', 'custom').
    """
    
    def __init__(self, sequences=None, fitness_values=None, graph=None, graph_type='hamming', **kwargs):
        self.sequences = []
        self.fitness_values = {}
        self.graph = None
        
        if sequences is not None and fitness_values is not None:
            self._init_from_sequences(sequences, fitness_values)
        elif graph is not None:
            self._init_from_graph(graph)
        else:
            raise ValueError("Either sequences and fitness_values or graph must be provided")
        
        # Create graph if not provided
        if self.graph is None and graph_type is not None:
            self.to_graph(graph_type=graph_type, **kwargs)
    
    def _init_from_sequences(self, sequences, fitness_values):
        """Initialize from sequences and fitness values."""
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
    
    def _init_from_graph(self, graph):
        """Initialize from NetworkX graph."""
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
    
    def get_fitness(self, sequence):
        """
        Get fitness value for a sequence.
        
        Parameters
        ----------
        sequence : Sequence or array-like
            Sequence to get fitness for.
            
        Returns
        -------
        float
            Fitness value.
            
        Raises
        ------
        KeyError
            If sequence is not in the landscape.
        """
        if not isinstance(sequence, Sequence):
            sequence = Sequence(sequence)
        
        seq_tuple = tuple(sequence.to_array())
        
        if seq_tuple in self.fitness_values:
            return self.fitness_values[seq_tuple]
        else:
            raise KeyError(f"Sequence {sequence} not found in fitness landscape")
    
    def set_fitness(self, sequence, fitness):
        """
        Set fitness value for a sequence.
        
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
    
    def to_graph(self, graph_type='hamming', **kwargs):
        """
        Convert to graph representation.
        
        Parameters
        ----------
        graph_type : str, optional
            Type of graph to create:
            - 'hamming': Connect sequences that differ by exactly one position
            - 'knn': Connect each sequence to its k nearest neighbors
            - 'custom': Use custom graph creation function
        **kwargs
            Additional parameters for graph creation.
            
        Returns
        -------
        networkx.Graph
            Graph representation of the landscape.
        """
        if graph_type == 'hamming':
            from .graph import create_hamming_graph
            self.graph = create_hamming_graph(self.sequences, list(self.fitness_values.values()), **kwargs)
        elif graph_type == 'knn':
            from .graph import create_knn_graph
            self.graph = create_knn_graph(self.sequences, list(self.fitness_values.values()), **kwargs)
        elif graph_type == 'custom':
            if 'create_graph_func' not in kwargs:
                raise ValueError("Custom graph type requires 'create_graph_func' parameter")
            create_graph_func = kwargs.pop('create_graph_func')
            self.graph = create_graph_func(self.sequences, list(self.fitness_values.values()), **kwargs)
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")
        
        return self.graph
    
    @classmethod
    def from_graph(cls, graph, **kwargs):
        """
        Create landscape from graph representation.
        
        Parameters
        ----------
        graph : networkx.Graph
            Graph representation of the landscape.
        **kwargs
            Additional parameters.
            
        Returns
        -------
        FitnessLandscape
            Fitness landscape.
        """
        return cls(graph=graph, **kwargs)
    
    def analyze(self, method, **kwargs):
        """
        Analyze landscape using specified method.
        
        Parameters
        ----------
        method : str
            Analysis method ('epistasis', 'ruggedness', 'paths', etc.)
        **kwargs
            Additional parameters for the analysis method.
            
        Returns
        -------
        dict
            Results of the analysis.
        """
        if method == 'epistasis':
            from ..analysis.epistasis import calculate_epistasis
            return calculate_epistasis(self, **kwargs)
        elif method == 'ruggedness':
            from ..analysis.ruggedness import calculate_ruggedness
            return calculate_ruggedness(self, **kwargs)
        elif method == 'paths':
            from ..analysis.paths import find_adaptive_paths
            return find_adaptive_paths(self, **kwargs)
        else:
            raise ValueError(f"Unsupported analysis method: {method}")
    
    def transform(self, transform_type, **kwargs):
        """
        Apply mathematical transformation to the landscape.
        
        Parameters
        ----------
        transform_type : str
            Type of transform ('walsh', 'fourier', 'eigenmode', etc.)
        **kwargs
            Additional parameters for the transformation.
            
        Returns
        -------
        object
            Result of the transformation.
        """
        if transform_type == 'walsh':
            from ..transforms.walsh_hadamard import walsh_transform
            return walsh_transform(self, **kwargs)
        elif transform_type == 'fourier':
            from ..transforms.graph_fourier import graph_fourier_transform
            return graph_fourier_transform(self.graph, **kwargs)
        elif transform_type == 'eigenmode':
            from ..transforms.eigenmode import eigenmode_decomposition
            return eigenmode_decomposition(self.graph, **kwargs)
        else:
            raise ValueError(f"Unsupported transform type: {transform_type}")
    
    def visualize(self, method='network', **kwargs):
        """
        Visualize the fitness landscape.
        
        Parameters
        ----------
        method : str
            Visualization method ('network', 'heatmap', '3d', etc.)
        **kwargs
            Additional parameters for visualization.
            
        Returns
        -------
        matplotlib.Figure or None
            Figure object if return_fig=True, otherwise None.
        """
        from ..utils.visualization import visualize_landscape
        return visualize_landscape(self, method=method, **kwargs)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.get_fitness(self.sequences[idx])
    
    def __iter__(self):
        for seq in self.sequences:
            yield seq, self.get_fitness(seq)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n_sequences={len(self.sequences)})"
