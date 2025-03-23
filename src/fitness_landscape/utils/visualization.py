"""
Diffusion Axes Visualization for fitness landscapes.

This module implements the method described in:
McCandlish, D. M. (2011). Visualizing fitness landscapes.
Evolution, 65(6), 1544-1558.

The method creates low-dimensional representations of fitness landscapes using diffusion maps,
which plot genotypes in a manner that captures important features of the landscape.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from ..core.landscape import FitnessLandscape
from ..core.sequence import Sequence, BinarySequence, MultialleleSequence
from ..core.graph import create_hamming_graph, create_knn_graph


class DiffusionAxesVisualization:
    """
    Diffusion Axes Visualization for fitness landscapes.
    
    This class implements the method described by McCandlish for creating low-dimensional
    representations of fitness landscapes using diffusion maps. The method plots genotypes
    in a manner that captures important features of the landscape.
    
    Attributes:
        sequence_length (int): Length of sequences in the landscape
        alphabet_size (int): Size of the alphabet for each position
        n_components (int): Number of diffusion components to compute
        diffusion_coordinates (Dict): Dictionary mapping sequences to diffusion coordinates
        eigenvalues (np.ndarray): Eigenvalues of the diffusion operator
        eigenvectors (np.ndarray): Eigenvectors of the diffusion operator
        sequences (List): List of sequences used in the analysis
    """
    
    def __init__(self, sequence_length: int, alphabet_size: int = 2, n_components: int = 3):
        """
        Initialize the DiffusionAxesVisualization model.
        
        Args:
            sequence_length: Length of sequences in the landscape
            alphabet_size: Size of the alphabet for each position (default: 2 for binary sequences)
            n_components: Number of diffusion components to compute (default: 3)
        """
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.n_components = n_components
        self.diffusion_coordinates = {}
        self.eigenvalues = None
        self.eigenvectors = None
        self.sequences = []
        self._is_fitted = False
        
    def fit_transform(self, landscape: Union[FitnessLandscape, Dict, List[Tuple]], 
                     alpha: float = 1.0, t: float = 1.0) -> Dict:
        """
        Compute diffusion coordinates for sequences in a landscape.
        
        Args:
            landscape: A FitnessLandscape object, dictionary mapping sequences to fitness values,
                      or a list of (sequence, fitness) tuples
            alpha: Normalization parameter for the diffusion process (default: 1.0)
            t: Time parameter for the diffusion process (default: 1.0)
            
        Returns:
            Dict: Dictionary mapping sequences to diffusion coordinates
        """
        # Extract sequences and fitnesses
        if isinstance(landscape, FitnessLandscape):
            sequences = list(landscape.genotype_to_fitness.keys())
            fitnesses = list(landscape.genotype_to_fitness.values())
        elif isinstance(landscape, dict):
            sequences = list(landscape.keys())
            fitnesses = list(landscape.values())
        elif isinstance(landscape, list):
            sequences = [item[0] for item in landscape]
            fitnesses = [item[1] for item in landscape]
        else:
            raise ValueError("Landscape must be a FitnessLandscape object, dictionary, or list of tuples")
        
        # Convert sequences to appropriate format if needed
        if isinstance(sequences[0], str):
            if self.alphabet_size == 2:
                sequences = [BinarySequence(seq) for seq in sequences]
            else:
                sequences = [MultialleleSequence(seq, self.alphabet_size) for seq in sequences]
        
        self.sequences = sequences
        
        # Create graph of sequences
        graph = create_hamming_graph(sequences)
        
        # Compute adjacency matrix
        adjacency = nx.adjacency_matrix(graph).astype(float)
        
        # Compute degree matrix
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        D_inv = np.diag(1.0 / np.sqrt(degrees))
        
        # Normalize adjacency matrix to create Markov transition matrix
        P = D_inv @ adjacency @ D_inv
        
        # Compute eigenvectors and eigenvalues
        n_components = min(self.n_components + 1, P.shape[0])
        eigenvalues, eigenvectors = eigsh(P, k=n_components, which='LM')
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Scale eigenvectors by eigenvalues raised to power t
        scaled_eigenvectors = eigenvectors[:, 1:] * (eigenvalues[1:] ** t)
        
        # Store results
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        # Create dictionary mapping sequences to diffusion coordinates
        self.diffusion_coordinates = {
            str(seq): scaled_eigenvectors[i, :self.n_components]
            for i, seq in enumerate(sequences)
        }
        
        self._is_fitted = True
        return self.diffusion_coordinates
    
    def transform(self, sequences: List[Union[str, Sequence]]) -> np.ndarray:
        """
        Project new sequences onto existing diffusion axes.
        
        Args:
            sequences: List of sequences to project
            
        Returns:
            np.ndarray: Array of diffusion coordinates for the sequences
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before transforming new sequences")
        
        # Convert sequences to strings if they are not already
        seq_strs = [str(seq) for seq in sequences]
        
        # Check if sequences are in the original set
        known_coords = np.array([self.diffusion_coordinates.get(seq, np.nan) for seq in seq_strs])
        unknown_indices = np.isnan(known_coords[:, 0])
        
        if np.all(~unknown_indices):
            return known_coords
        
        # For sequences not in the original set, compute Nystrom extension
        unknown_seqs = [sequences[i] for i in range(len(sequences)) if unknown_indices[i]]
        
        # Compute distances to original sequences
        distances = np.zeros((len(unknown_seqs), len(self.sequences)))
        for i, seq1 in enumerate(unknown_seqs):
            for j, seq2 in enumerate(self.sequences):
                if isinstance(seq1, str):
                    if self.alphabet_size == 2:
                        seq1 = BinarySequence(seq1)
                    else:
                        seq1 = MultialleleSequence(seq1, self.alphabet_size)
                distances[i, j] = seq1.distance(seq2)
        
        # Convert distances to similarities
        sigma = np.median(distances)
        similarities = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalize similarities
        row_sums = similarities.sum(axis=1)
        normalized_similarities = similarities / row_sums[:, np.newaxis]
        
        # Project onto eigenvectors
        projections = normalized_similarities @ self.eigenvectors[:, 1:self.n_components+1]
        
        # Create result array
        result = np.zeros((len(sequences), self.n_components))
        result[~unknown_indices] = known_coords[~unknown_indices]
        result[unknown_indices] = projections
        
        return result
    
    def plot_landscape(self, ax=None, colormap='viridis', 
                      dimensions: List[int] = [0, 1], 
                      fitness_values: Optional[Dict] = None,
                      title: Optional[str] = None,
                      show_colorbar: bool = True) -> plt.Axes:
        """
        Create 2D or 3D visualization of the landscape.
        
        Args:
            ax: Matplotlib axes to plot on (default: None, creates new axes)
            colormap: Colormap to use for fitness values (default: 'viridis')
            dimensions: Which diffusion dimensions to plot (default: [0, 1])
            fitness_values: Dictionary mapping sequences to fitness values (default: None)
            title: Title for the plot (default: None)
            show_colorbar: Whether to show a colorbar (default: True)
            
        Returns:
            plt.Axes: The matplotlib axes with the plot
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Get coordinates and sequences
        coords = np.array(list(self.diffusion_coordinates.values()))
        seqs = list(self.diffusion_coordinates.keys())
        
        # Create figure if needed
        if ax is None:
            if len(dimensions) == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points
        if len(dimensions) == 3:
            # 3D plot
            x = coords[:, dimensions[0]]
            y = coords[:, dimensions[1]]
            z = coords[:, dimensions[2]]
            
            if fitness_values is not None:
                c = np.array([fitness_values.get(seq, 0) for seq in seqs])
                scatter = ax.scatter(x, y, z, c=c, cmap=colormap, s=50, alpha=0.8)
                if show_colorbar:
                    plt.colorbar(scatter, ax=ax, label='Fitness')
            else:
                ax.scatter(x, y, z, s=50, alpha=0.8)
            
            ax.set_xlabel(f'Diffusion Axis {dimensions[0]+1}')
            ax.set_ylabel(f'Diffusion Axis {dimensions[1]+1}')
            ax.set_zlabel(f'Diffusion Axis {dimensions[2]+1}')
        else:
            # 2D plot
            x = coords[:, dimensions[0]]
            y = coords[:, dimensions[1]]
            
            if fitness_values is not None:
                c = np.array([fitness_values.get(seq, 0) for seq in seqs])
                scatter = ax.scatter(x, y, c=c, cmap=colormap, s=50, alpha=0.8)
                if show_colorbar:
                    plt.colorbar(scatter, ax=ax, label='Fitness')
            else:
                ax.scatter(x, y, s=50, alpha=0.8)
            
            ax.set_xlabel(f'Diffusion Axis {dimensions[0]+1}')
            ax.set_ylabel(f'Diffusion Axis {dimensions[1]+1}')
        
        if title:
            ax.set_title(title)
        
        return ax
    
    def plot_paths(self, paths: List[List[Union[str, Sequence]]], 
                  ax=None, colors=None, labels=None,
                  dimensions: List[int] = [0, 1]) -> plt.Axes:
        """
        Visualize evolutionary paths on the diffusion map.
        
        Args:
            paths: List of paths, where each path is a list of sequences
            ax: Matplotlib axes to plot on (default: None, creates new axes)
            colors: List of colors for the paths (default: None, uses default colors)
            labels: List of labels for the paths (default: None)
            dimensions: Which diffusion dimensions to plot (default: [0, 1])
            
        Returns:
            plt.Axes: The matplotlib axes with the plot
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before plotting paths")
        
        # Create figure if needed
        if ax is None:
            if len(dimensions) == 3:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set default colors if not provided
        if colors is None:
            colors = plt.cm.tab10.colors
        
        # Plot each path
        for i, path in enumerate(paths):
            # Convert sequences to strings if they are not already
            path_strs = [str(seq) for seq in path]
            
            # Get coordinates for the path
            path_coords = self.transform(path_strs)
            
            # Plot the path
            color = colors[i % len(colors)]
            label = labels[i] if labels is not None and i < len(labels) else f'Path {i+1}'
            
            if len(dimensions) == 3:
                # 3D plot
                x = path_coords[:, dimensions[0]]
                y = path_coords[:, dimensions[1]]
                z = path_coords[:, dimensions[2]]
                ax.plot(x, y, z, '-o', color=color, label=label, linewidth=2, markersize=8)
            else:
                # 2D plot
                x = path_coords[:, dimensions[0]]
                y = path_coords[:, dimensions[1]]
                ax.plot(x, y, '-o', color=color, label=label, linewidth=2, markersize=8)
        
        # Add legend if there are labels
        if labels is not None:
            ax.legend()
        
        return ax
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Get the eigenvalues of the diffusion operator.
        
        Returns:
            np.ndarray: Array of eigenvalues
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing eigenvalues")
        
        return self.eigenvalues
    
    def get_eigenvectors(self) -> np.ndarray:
        """
        Get the eigenvectors of the diffusion operator.
        
        Returns:
            np.ndarray: Array of eigenvectors
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing eigenvectors")
        
        return self.eigenvectors
