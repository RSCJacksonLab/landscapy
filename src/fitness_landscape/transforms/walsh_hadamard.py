"""
Walsh-Hadamard transform implementations for fitness landscape analysis.

This module provides functions for computing Walsh-Hadamard transforms of fitness landscapes,
including extensions for multiallelic landscapes.
"""

import numpy as np
import torch
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable
from ..core.landscape import FitnessLandscape
from ..core.sequence import Sequence, BinarySequence


def walsh_transform(landscape, order=None, backend='numpy'):
    """
    Compute Walsh-Hadamard transform of a fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to transform.
    order : int or None, optional
        Maximum order of coefficients to compute.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    array-like
        Walsh coefficients.
    """
    # Validate input
    if not isinstance(landscape, FitnessLandscape):
        raise TypeError("landscape must be a FitnessLandscape object")
    
    # Check if all sequences have the same length and are binary
    sequences = landscape.sequences
    if not sequences:
        raise ValueError("Landscape contains no sequences")
    
    seq_length = len(sequences[0])
    for seq in sequences:
        if len(seq) != seq_length:
            raise ValueError("All sequences must have the same length")
        if not set(seq.sequence).issubset({0, 1}):
            raise ValueError("Walsh transform requires binary sequences (0s and 1s)")
    
    # Extract fitness values in the same order as sequences
    fitness_values = np.array([landscape.get_fitness(seq) for seq in sequences])
    
    # Create sequence matrix where each row is a sequence
    sequence_matrix = np.array([seq.to_array() for seq in sequences])
    
    # Compute Walsh transform based on backend
    if backend == 'numpy':
        return _walsh_transform_numpy(sequence_matrix, fitness_values, order)
    elif backend == 'torch':
        return _walsh_transform_torch(sequence_matrix, fitness_values, order)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _walsh_transform_numpy(sequence_matrix, fitness_values, order=None):
    """Compute Walsh transform using NumPy."""
    n_sequences, seq_length = sequence_matrix.shape
    
    # Compute all possible binary masks up to the specified order
    if order is None:
        order = seq_length
    
    # Initialize coefficient array
    coefficients = np.zeros(2**seq_length)
    
    # Compute Walsh coefficients
    for i in range(n_sequences):
        seq = sequence_matrix[i]
        fitness = fitness_values[i]
        
        # Compute all interaction terms for this sequence
        for mask in range(2**seq_length):
            # Convert mask to binary and check if it's within the order limit
            mask_bits = [(mask >> j) & 1 for j in range(seq_length)]
            if sum(mask_bits) <= order:
                # Compute Walsh basis function
                walsh_basis = 1
                for j in range(seq_length):
                    if mask_bits[j] == 1:
                        walsh_basis *= (-1)**seq[j]
                
                # Update coefficient
                coefficients[mask] += fitness * walsh_basis
    
    # Normalize coefficients
    coefficients /= n_sequences
    
    return coefficients


def _walsh_transform_torch(sequence_matrix, fitness_values, order=None):
    """Compute Walsh transform using PyTorch."""
    # Convert to PyTorch tensors
    sequence_tensor = torch.tensor(sequence_matrix, dtype=torch.float32)
    fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32)
    
    n_sequences, seq_length = sequence_tensor.shape
    
    # Compute all possible binary masks up to the specified order
    if order is None:
        order = seq_length
    
    # Initialize coefficient tensor
    coefficients = torch.zeros(2**seq_length)
    
    # Compute Walsh coefficients
    for i in range(n_sequences):
        seq = sequence_tensor[i]
        fitness = fitness_tensor[i]
        
        # Compute all interaction terms for this sequence
        for mask in range(2**seq_length):
            # Convert mask to binary and check if it's within the order limit
            mask_bits = [(mask >> j) & 1 for j in range(seq_length)]
            if sum(mask_bits) <= order:
                # Compute Walsh basis function
                walsh_basis = torch.tensor(1.0)
                for j in range(seq_length):
                    if mask_bits[j] == 1:
                        walsh_basis *= (-1)**seq[j].item()
                
                # Update coefficient
                coefficients[mask] += fitness * walsh_basis
    
    # Normalize coefficients
    coefficients /= n_sequences
    
    return coefficients


def inverse_walsh_transform(coefficients, sequences=None, backend='numpy'):
    """
    Compute inverse Walsh-Hadamard transform.
    
    Parameters
    ----------
    coefficients : array-like
        Walsh coefficients.
    sequences : array-like or None, optional
        Sequences to compute fitness for. If None, compute for all possible sequences.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    array-like
        Reconstructed fitness values.
    """
    if backend == 'numpy':
        return _inverse_walsh_transform_numpy(coefficients, sequences)
    elif backend == 'torch':
        return _inverse_walsh_transform_torch(coefficients, sequences)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _inverse_walsh_transform_numpy(coefficients, sequences=None):
    """Compute inverse Walsh transform using NumPy."""
    coefficients = np.asarray(coefficients)
    
    if sequences is None:
        # Determine sequence length from coefficients
        seq_length = int(np.log2(len(coefficients)))
        
        # Generate all possible binary sequences
        from ..core.sequence import generate_sequences
        sequences = generate_sequences(seq_length, [0, 1], strategy='complete')
        sequences = np.array([seq.to_array() for seq in sequences])
    else:
        # Convert sequences to numpy array if needed
        if isinstance(sequences[0], Sequence):
            sequences = np.array([seq.to_array() for seq in sequences])
        else:
            sequences = np.asarray(sequences)
    
    n_sequences, seq_length = sequences.shape
    
    # Initialize fitness values
    fitness_values = np.zeros(n_sequences)
    
    # Compute fitness values using inverse Walsh transform
    for i in range(n_sequences):
        seq = sequences[i]
        
        # Compute fitness as sum of coefficients * basis functions
        for mask in range(len(coefficients)):
            # Convert mask to binary
            mask_bits = [(mask >> j) & 1 for j in range(seq_length)]
            
            # Compute Walsh basis function
            walsh_basis = 1
            for j in range(seq_length):
                if mask_bits[j] == 1:
                    walsh_basis *= (-1)**seq[j]
            
            # Update fitness
            fitness_values[i] += coefficients[mask] * walsh_basis
    
    return fitness_values


def _inverse_walsh_transform_torch(coefficients, sequences=None):
    """Compute inverse Walsh transform using PyTorch."""
    # Convert to PyTorch tensor
    coefficients = torch.tensor(coefficients, dtype=torch.float32)
    
    if sequences is None:
        # Determine sequence length from coefficients
        seq_length = int(torch.log2(torch.tensor(len(coefficients))))
        
        # Generate all possible binary sequences
        from ..core.sequence import generate_sequences
        sequences = generate_sequences(seq_length, [0, 1], strategy='complete')
        sequences = torch.tensor([seq.to_array() for seq in sequences], dtype=torch.float32)
    else:
        # Convert sequences to torch tensor if needed
        if isinstance(sequences[0], Sequence):
            sequences = torch.tensor([seq.to_array() for seq in sequences], dtype=torch.float32)
        else:
            sequences = torch.tensor(sequences, dtype=torch.float32)
    
    n_sequences, seq_length = sequences.shape
    
    # Initialize fitness values
    fitness_values = torch.zeros(n_sequences)
    
    # Compute fitness values using inverse Walsh transform
    for i in range(n_sequences):
        seq = sequences[i]
        
        # Compute fitness as sum of coefficients * basis functions
        for mask in range(len(coefficients)):
            # Convert mask to binary
            mask_bits = [(mask >> j) & 1 for j in range(seq_length)]
            
            # Compute Walsh basis function
            walsh_basis = torch.tensor(1.0)
            for j in range(seq_length):
                if mask_bits[j] == 1:
                    walsh_basis *= (-1)**seq[j].item()
            
            # Update fitness
            fitness_values[i] += coefficients[mask] * walsh_basis
    
    return fitness_values


def walsh_coefficients(landscape, order=None, backend='numpy'):
    """
    Extract Walsh coefficients up to specified order.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.
    order : int or None, optional
        Maximum order of coefficients to compute.
    backend : str, optional
        Computational backend ('numpy', 'torch').
        
    Returns
    -------
    dict
        Dictionary mapping interaction terms to coefficients.
    """
    # Compute Walsh transform
    coefficients = walsh_transform(landscape, order=order, backend=backend)
    
    # Get sequence length
    seq_length = len(landscape.sequences[0])
    
    # Create dictionary mapping interaction terms to coefficients
    result = {}
    
    for mask in range(len(coefficients)):
        # Convert mask to binary
        mask_bits = [(mask >> j) & 1 for j in range(seq_length)]
        
        # Skip if order is exceeded
        if order is not None and sum(mask_bits) > order:
            continue
        
        # Create interaction term string
        if sum(mask_bits) == 0:
            term = "intercept"
        else:
            term = ",".join([str(j) for j in range(seq_length) if mask_bits[j] == 1])
        
        # Add to result
        result[term] = coefficients[mask]
    
    return result


class MultialleleWalshTransform:
    """
    Extended Walsh-Hadamard transform for multiallelic landscapes.
    
    Parameters
    ----------
    alphabet_sizes : list or array-like
        Number of possible values at each position.
    backend : str, optional
        Computational backend ('numpy', 'torch').
    """
    
    def __init__(self, alphabet_sizes, backend='numpy'):
        self.alphabet_sizes = np.asarray(alphabet_sizes)
        self.backend = backend
        self.n_positions = len(alphabet_sizes)
        
        # Precompute basis matrices for each position
        self.basis_matrices = []
        for size in alphabet_sizes:
            self.basis_matrices.append(self._create_basis_matrix(size))
    
    def _create_basis_matrix(self, size):
        """Create orthogonal basis matrix for a given alphabet size."""
        if self.backend == 'numpy':
            # Create Fourier basis matrix
            matrix = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    matrix[i, j] = np.cos(2 * np.pi * i * j / size)
            
            # Normalize columns
            for j in range(size):
                matrix[:, j] /= np.sqrt(np.sum(matrix[:, j] ** 2))
            
            return matrix
        
        elif self.backend == 'torch':
            # Create Fourier basis matrix
            matrix = torch.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    matrix[i, j] = torch.cos(2 * torch.pi * i * j / size)
            
            # Normalize columns
            for j in range(size):
                matrix[:, j] /= torch.sqrt(torch.sum(matrix[:, j] ** 2))
            
            return matrix
        
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def transform(self, landscape):
        """
        Compute transform for multiallelic landscape.
        
        Parameters
        ----------
        landscape : FitnessLandscape
            Fitness landscape to transform.
            
        Returns
        -------
        array-like
            Transform coefficients.
        """
        # Validate input
        if not isinstance(landscape, FitnessLandscape):
            raise TypeError("landscape must be a FitnessLandscape object")
        
        # Check if all sequences have the same length
        sequences = landscape.sequences
        if not sequences:
            raise ValueError("Landscape contains no sequences")
        
        seq_length = len(sequences[0])
        if seq_length != self.n_positions:
            raise ValueError(f"Sequence length ({seq_length}) does not match alphabet_sizes length ({self.n_positions})")
        
        # Extract fitness values and sequences
        fitness_values = np.array([landscape.get_fitness(seq) for seq in sequences])
        sequence_matrix = np.array([seq.to_array() for seq in sequences])
        
        # Compute transform based on backend
        if self.backend == 'numpy':
            return self._transform_numpy(sequence_matrix, fitness_values)
        elif self.backend == 'torch':
            return self._transform_torch(sequence_matrix, fitness_values)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _transform_numpy(self, sequence_matrix, fitness_values):
        """Compute multiallelic Walsh transform using NumPy."""
        n_sequences, seq_length = sequence_matrix.shape
        
        # Initialize coefficient array
        total_coeffs = np.prod(self.alphabet_sizes)
        coefficients = np.zeros(total_coeffs)
        
        # Compute coefficients
        for i in range(n_sequences):
            seq = sequence_matrix[i]
            fitness = fitness_values[i]
            
            # Compute basis function value for each position
            position_values = []
            for pos in range(seq_length):
                pos_value = self.basis_matrices[pos][seq[pos], :]
                position_values.append(pos_value)
            
            # Compute all interaction terms
            for idx in np.ndindex(*self.alphabet_sizes):
                # Compute product of basis functions
                basis_product = 1.0
                for pos, basis_idx in enumerate(idx):
                    basis_product *= position_values[pos][basis_idx]
                
                # Compute flat index
                flat_idx = np.ravel_multi_index(idx, self.alphabet_sizes)
                
                # Update coefficient
                coefficients[flat_idx] += fitness * basis_product
        
        # Normalize coefficients
        coefficients /= n_sequences
        
        return coefficients
    
    def _transform_torch(self, sequence_matrix, fitness_values):
        """Compute multiallelic Walsh transform using PyTorch."""
        # Convert to PyTorch tensors
        sequence_tensor = torch.tensor(sequence_matrix, dtype=torch.long)
        fitness_tensor = torch.tensor(fitness_values, dtype=torch.float32)
        
        n_sequences, seq_length = sequence_tensor.shape
        
        # Initialize coefficient array
        total_coeffs = np.prod(self.alphabet_sizes)
        coefficients = torch.zeros(total_coeffs)
        
        # Compute coefficients
        for i in range(n_sequences):
            seq = sequence_tensor[i]
            fitness = fitness_tensor[i]
            
            # Compute basis function value for each position
            position_values = []
            for pos in range(seq_length):
                pos_value = self.basis_matrices[pos][seq[pos], :]
                position_values.append(pos_value)
            
            # Compute all interaction terms
            for idx in np.ndindex(*self.alphabet_sizes):
                # Compute product of basis functions
                basis_product = torch.tensor(1.0)
                for pos, basis_idx in enumerate(idx):
                    basis_product *= position_values[pos][basis_idx]
                
                # Compute flat index
                flat_idx = np.ravel_multi_index(idx, self.alphabet_sizes)
                
                # Update coefficient
                coefficients[flat_idx] += fitness * basis_product
        
        # Normalize coefficients
        coefficients /= n_sequences
        
        return coefficients
    
    def inverse_transform(self, coefficients, sequences=None):
        """
        Compute inverse transform.
        
        Parameters
        ----------
        coefficients : array-like
            Transform coefficients.
        sequences : array-like or None, optional
            Sequences to compute fitness for. If None, compute for all possible sequences.
            
        Returns
        -------
        array-like
            Reconstructed fitness values.
        """
        if self.backend == 'numpy':
            return self._inverse_transform_numpy(coefficients, sequences)
        elif self.backend == 'torch':
            return self._inverse_transform_torch(coefficients, sequences)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _inverse_transform_numpy(self, coefficients, sequences=None):
        """Compute inverse multiallelic Walsh transform using NumPy."""
        coefficients = np.asarray(coefficients)
        
        if sequences is None:
            # Generate all possible sequences
            from itertools import product
            all_sequences = list(product(*[range(size) for size in self.alphabet_sizes]))
            sequences = np.array(all_sequences)
        else:
            # Convert sequences to numpy array if needed
            if isinstance(sequences[0], Sequence):
                sequences = np.array([seq.to_array() for seq in sequences])
            else:
                sequences = np.asarray(sequences)
        
        n_sequences, seq_length = sequences.shape
        
        # Initialize fitness values
        fitness_values = np.zeros(n_sequences)
        
        # Compute fitness values
        for i in range(n_sequences):
            seq = sequences[i]
            
            # Compute basis function value for each position
            position_values = []
            for pos in range(seq_length):
                pos_value = self.basis_matrices[pos][seq[pos], :]
                position_values.append(pos_value)
            
            # Compute fitness as sum of coefficients * basis functions
            for idx in np.ndindex(*self.alphabet_sizes):
                # Compute product of basis functions
                basis_product = 1.0
                for pos, basis_idx in enumerate(idx):
                    basis_product *= position_values[pos][basis_idx]
                
                # Compute flat index
                flat_idx = np.ravel_multi_index(idx, self.alphabet_sizes)
                
                # Update fitness
                fitness_values[i] += coefficients[flat_idx] * basis_product
        
        return fitness_values
    
    def _inverse_transform_torch(self, coefficients, sequences=None):
        """Compute inverse multiallelic Walsh transform using PyTorch."""
        # Convert to PyTorch tensor
        coefficients = torch.tensor(coefficients, dtype=torch.float32)
        
        if sequences is None:
            # Generate all possible sequences
            from itertools import product
            all_sequences = list(product(*[range(size) for size in self.alphabet_sizes]))
            sequences = torch.tensor(all_sequences, dtype=torch.long)
        else:
            # Convert sequences to torch tensor if needed
            if isinstance(sequences[0], Sequence):
                sequences = torch.tensor([seq.to_array() for seq in sequences], dtype=torch.long)
            else:
                sequences = torch.tensor(sequences, dtype=torch.long)
        
        n_sequences, seq_length = sequences.shape
        
        # Initialize fitness values
        fitness_values = torch.zeros(n_sequences)
        
        # Compute fitness values
        for i in range(n_sequences):
            seq = sequences[i]
            
            # Compute basis function value for each position
            position_values = []
            for pos in range(seq_length):
                pos_value = self.basis_matrices[pos][seq[pos], :]
                position_values.append(pos_value)
            
            # Compute fitness as sum of coefficients * basis functions
            for idx in np.ndindex(*self.alphabet_sizes):
                # Compute product of basis functions
                basis_product = torch.tensor(1.0)
                for pos, basis_idx in enumerate(idx):
                    basis_product *= position_values[pos][basis_idx]
                
                # Compute flat index
                flat_idx = np.ravel_multi_index(idx, self.alphabet_sizes)
                
                # Update fitness
                fitness_values[i] += coefficients[flat_idx] * basis_product
        
        return fitness_values
