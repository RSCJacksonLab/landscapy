"""
Sequence representations and operations for fitness landscape analysis.

This module provides classes and functions for representing and manipulating
biological sequences (DNA, RNA, protein) and abstract sequences.
"""

import numpy as np
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable


class Sequence:
    """
    Base class for sequence representations.
    
    Parameters
    ----------
    sequence : array-like
        Sequence data as a list, array, or other iterable.
    alphabet : list or None, optional
        Possible values at each position. If None, inferred from sequence.
    """
    
    def __init__(self, sequence, alphabet=None):
        self.sequence = np.asarray(sequence)
        
        if alphabet is None:
            # Infer alphabet from unique values in sequence
            self.alphabet = sorted(set(self.sequence.flatten()))
        else:
            self.alphabet = sorted(set(alphabet))
        
        self.length = len(self.sequence)
        self.alphabet_size = len(self.alphabet)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.sequence[idx]
    
    def __eq__(self, other):
        if isinstance(other, Sequence):
            return np.array_equal(self.sequence, other.sequence)
        return np.array_equal(self.sequence, np.asarray(other))
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.sequence})"
    
    def distance(self, other, metric='hamming'):
        """
        Calculate distance between this sequence and another.
        
        Parameters
        ----------
        other : Sequence or array-like
            Sequence to compare with.
        metric : str, optional
            Distance metric ('hamming', 'euclidean', etc.)
            
        Returns
        -------
        float
            Distance between sequences.
        """
        other_seq = other.sequence if isinstance(other, Sequence) else np.asarray(other)
        
        if metric == 'hamming':
            return np.sum(self.sequence != other_seq)
        elif metric == 'euclidean':
            return np.sqrt(np.sum((self.sequence - other_seq) ** 2))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
    
    def mutate(self, positions=None, values=None):
        """
        Create a mutated copy of the sequence.
        
        Parameters
        ----------
        positions : int or list, optional
            Position(s) to mutate. If None, a random position is chosen.
        values : any or list, optional
            Value(s) to set at the position(s). If None, random values from
            the alphabet are chosen.
            
        Returns
        -------
        Sequence
            Mutated sequence.
        """
        new_sequence = self.sequence.copy()
        
        if positions is None:
            # Choose a random position
            positions = [np.random.randint(0, self.length)]
        elif isinstance(positions, int):
            positions = [positions]
        
        if values is None:
            # Choose random values from alphabet
            values = [np.random.choice([v for v in self.alphabet if v != self.sequence[pos]]) 
                     for pos in positions]
        elif not isinstance(values, list):
            values = [values]
        
        for pos, val in zip(positions, values):
            new_sequence[pos] = val
        
        return self.__class__(new_sequence, self.alphabet)
    
    def to_array(self):
        """
        Convert sequence to numpy array.
        
        Returns
        -------
        numpy.ndarray
            Sequence as numpy array.
        """
        return self.sequence.copy()
    
    def to_one_hot(self):
        """
        Convert sequence to one-hot encoding.
        
        Returns
        -------
        numpy.ndarray
            One-hot encoded sequence.
        """
        # Create mapping from alphabet to indices
        alphabet_map = {val: idx for idx, val in enumerate(self.alphabet)}
        
        # Convert sequence to indices
        indices = np.array([alphabet_map[val] for val in self.sequence])
        
        # Create one-hot encoding
        one_hot = np.zeros((self.length, self.alphabet_size))
        one_hot[np.arange(self.length), indices] = 1
        
        return one_hot


class BinarySequence(Sequence):
    """
    Binary sequence representation with efficient bit storage.
    
    Parameters
    ----------
    sequence : array-like
        Binary sequence data (0s and 1s).
    """
    
    def __init__(self, sequence):
        super().__init__(sequence, alphabet=[0, 1])
        
        # Validate binary sequence
        if not set(self.sequence).issubset({0, 1}):
            raise ValueError("Binary sequence must contain only 0s and 1s")
        
        # Convert to bit array for efficient storage
        self._bit_array = np.packbits(self.sequence.astype(bool))
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice indexing
            return self.sequence[idx]
        else:
            # Handle integer indexing
            return self.sequence[idx]
    
    def hamming_distance(self, other):
        """
        Calculate Hamming distance using efficient bit operations.
        
        Parameters
        ----------
        other : BinarySequence or array-like
            Sequence to compare with.
            
        Returns
        -------
        int
            Hamming distance.
        """
        if isinstance(other, BinarySequence) and len(self) == len(other):
            # Use XOR and bit counting for efficiency
            xor_result = np.bitwise_xor(self._bit_array, other._bit_array)
            return np.sum(np.unpackbits(xor_result)[:len(self)])
        else:
            # Fall back to standard implementation
            return super().distance(other, metric='hamming')


class MultialleleSequence(Sequence):
    """
    Multiallelic sequence representation.
    
    Parameters
    ----------
    sequence : array-like
        Sequence data with multiple possible values at each position.
    alphabet : list or None, optional
        Possible values at each position. If None, inferred from sequence.
    """
    
    def __init__(self, sequence, alphabet=None):
        super().__init__(sequence, alphabet)
    
    def to_categorical(self):
        """
        Convert sequence to categorical encoding (integers).
        
        Returns
        -------
        numpy.ndarray
            Categorically encoded sequence.
        """
        # Create mapping from alphabet to indices
        alphabet_map = {val: idx for idx, val in enumerate(self.alphabet)}
        
        # Convert sequence to indices
        return np.array([alphabet_map[val] for val in self.sequence])


def generate_sequences(length, alphabet, strategy='complete', n=None, seed=None):
    """
    Generate sequences based on strategy.
    
    Parameters
    ----------
    length : int
        Length of sequences.
    alphabet : list or array-like
        Possible values at each position.
    strategy : str, optional
        Generation strategy:
        - 'complete': All possible sequences (combinatorial)
        - 'random': Random sampling of sequence space
        - 'mutational': Start from a random sequence and apply mutations
    n : int, optional
        Number of sequences to generate (for 'random' and 'mutational').
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    list
        Generated sequences.
    """
    if seed is not None:
        np.random.seed(seed)
    
    alphabet = list(alphabet)
    alphabet_size = len(alphabet)
    
    if strategy == 'complete':
        # Generate all possible combinations
        if alphabet_size ** length > 10**6 and n is None:
            raise ValueError(
                f"Generating all {alphabet_size}^{length} = {alphabet_size ** length} "
                f"sequences would be too memory-intensive. "
                f"Use strategy='random' with a specified n instead."
            )
        
        # Use numpy's meshgrid to generate all combinations
        grids = np.meshgrid(*[alphabet for _ in range(length)], indexing='ij')
        combinations = np.stack([grid.flatten() for grid in grids], axis=-1)
        
        # If n is specified, sample from the complete set
        if n is not None and n < len(combinations):
            indices = np.random.choice(len(combinations), size=n, replace=False)
            combinations = combinations[indices]
        
        return [Sequence(seq, alphabet) for seq in combinations]
    
    elif strategy == 'random':
        if n is None:
            raise ValueError("Number of sequences (n) must be specified for strategy='random'")
        
        # Generate random sequences
        sequences = []
        for _ in range(n):
            seq = np.random.choice(alphabet, size=length)
            sequences.append(Sequence(seq, alphabet))
        
        return sequences
    
    elif strategy == 'mutational':
        if n is None:
            raise ValueError("Number of sequences (n) must be specified for strategy='mutational'")
        
        # Start with a random sequence
        start_seq = Sequence(np.random.choice(alphabet, size=length), alphabet)
        sequences = [start_seq]
        
        # Generate sequences by mutation
        for _ in range(n - 1):
            # Choose a random sequence from the current set
            parent = np.random.choice(sequences)
            
            # Mutate at a random position
            mutant = parent.mutate()
            sequences.append(mutant)
        
        return sequences
    
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


def sequence_distance(seq1, seq2, metric='hamming'):
    """
    Calculate distance between sequences.
    
    Parameters
    ----------
    seq1, seq2 : Sequence or array-like
        Sequences to compare.
    metric : str, optional
        Distance metric ('hamming', 'euclidean', etc.)
        
    Returns
    -------
    float
        Distance between sequences.
    """
    if isinstance(seq1, Sequence):
        return seq1.distance(seq2, metric=metric)
    elif isinstance(seq2, Sequence):
        return seq2.distance(seq1, metric=metric)
    else:
        # Convert to numpy arrays
        array1 = np.asarray(seq1)
        array2 = np.asarray(seq2)
        
        if metric == 'hamming':
            return np.sum(array1 != array2)
        elif metric == 'euclidean':
            return np.sqrt(np.sum((array1 - array2) ** 2))
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
