"""
Minimum Epistasis Interpolation for fitness landscapes.

This module implements the method described in:
McCandlish, D. M. (2020). Inferring fitness landscapes by regression produces biased estimates of epistasis.
Proceedings of the National Academy of Sciences, 117(20), 10881-10889.

The method infers the least epistatic possible sequence-function relationship compatible
with available data by minimizing the expected squared epistatic coefficient for random
pairs of mutations across genetic backgrounds.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from ..core.landscape import FitnessLandscape
from ..core.sequence import Sequence, BinarySequence, MultialleleSequence
from ..transforms.walsh_hadamard import walsh_transform, inverse_walsh_transform


class MinimumEpistasisInterpolation:
    """
    Minimum Epistasis Interpolation for fitness landscapes.
    
    This class implements the method described by McCandlish for inferring the least
    epistatic possible sequence-function relationship compatible with available data.
    
    Attributes:
        sequence_length (int): Length of sequences in the landscape
        alphabet_size (int): Size of the alphabet for each position
        epistatic_coefficients (Dict): Dictionary of epistatic coefficients
        observed_sequences (List): List of observed sequences
        observed_fitnesses (np.ndarray): Array of observed fitness values
    """
    
    def __init__(self, sequence_length: int, alphabet_size: int = 2):
        """
        Initialize the MinimumEpistasisInterpolation model.
        
        Args:
            sequence_length: Length of sequences in the landscape
            alphabet_size: Size of the alphabet for each position (default: 2 for binary sequences)
        """
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.epistatic_coefficients = {}
        self.observed_sequences = []
        self.observed_fitnesses = None
        self._is_fitted = False
        
    def fit(self, landscape: Union[FitnessLandscape, Dict, List[Tuple]]) -> 'MinimumEpistasisInterpolation':
        """
        Fit the model to observed fitness landscape data.
        
        Args:
            landscape: A FitnessLandscape object, dictionary mapping sequences to fitness values,
                      or a list of (sequence, fitness) tuples
            
        Returns:
            self: The fitted model
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
        
        # Store observed data
        self.observed_sequences = [str(seq) for seq in sequences]
        self.observed_fitnesses = np.array(fitnesses)
        
        # Normalize fitness values
        normalized_fitnesses = (self.observed_fitnesses - np.mean(self.observed_fitnesses)) / np.std(self.observed_fitnesses)
        
        # Create binary representation matrix for observed sequences
        n_samples = len(sequences)
        n_coeffs = 2 ** self.sequence_length
        
        # Use sparse matrix for efficiency
        rows = []
        cols = []
        data = []
        
        for i, seq in enumerate(self.observed_sequences):
            # For each sequence, determine which Walsh functions it activates
            for j in range(n_coeffs):
                # Convert j to binary representation
                pattern = format(j, f'0{self.sequence_length}b')
                
                # Check if pattern matches sequence
                match = True
                for k in range(self.sequence_length):
                    if pattern[k] == '1' and seq[k] == '0':
                        match = False
                        break
                
                if match:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)
        
        X = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_coeffs))
        
        # Compute minimum epistasis solution
        # This is the solution to the linear system X * coeffs = fitnesses
        # with the constraint that the sum of squared epistatic coefficients is minimized
        
        # Use the pseudoinverse approach for minimum norm solution
        XtX = X.T @ X
        Xty = X.T @ normalized_fitnesses
        
        # Solve the system using sparse linear algebra
        coeffs = sparse_linalg.lsqr(XtX, Xty)[0]
        
        # Store epistatic coefficients
        self.epistatic_coefficients = {
            format(i, f'0{self.sequence_length}b'): coeffs[i]
            for i in range(len(coeffs))
        }
        
        self._is_fitted = True
        return self
    
    def predict(self, sequences: List[Union[str, Sequence]]) -> np.ndarray:
        """
        Predict fitness values for new sequences.
        
        Args:
            sequences: List of sequences to predict
            
        Returns:
            np.ndarray: Array of predicted fitness values
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert sequences to strings if they are not already
        seq_strs = [str(seq) for seq in sequences]
        
        # Initialize predictions
        predictions = np.zeros(len(seq_strs))
        
        # For each sequence, sum the contributions of all matching patterns
        for i, seq in enumerate(seq_strs):
            for pattern, coeff in self.epistatic_coefficients.items():
                # Check if pattern matches sequence
                match = True
                for j in range(self.sequence_length):
                    if pattern[j] == '1' and seq[j] == '0':
                        match = False
                        break
                
                if match:
                    predictions[i] += coeff
        
        # Rescale predictions to match the scale of observed fitnesses
        if self.observed_fitnesses is not None:
            mean_obs = np.mean(self.observed_fitnesses)
            std_obs = np.std(self.observed_fitnesses)
            predictions = predictions * std_obs + mean_obs
        
        return predictions
    
    def get_epistatic_coefficients(self) -> Dict[str, float]:
        """
        Get the epistatic coefficients.
        
        Returns:
            Dict[str, float]: Dictionary mapping binary patterns to coefficients
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing epistatic coefficients")
        
        return self.epistatic_coefficients
    
    def get_epistasis_statistics(self) -> Dict:
        """
        Get statistics about the epistatic coefficients.
        
        Returns:
            Dict: Dictionary containing statistics about epistasis
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before accessing epistasis statistics")
        
        # Extract coefficients
        coeffs = np.array(list(self.epistatic_coefficients.values()))
        patterns = list(self.epistatic_coefficients.keys())
        
        # Calculate basic statistics
        stats = {
            'mean': np.mean(np.abs(coeffs)),
            'std': np.std(np.abs(coeffs)),
            'min': np.min(np.abs(coeffs)),
            'max': np.max(np.abs(coeffs)),
            'by_order': {}
        }
        
        # Group by order (number of 1s in the binary pattern)
        for pattern, coeff in zip(patterns, coeffs):
            order = bin(int(pattern, 2)).count('1')
            if order not in stats['by_order']:
                stats['by_order'][order] = {
                    'coeffs': [],
                    'mean': 0,
                    'std': 0,
                    'count': 0
                }
            stats['by_order'][order]['coeffs'].append(abs(coeff))
        
        # Calculate statistics for each order
        for order in stats['by_order']:
            coeffs = stats['by_order'][order]['coeffs']
            stats['by_order'][order]['mean'] = np.mean(coeffs)
            stats['by_order'][order]['std'] = np.std(coeffs)
            stats['by_order'][order]['count'] = len(coeffs)
            del stats['by_order'][order]['coeffs']  # Remove raw coefficients to keep output clean
        
        return stats
    
    def calculate_epistasis(self, sequence1: Union[str, Sequence], 
                           sequence2: Union[str, Sequence]) -> float:
        """
        Calculate epistasis between two sequences.
        
        Args:
            sequence1: First sequence
            sequence2: Second sequence
            
        Returns:
            float: Epistasis value
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before calculating epistasis")
        
        # Convert sequences to strings if they are not already
        seq1_str = str(sequence1)
        seq2_str = str(sequence2)
        
        # Ensure sequences have the correct length
        if len(seq1_str) != self.sequence_length or len(seq2_str) != self.sequence_length:
            raise ValueError(f"Sequences must have length {self.sequence_length}")
        
        # Calculate fitness of each sequence
        fitness1 = self.predict([seq1_str])[0]
        fitness2 = self.predict([seq2_str])[0]
        
        # Calculate fitness of the double mutant
        # Find positions where sequences differ
        diff_positions = [i for i in range(self.sequence_length) if seq1_str[i] != seq2_str[i]]
        
        if not diff_positions:
            return 0.0  # No differences, no epistasis
        
        # Create a sequence with mutations from both sequences
        double_mutant = list(seq1_str)
        for pos in diff_positions:
            double_mutant[pos] = seq2_str[pos]
        double_mutant = ''.join(double_mutant)
        
        # Calculate fitness of the double mutant
        fitness_double = self.predict([double_mutant])[0]
        
        # Calculate epistasis as deviation from additivity
        wild_type = ''.join(['0'] * self.sequence_length)
        fitness_wild = self.predict([wild_type])[0]
        
        epistasis = fitness_double - fitness1 - fitness2 + fitness_wild
        return epistasis
