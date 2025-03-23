"""
Sparsity Analysis Methods for fitness landscapes.

This module implements the methods described in:
Brookes, D. H., Aghazadeh, A., & Listgarten, J. (2022). On the sparsity of fitness functions 
and implications for learning. Proceedings of the National Academy of Sciences, 119(1).

The methods leverage the observation that empirical fitness functions display substantial 
sparsity when represented in terms of epistatic interactions, using Compressed Sensing theory 
to provide scaling laws for sample requirements.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import matplotlib.pyplot as plt

from ..core.landscape import FitnessLandscape
from ..core.sequence import Sequence, BinarySequence, MultialleleSequence
from ..transforms.walsh_hadamard import walsh_transform, inverse_walsh_transform


class SparsityAnalysis:
    """
    Sparsity Analysis Methods for fitness landscapes.
    
    This class implements the methods described by Listgarten et al. for analyzing
    the sparsity of fitness landscapes and its implications for learning.
    
    Attributes:
        sequence_length (int): Length of sequences in the landscape
        alphabet_size (int): Size of the alphabet for each position
        sparsity (float): Estimated sparsity of the fitness landscape
        walsh_coefficients (Dict): Dictionary of Walsh coefficients
        significant_coefficients (Dict): Dictionary of significant Walsh coefficients
    """
    
    def __init__(self, sequence_length: int, alphabet_size: int = 2):
        """
        Initialize the SparsityAnalysis model.
        
        Args:
            sequence_length: Length of sequences in the landscape
            alphabet_size: Size of the alphabet for each position (default: 2 for binary sequences)
        """
        self.sequence_length = sequence_length
        self.alphabet_size = alphabet_size
        self.sparsity = None
        self.walsh_coefficients = {}
        self.significant_coefficients = {}
        self._is_fitted = False
        
    def estimate_sparsity(self, landscape: Union[FitnessLandscape, Dict, List[Tuple]], 
                         threshold: float = 0.01) -> float:
        """
        Estimate the sparsity of a fitness landscape.
        
        Args:
            landscape: A FitnessLandscape object, dictionary mapping sequences to fitness values,
                      or a list of (sequence, fitness) tuples
            threshold: Threshold for considering a Walsh coefficient significant (default: 0.01)
            
        Returns:
            float: Estimated sparsity (fraction of significant Walsh coefficients)
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
        
        # Check if we have a complete landscape
        total_possible_sequences = self.alphabet_size ** self.sequence_length
        if len(sequences) < total_possible_sequences:
            # Incomplete landscape, use compressed sensing approach
            return self._estimate_sparsity_incomplete(sequences, fitnesses, threshold)
        else:
            # Complete landscape, use direct Walsh transform
            return self._estimate_sparsity_complete(sequences, fitnesses, threshold)
    
    def calculate_sample_complexity(self, sparsity: float = None, 
                                   sequence_length: int = None, 
                                   alphabet_size: int = None) -> int:
        """
        Calculate the number of samples needed to recover a sparse fitness landscape.
        
        Args:
            sparsity: Sparsity of the fitness landscape (default: None, uses estimated sparsity)
            sequence_length: Length of sequences (default: None, uses instance value)
            alphabet_size: Size of the alphabet (default: None, uses instance value)
            
        Returns:
            int: Estimated number of samples needed
        """
        if sparsity is None:
            if self.sparsity is None:
                raise ValueError("Sparsity must be provided or estimated first")
            sparsity = self.sparsity
        
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        if alphabet_size is None:
            alphabet_size = self.alphabet_size
        
        # Calculate total number of possible Walsh coefficients
        total_coefficients = alphabet_size ** sequence_length
        
        # Calculate number of significant coefficients
        significant_coeffs = int(sparsity * total_coefficients)
        
        # Apply compressed sensing theory: O(k log(n/k)) samples needed
        # where k is the number of significant coefficients and n is the total number
        if significant_coeffs == 0:
            return 1  # Completely non-epistatic landscape
        
        samples_needed = int(significant_coeffs * np.log(total_coefficients / significant_coeffs))
        
        # Ensure we don't exceed the total number of possible sequences
        return min(samples_needed, total_coefficients)
    
    def generate_gnk_model(self, interaction_structure: Union[List[List[int]], str] = 'adjacent',
                          random_seed: int = None) -> Dict[str, float]:
        """
        Generate a Generalized NK (GNK) model fitness landscape.
        
        Args:
            interaction_structure: Structure of interactions between positions
                                  Can be a list of lists specifying neighborhoods for each position,
                                  or a string: 'adjacent' or 'random' (default: 'adjacent')
            random_seed: Random seed for reproducibility (default: None)
            
        Returns:
            Dict[str, float]: Dictionary mapping sequences to fitness values
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Define neighborhoods based on interaction structure
        if isinstance(interaction_structure, str):
            if interaction_structure == 'adjacent':
                # Each position interacts with itself and adjacent positions
                neighborhoods = []
                for i in range(self.sequence_length):
                    neighborhood = [i]
                    if i > 0:
                        neighborhood.append(i-1)
                    if i < self.sequence_length - 1:
                        neighborhood.append(i+1)
                    neighborhoods.append(neighborhood)
            elif interaction_structure == 'random':
                # Each position interacts with itself and K-1 random other positions
                K = min(3, self.sequence_length)  # Default to 3 interactions
                neighborhoods = []
                for i in range(self.sequence_length):
                    other_positions = [j for j in range(self.sequence_length) if j != i]
                    selected = np.random.choice(other_positions, min(K-1, len(other_positions)), replace=False)
                    neighborhood = [i] + selected.tolist()
                    neighborhoods.append(neighborhood)
            else:
                raise ValueError("Interaction structure must be 'adjacent', 'random', or a list of neighborhoods")
        else:
            neighborhoods = interaction_structure
        
        # Generate random sub-functions for each neighborhood
        sub_functions = []
        for neighborhood in neighborhoods:
            k = len(neighborhood)
            n_configs = self.alphabet_size ** k
            sub_function = np.random.normal(0, 1, n_configs)
            sub_functions.append((neighborhood, sub_function))
        
        # Generate all possible sequences
        all_sequences = self._generate_all_possible_sequences()
        
        # Calculate fitness for each sequence
        fitness_values = {}
        for seq in all_sequences:
            seq_str = str(seq)
            fitness = 0.0
            
            # Convert sequence to array of integers
            if isinstance(seq, BinarySequence):
                seq_array = np.array([int(c) for c in seq_str])
            else:
                seq_array = np.array([int(c) for c in seq_str])
            
            # Sum contributions from each sub-function
            for neighborhood, sub_function in sub_functions:
                # Extract relevant positions from sequence
                sub_seq = seq_array[neighborhood]
                
                # Calculate index into sub-function table
                idx = 0
                for i, pos in enumerate(sub_seq):
                    idx += pos * (self.alphabet_size ** i)
                
                # Add contribution
                fitness += sub_function[idx]
            
            # Store fitness
            fitness_values[seq_str] = fitness
        
        return fitness_values
    
    def compare_empirical_vs_gnk(self, landscape: Union[FitnessLandscape, Dict, List[Tuple]],
                               gnk_model: Dict[str, float],
                               plot: bool = True) -> Dict:
        """
        Compare empirical landscape to GNK model.
        
        Args:
            landscape: A FitnessLandscape object, dictionary mapping sequences to fitness values,
                      or a list of (sequence, fitness) tuples
            gnk_model: Dictionary mapping sequences to fitness values from a GNK model
            plot: Whether to generate comparison plots (default: True)
            
        Returns:
            Dict: Dictionary containing comparison metrics
        """
        # Extract sequences and fitnesses from empirical landscape
        if isinstance(landscape, FitnessLandscape):
            emp_sequences = list(landscape.genotype_to_fitness.keys())
            emp_fitnesses = list(landscape.genotype_to_fitness.values())
        elif isinstance(landscape, dict):
            emp_sequences = list(landscape.keys())
            emp_fitnesses = list(landscape.values())
        elif isinstance(landscape, list):
            emp_sequences = [item[0] for item in landscape]
            emp_fitnesses = [item[1] for item in landscape]
        else:
            raise ValueError("Landscape must be a FitnessLandscape object, dictionary, or list of tuples")
        
        # Ensure sequences are in string format
        emp_sequences = [str(seq) for seq in emp_sequences]
        
        # Get common sequences
        common_sequences = set(emp_sequences).intersection(set(gnk_model.keys()))
        
        if not common_sequences:
            raise ValueError("No common sequences between empirical landscape and GNK model")
        
        # Extract fitness values for common sequences
        common_seqs = list(common_sequences)
        emp_fitness = np.array([dict(zip(emp_sequences, emp_fitnesses))[seq] for seq in common_seqs])
        gnk_fitness = np.array([gnk_model[seq] for seq in common_seqs])
        
        # Normalize fitness values
        emp_fitness = (emp_fitness - np.mean(emp_fitness)) / np.std(emp_fitness)
        gnk_fitness = (gnk_fitness - np.mean(gnk_fitness)) / np.std(gnk_fitness)
        
        # Calculate comparison metrics
        correlation = np.corrcoef(emp_fitness, gnk_fitness)[0, 1]
        mse = np.mean((emp_fitness - gnk_fitness) ** 2)
        
        # Estimate sparsity for both landscapes
        emp_sparsity = self.estimate_sparsity({seq: val for seq, val in zip(common_seqs, emp_fitness)})
        self.sparsity = None  # Reset to ensure we calculate for GNK model
        gnk_sparsity = self.estimate_sparsity({seq: gnk_model[seq] for seq in common_seqs})
        
        # Generate comparison plots if requested
        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Scatter plot of fitness values
            axes[0].scatter(emp_fitness, gnk_fitness, alpha=0.6)
            axes[0].set_xlabel('Empirical Fitness')
            axes[0].set_ylabel('GNK Model Fitness')
            axes[0].set_title(f'Fitness Correlation: {correlation:.3f}')
            
            # Distribution of Walsh coefficients
            emp_walsh = np.array(list(self.walsh_coefficients.values()))
            axes[1].hist(np.abs(emp_walsh), bins=30, alpha=0.6, label='Empirical')
            
            self.estimate_sparsity({seq: gnk_model[seq] for seq in common_seqs})
            gnk_walsh = np.array(list(self.walsh_coefficients.values()))
            axes[1].hist(np.abs(gnk_walsh), bins=30, alpha=0.6, label='GNK Model')
            
            axes[1].set_xlabel('Absolute Walsh Coefficient')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Walsh Coefficient Distribution')
            axes[1].legend()
            
            # Sparsity comparison
            axes[2].bar(['Empirical', 'GNK Model'], [emp_sparsity, gnk_sparsity])
            axes[2].set_ylabel('Sparsity')
            axes[2].set_title('Landscape Sparsity Comparison')
            
            plt.tight_layout()
        
        # Return comparison metrics
        return {
            'correlation': correlation,
            'mse': mse,
            'empirical_sparsity': emp_sparsity,
            'gnk_sparsity': gnk_sparsity,
            'sample_complexity_empirical': self.calculate_sample_complexity(emp_sparsity),
            'sample_complexity_gnk': self.calculate_sample_complexity(gnk_sparsity)
        }
    
    def _estimate_sparsity_complete(self, sequences: List[Sequence], 
                                   fitnesses: List[float], 
                                   threshold: float) -> float:
        """
        Estimate sparsity for a complete fitness landscape using direct Walsh transform.
        
        Args:
            sequences: List of sequences
            fitnesses: List of fitness values
            threshold: Threshold for considering a Walsh coefficient significant
            
        Returns:
            float: Estimated sparsity (fraction of significant Walsh coefficients)
        """
        # Sort sequences and fitnesses
        seq_fitness = [(str(seq), fit) for seq, fit in zip(sequences, fitnesses)]
        seq_fitness.sort(key=lambda x: x[0])
        
        sorted_seqs = [item[0] for item in seq_fitness]
        sorted_fits = np.array([item[1] for item in seq_fitness])
        
        # Normalize fitness values
        normalized_fits = (sorted_fits - np.mean(sorted_fits)) / np.std(sorted_fits)
        
        # Calculate Walsh coefficients
        walsh_coeffs = walsh_transform(normalized_fits)
        
        # Store Walsh coefficients
        self.walsh_coefficients = {
            format(i, f'0{self.sequence_length}b'): walsh_coeffs[i]
            for i in range(len(walsh_coeffs))
        }
        
        # Identify significant coefficients
        max_coeff = np.max(np.abs(walsh_coeffs))
        significant_mask = np.abs(walsh_coeffs) > threshold * max_coeff
        
        self.significant_coefficients = {
            format(i, f'0{self.sequence_length}b'): walsh_coeffs[i]
            for i in range(len(walsh_coeffs)) if significant_mask[i]
        }
        
        # Calculate sparsity
        self.sparsity = np.sum(significant_mask) / len(walsh_coeffs)
        self._is_fitted = True
        
        return self.sparsity
    
    def _estimate_sparsity_incomplete(self, sequences: List[Sequence], 
                                     fitnesses: List[float], 
                                     threshold: float) -> float:
        """
        Estimate sparsity for an incomplete fitness landscape using compressed sensing.
        
        Args:
            sequences: List of sequences
            fitnesses: List of fitness values
            threshold: Threshold for considering a Walsh coefficient significant
            
        Returns:
            float: Estimated sparsity (fraction of significant Walsh coefficients)
        """
        # Convert sequences to binary representation for Walsh transform
        binary_seqs = []
        for seq in sequences:
            if isinstance(seq, BinarySequence):
                binary_seqs.append(str(seq))
            elif isinstance(seq, MultialleleSequence):
                # For multiallelic sequences, we need a different approach
                # This is a simplified version that works for binary sequences
                binary_seqs.append(str(seq))
            else:
                binary_seqs.append(str(seq))
        
        # Normalize fitness values
        normalized_fits = np.array(fitnesses)
        normalized_fits = (normalized_fits - np.mean(normalized_fits)) / np.std(normalized_fits)
        
        # Create measurement matrix for compressed sensing
        n_samples = len(sequences)
        n_coeffs = self.alphabet_size ** self.sequence_length
        
        # Use sparse matrix for efficiency
        rows = []
        cols = []
        data = []
        
        for i, seq in enumerate(binary_seqs):
            # Convert sequence to index
            idx = int(seq, self.alphabet_size) if self.alphabet_size <= 10 else 0
            rows.append(i)
            cols.append(idx)
            data.append(1.0)
        
        measurement_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_coeffs))
        
        # Use L1 regularization to find sparse solution
        from sklearn.linear_model import Lasso
        
        # Determine alpha based on noise level (assuming some noise in measurements)
        alpha = 0.01  # Default regularization strength
        
        # Fit Lasso model
        lasso = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False)
        lasso.fit(measurement_matrix, normalized_fits)
        
        # Extract Walsh coefficients
        walsh_coeffs = lasso.coef_
        
        # Store Walsh coefficients
        self.walsh_coefficients = {
            format(i, f'0{self.sequence_length}b'): walsh_coeffs[i]
            for i in range(len(walsh_coeffs))
        }
        
        # Identify significant coefficients
        max_coeff = np.max(np.abs(walsh_coeffs))
        significant_mask = np.abs(walsh_coeffs) > threshold * max_coeff
        
        self.significant_coefficients = {
            format(i, f'0{self.sequence_length}b'): walsh_coeffs[i]
            for i in range(len(walsh_coeffs)) if significant_mask[i]
        }
        
        # Calculate sparsity
        self.sparsity = np.sum(significant_mask) / len(walsh_coeffs)
        self._is_fitted = True
        
        return self.sparsity
    
    def _generate_all_possible_sequences(self) -> List[Sequence]:
        """
        Generate all possible sequences for the given sequence length and alphabet size.
        
        Returns:
            List[Sequence]: List of all possible sequences
        """
        if self.alphabet_size == 2:
            # For binary sequences, use more efficient generation
            return [BinarySequence(format(i, f'0{self.sequence_length}b')) 
                   for i in range(2**self.sequence_length)]
        else:
            # For multiallelic sequences, use cartesian product
            import itertools
            alphabet = list(range(self.alphabet_size))
            return [MultialleleSequence(''.join(map(str, seq)), self.alphabet_size) 
                   for seq in itertools.product(alphabet, repeat=self.sequence_length)]
