import numpy as np
import torch
from typing import List, Union, Optional, Tuple, Dict, Any, Callable, Iterable, Literal
from ..core.landscape import FitnessLandscape
from ..core.sequence import BaseNumpySequence, BinarySequence, generate_sequences

def walsh_transform(landscape: FitnessLandscape,
                    order: int = None,
                    backend: Literal['numpy', 'torch']='numpy') -> np.ndarray:
    """
    Compute Walsh-Hadamard transform of a fitness landscape.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to transform.
    order : int, default=`None`
        Maximum order of coefficients to compute.
    backend : str, default=`numpy`
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

    for seq in sequences:
        if not isinstance(seq, BinarySequence):

            # The Walsh-Hadamard transform is only defined for binary sequences.
            raise TypeError(f"All sequences must be BinarySequence, but found {type(seq)}")
        if not np.all(np.isin(seq.to_array().astype(int), [0, 1])):
            
            raise TypeError("All sequences in landscape must be binary (contain only 0s and 1s)")

    # Get fitness values and sequence length
    fitness_values = landscape.get_signal()
    N = len(sequences[0])

    # Create sequence matrix where each row is a sequence
    sequence_matrix = np.array([seq.to_array() for seq in sequences])
    
    # Compute Walsh transform based on backend
    if backend == 'numpy':
        return _walsh_transform_numpy(sequence_matrix, fitness_values, order)
    elif backend == 'torch':
        return _walsh_transform_torch(sequence_matrix, fitness_values, order)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _walsh_transform_numpy(sequence_matrix: np.ndarray,
                           fitness_values: np.ndarray,
                           order: int) -> np.ndarray:
    """
    Helper function to compute the walsh trasnform using the numpy
    backend. 

    Parameters
    ----------
    sequene_matrix : np.ndarray
        The matrix array of sequences. 
    
    fitness_values : np.ndarray
        The fitnes values. 
    
    order : int, default=`None`
        The maximum ordr to compute interaction terms up to.
    """
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


def _walsh_transform_torch(sequence_matrix: Union[torch.Tensor, np.ndarray],
                           fitness_values: Union[torch.Tensor, np.ndarray],
                           order: int) -> torch.Tensor:
    """
    Helper function to compute the walsh trasnform using the torch
    backend. 

    Parameters
    ----------
    sequene_matrix : np.ndarray or torch.tensor
        The matrix array of sequences. 
    
    fitness_values : np.ndarray or torch.tensor
        The fitnes values. 
    
    order : int, default=`None`
        The maximum ordr to compute interaction terms up to.

    Returns
    -------
    coefficients : torch.tensor
        The WHT coefficients.
    """
    # Convert to PyTorch tensors

    if isinstance(sequence_matrix, np.ndarray):
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


def inverse_walsh_transform(coefficients: Union[torch.Tensor, np.ndarray],
                            sequences: List = None,
                            backend: Literal['numpy', 'torch']='numpy') -> Union[torch.tensor, np.ndarray]:
    """
    Compute inverse Walsh-Hadamard transform.
    
    Parameters
    ----------
    coefficients : array-like
        Walsh coefficients.
    sequences : array-like, default=`None`
        Sequences to compute fitness for. If None, compute for all
        possible sequences.
    backend : str, default=`numpy`
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


def _inverse_walsh_transform_numpy(coefficients: np.ndarray,
                                   sequences=None) -> np.ndarray:
    """
    Helper function to compute inverse welsh-transform using numpy
    backend. 

    Parameters
    ----------
    coeffiicents : np.ndarray
        The WHT coefficients. 
    
    sequences : array-like, default=`None`
        Sequences to compute fitness for. If None, compute for all

    Returns
    -------
    fitness_values : np.ndarray
        The reconstructed fitness array.
    """
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
        if isinstance(sequences[0], BaseNumpySequence):
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


def _inverse_walsh_transform_torch(coefficients: Union[torch.Tensor, np.ndarray],
                                   sequences=None) -> torch.Tensor:
    """
    Helper function to compute inverse welsh-transform using torch
    backend. 

    Parameters
    ----------
    coeffiicents : np.ndarray or torch.Tensor
        The WHT coefficients. 
    
    sequences : array-like, default=`None`
        Sequences to compute fitness for. If None, compute for all

    Returns
    -------
    fitness_values : np.ndarray
        The reconstructed fitness array.
    """
    # Convert to PyTorch tensor
    if isinstance(coefficients, np.ndarray):
        coefficients = torch.tensor(coefficients, dtype=torch.float32)

    if sequences is None:
        # Determine sequence length from coefficients
        seq_length = int(torch.log2(torch.tensor(len(coefficients))))

        # Generate all possible binary sequences
        sequences = generate_sequences(seq_length, [0, 1])
        # Convert to a single NumPy array before creating the tensor
        sequence_data = np.array([seq.to_array() for seq in sequences])
        sequences = torch.tensor(sequence_data, dtype=torch.float32)
    else:
        # Convert sequences to torch tensor if needed
        if isinstance(sequences[0], BaseNumpySequence):
            # The efficient way: create a NumPy array first
            sequence_data = np.array([seq.to_array() for seq in sequences])
            sequences = torch.tensor(sequence_data, dtype=torch.float32)
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


def walsh_coefficients(landscape: FitnessLandscape,
                       order: int = None,
                       backend: Literal['numpy', 'torch']='numpy') -> Dict:
    """
    Extract Walsh coefficients up to specified order.
    
    Parameters
    ----------
    landscape : FitnessLandscape
        Fitness landscape to analyze.

    order : int, default=`None`
        Maximum order of coefficients to compute.
    
    backend : str, default=`numpy`
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