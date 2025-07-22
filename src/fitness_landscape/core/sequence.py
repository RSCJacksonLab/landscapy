from __future__ import annotations
from typing import Iterable, List, Sequence as _SeqLike, Union, Literal

import numpy as np
from cogent3.core.sequence import Sequence as _C3Sequence
from cogent3.core.moltype import MolType
from cogent3 import get_moltype

# Helper utilities
_SeqConvertible = Union["BaseNumpySequence", _SeqLike[int], np.ndarray, _C3Sequence]

def _to_numpy(x: _SeqConvertible) -> np.ndarray:
    """
    Helper function to return a 1d np array view of x without copying
    to a new array. 

    Parameters
    ----------
    x : BaseNumpySequence, sequence-like, np.ndarray, cogent3.Sequence
        The sequence. 
    
    Returns
    -------
    np.ndarray
        The sequence array    
    """
    if isinstance(x, BaseNumpySequence):
        return x._np
    
    # Handle cogent3 Sequence objects
    if isinstance(x, _C3Sequence):
        return np.array(list(str(x)))
    return np.asarray(x).ravel()

class BaseNumpySequence:
    """
    Base class for sequences represented as numpy arrays and
    interfacing with cogent3 sequences.

    Attributes
    ----------
    sequence : _SeqConvertible
        The sequence data as a _SeqConvertible type.
    alphabet : Iterable, default None
        The alphabet of the sequence, if applicable.
    moltype : MolType, default None
        The moltype of the sequence, if applicable.     
    """
    def __init__(self,
                 sequence: _SeqConvertible,
                 *,
                 alphabet: Union[Iterable, None] = None,
                 moltype: Union[str, MolType, None] = None) -> None:
        
        # If providing sequence as a cogent3.Sequence, handle it directly and update attributes.
        if isinstance(sequence, _C3Sequence) and not isinstance(sequence, BaseNumpySequence):
            self._np = _to_numpy(sequence)
            self._c3_seq = sequence
            self.alphabet = list(self._c3_seq.moltype.alphabet)
            return

        self._np: np.ndarray = _to_numpy(sequence)
        self._c3_seq = None  # Default to None

        if alphabet is None:
            alphabet = sorted(list(set(self._np)))
        self.alphabet: List[str] = list(map(str, alphabet))

        # Only try to create a cogent3 sequence if a moltype is given.
        if moltype is not None:
            try:
                if isinstance(moltype, str):
                    moltype_obj = get_moltype(moltype)
                else:
                    moltype_obj = moltype
                
                seq_str = "".join(map(str, self._np.tolist()))
                # Use the factory method from the moltype object
                self._c3_seq = moltype_obj.make_seq(seq_str)
            except (ValueError, TypeError, KeyError):
                # If moltype is not recognized or fails fall back and do not create the cogent3 object.
                self._c3_seq = None


    def __len__(self):
        return len(self._np)

    def __eq__(self, other):
        if not isinstance(other, BaseNumpySequence):
            return NotImplemented
        return np.array_equal(self._np, other._np)

    def __hash__(self):
        return hash(tuple(self._np))

    def __repr__(self):
        return f"{self.__class__.__name__}({self._np.tolist()})"

    @property
    def sequence(self) -> np.ndarray:
        return self._np

    @property
    def ndarray(self) -> np.ndarray:
        return self._np

    def to_array(self) -> np.ndarray:
        return self._np.copy()

    def distance(self, other: _SeqConvertible, *, metric: Literal["hamming", "euclidean"] = "hamming") -> float:
        other_arr = _to_numpy(other)
        if other_arr.shape != self._np.shape:
            raise ValueError("Sequences must be the same length")
        if metric == "hamming":
            return float(np.sum(self._np.astype(str) != other_arr.astype(str)))
        elif metric == "euclidean":
            try:
                return float(np.linalg.norm(self._np.astype(float) - other_arr.astype(float)))
            except ValueError as e:
                raise ValueError("Euclidean metric requires numeric sequence values") from e
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def mutate(self,
            positions: Union[int, Iterable[int], None] = None,
            *,
            values: Union[Iterable, None] = None,
            rng: Union[np.random.Generator, None] = None) -> "BaseNumpySequence":
        """
        Create a mutated copy of the sequence.
        """
        rng = rng or np.random.default_rng()
        new_np = self._np.copy()

        if positions is None:
            positions = [rng.integers(0, len(self))]
        elif isinstance(positions, int):
            positions = [positions]
        else:
            positions = list(positions)

        if values is None:
            # Ensure mutated value is a string to match alphabet type
            values = [rng.choice([a for a in self.alphabet if a != str(new_np[p])]) for p in positions]
        elif not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]

        if len(values) != len(positions):
            raise ValueError("Length of values must equal length of positions.")

        for p, v in zip(positions, values):
            new_np[p] = v

        # Get the moltype from the internal cogent3 sequence object
        current_moltype = self._c3_seq.moltype if self._c3_seq else None
        return self.__class__(new_np, alphabet=self.alphabet, moltype=current_moltype)

# BinarySequence can now be much simpler
class BinarySequence(BaseNumpySequence):
    """
    Binary {0,1} sequence.
    """
    def __init__(self,
                 sequence: _SeqConvertible) -> None:
        arr = _to_numpy(sequence).astype(int)
        if not set(arr).issubset({0, 1}):
            raise ValueError("BinarySequence accepts only 0/1 symbols")
        
        super().__init__(arr, alphabet=['0', '1'], moltype=None)

class MultialleleSequence(BaseNumpySequence):
    """
    A sequence with multiple alleles at each position.
    """

    def __init__(self,
                 sequence: _SeqConvertible,
                 alphabet: Iterable):
        arr = _to_numpy(sequence)
        if not set(arr).issubset(set(alphabet)):
            raise ValueError("MultialleleSequence accepts only symbols from the alphabet")
        super().__init__(arr,
                         alphabet=alphabet,
                         moltype=None)

class SoftSequence(BaseNumpySequence):
    """
    A posterior-probability “soft” sequence.

    Parameters
    ----------
    posterior : np.ndarray
        Shape (L, A). Rows are sites, columns are alphabet symbols.
    alphabet : Iterable
        The ordered alphabet corresponding to columns of `posterior`.
    hard_rule : {'argmax', 'sample'}, default 'argmax'
        How to derive the proxy hard sequence used by existing code.
    rng : np.random.Generator, optional
        RNG used when `hard_rule='sample'`.
    """
    def __init__(self,
                 aa_posterior: np.ndarray,
                 *,
                 alphabet: Iterable,
                 hard_rule: str = "argmax",
                 gap_posterior: np.ndarray = None,
                 rng: np.random.Generator | None = None):

        if gap_posterior is not None:

            if aa_posterior.shape[0] != gap_posterior.shape[0]:
                raise ValueError('gap and amino acid length must be the same')
            
            self.posterior = np.asarray(self.compute_conditional_gap_dist(aa_post_dist=aa_posterior,
                                                          gap_post_dist=gap_posterior))
        else:
            self.posterior = np.asarray(aa_posterior, dtype=float)
                
        self.alphabet = list(alphabet)
        self._rng = rng or np.random.default_rng()

        if hard_rule == "argmax":
            hard = self.posterior.argmax(axis=1)
        elif hard_rule == "sample":
            hard = [self._rng.choice(len(self.alphabet), p=row) for row in self.posterior]
        else:
            raise ValueError("hard_rule must be 'argmax' or 'sample'")

        super().__init__([self.alphabet[i] for i in hard], alphabet=self.alphabet)

    def map_values(self) -> np.ndarray:
        """
        
        """
        return self.posterior.max(axis=1)

    def entropy(self) -> np.ndarray:
        """
        
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(self.posterior > 0, np.log(self.posterior), 0.0)
        return -np.sum(self.posterior * logp, axis=1)

    def resample(self) -> "SoftSequence":
        """
        Generate a new *hard* proxy by sampling each posterior row.
        """
        return SoftSequence(self.posterior,
                            self.alphabet,
                            hard_rule="sample",
                            rng=self._rng)

    @staticmethod
    def compute_conditional_gap_dist(
        aa_post_dist: np.ndarray,       # (L, 20)
        gap_post_dist: np.ndarray,      # (L, 2)  gap / no‑gap
    ) -> np.ndarray:
        """

        """
        if aa_post_dist.shape[0] != gap_post_dist.shape[0]:
            raise ValueError("aa_post_dist and gap_post_dist must share length L")

        d = gap_post_dist[:, 0:1]
        cond = np.empty((aa_post_dist.shape[0], 21), dtype=float)
        cond[:, :20] = aa_post_dist * (1.0 - d)
        cond[:, 20]  = d[:, 0]
        return cond


def make_sequence(sequence: _SeqConvertible,
                  *,
                  binary: bool | None = None,
                  alphabet: Iterable | None = None,
                  moltype: str | None = None) -> BaseNumpySequence:
    """
    
    """
    seq_np = _to_numpy(sequence)
    if binary is True or (binary is None and set(seq_np).issubset({0, 1})):
        return BinarySequence(seq_np)
    
    return BaseNumpySequence(seq_np, alphabet=alphabet, moltype=moltype)


def sequence_distance(seq1: Union[BaseNumpySequence, np.ndarray],
                      seq2: Union[BaseNumpySequence, np.ndarray],
                      metric: Literal['hamming', 'euclidean'] = 'hamming') -> Union[float, int]:
    """
    Function to calculate distance between sequences.
    
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
    if isinstance(seq1, BaseNumpySequence):
        return seq1.distance(seq2, metric=metric)
    elif isinstance(seq2, BaseNumpySequence):
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

def generate_sequences(length: int,
                       alphabet: List) -> List[BaseNumpySequence]:
    """
    Function to generate all combinatorial sequences of a set length
    with a given alphabet of characters. 

    Parameters
    ----------
    length : int
        The length of combinatorial sequences to produce. 

    alphabet : List
        The list of characters in the alphabet. 
    
    Returns
    -------
    sequences : List
        List of `BaseNumpySequence` objects.
    """
    if length == 0:
        return []
    if length == 1:
        return [BaseNumpySequence([s]) for s in alphabet]
    sequences = []
    for s in alphabet:
        for sub_sequence in generate_sequences(length - 1, alphabet):
            sequences.append(BaseNumpySequence([s] + sub_sequence.to_array().tolist()))
    return sequences

# TODO: Functional code to convert BaseNumpySequence class to Binary or multiallelic.