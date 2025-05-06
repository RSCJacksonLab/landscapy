from __future__ import annotations
from typing import Iterable, List, Mapping, Sequence as _SeqLike, Union, Literal

import numpy as np
from cogent3.core.sequence import Sequence as _C3Sequence

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
        return x._np  # already ndarray view

    # cogent3 sequence objects behave like strings; convert to char list
    if isinstance(x, _C3Sequence):
        return np.fromiter(str(x), dtype="U1", count=-1)

    # Fall back.
    return np.asarray(x).ravel()

# Core Sequence class
class BaseNumpySequence(_C3Sequence):
    """
    Base class for sequence representations. Inherits from the
    `cogent3.Sequence` class. 
    
    Attributes
    ----------
    sequence : array-like
        Sequence data as a list, array, or other iterable.
    alphabet : list, default=`None`
        Possible value
    moltype : str, default=`None`
        The cogent3 moltype.
    """

    def __init__(self,
                 sequence: _SeqConvertible,
                 *,
                 alphabet: Union[Iterable, None] = None,
                 moltype: Union[str, None] = None) -> None:
        
        self._np: np.ndarray = _to_numpy(sequence).astype("U1")  # 1‑char strings

        if alphabet is None:
            alphabet = sorted({*self._np})
        self.alphabet: List[str] = list(alphabet)

        super().__init__("".join(self._np.tolist()), moltype=moltype) # Initialise Cogent3 Sequence class from string.

    # Legacy public attribute used by existing analyses
    # Provide a read‑only view named `sequence` so code that calls
    # `set(seq.sequence)` continues to work without changes.

    # TODO: update the analyses functions and remove legacy properties.
    
    @property
    def sequence(self) -> np.ndarray:
        """Legacy accessor – identical to :pyattr:`ndarray` (read‑only NumPy view)."""
        return self._np

    @property
    def ndarray(self) -> np.ndarray:
        """A **view** (read‑only) of the internal 1‑D NumPy array."""
        return self._np

    # Provide legacy alias used by existing codebase
    def to_array(self) -> np.ndarray:
        """Return ``ndarray.copy()`` for backward compatibility."""
        return self._np.copy()

    def distance(self,
                 other: _SeqConvertible,
                 *,
                 metric: Literal["hamming", "euclidean"] = "hamming") -> float:
        """
        Method to calculate distance between this sequence and another.
        
        Parameters
        ----------
        other : array-like
            Sequence to compare with.
        metric : str, optional
            Distance metric ('hamming', 'euclidean').
            
        Returns
        -------
        float
            Distance between sequences.
        """

        other_arr = _to_numpy(other)
        if other_arr.shape != self._np.shape:
            raise ValueError("Sequences must be the same length for distance calculation")
        
        #TODO: Add (soft) alignment

        if metric == "hamming":
            return float(np.sum(self._np != other_arr))
        elif metric == "euclidean":
            
            # Cast to numbers if possible, else fall back
            try:
                return float(np.linalg.norm(self._np.astype(float) - other_arr.astype(float)))
            except ValueError as e:

                raise ValueError("Euclidean metric requires numeric sequence values") from e
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")

    def mutate(self,
               positions: Union[int, Iterable[int], None] = None,
               *,
               values: Union[Iterable, None] = None,
               rng: Union[np.random.Generator, None] = None) -> "BaseNumpySequence":
        """
        Create a mutated copy of the sequence.
        
        Parameters
        ----------
        positions : int or list, optional
            Position(s) to mutate. If None, a random position is
            chosen.
        values : any or list, optional
            Value(s) to set at the position(s). If None, random values
            from the alphabet are chosen.
        rng : np.random.Generator
            The RNG if no values are provided.
            
        Returns
        -------
        BaseNumpySequence
            Mutated sequence.
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
            values = [rng.choice([a for a in self.alphabet if a != new_np[p]]) for p in positions]
        elif not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]

        if len(values) != len(positions):
            raise ValueError("Length of values must equal length of positions.")

        for p, v in zip(positions, values):
            new_np[p] = v

        return self.__class__(new_np, alphabet=self.alphabet, moltype=self.moltype)

    def to_one_hot(self,
                   mapping: Union[Mapping[str, int], None] = None) -> np.ndarray:
        """
        Method to compute a (L, |alphabet|) one hot matrix for the
        sequence.

        Parameters
        ----------
        mapping : Mapping
            The index : character mapping. 
        
        Returns
        -------
        mat : np.ndarray
            The (L, |alphabet|) ohe matrix.
        """

        if mapping is None:
            mapping = {sym: i for i, sym in enumerate(self.alphabet)}
        idxs = np.array([mapping[s] for s in self._np], dtype=int)
        mat = np.zeros((len(self), len(mapping)), dtype=int)
        mat[np.arange(len(self)), idxs] = 1
        return mat
    
    def to_integer(self,
                   mapping: Union[Mapping[str, int], None] = None) -> np.ndarray:
        """
        Method to convert interger representation of sequence. Useful
        in multiallelic systems. 

        Parameters
        ----------
        mapping : Mapping
            The index : character mapping. 
        
        Returns
        -------
        np.ndarray
            The (L, 1) integer array.        
        """
        if mapping is None:
            mapping = {sym: i for i, sym in enumerate(self.alphabet)}
        return np.array([mapping[s] for s in self.ndarray], dtype=int)
    
    def remove_gap_arr(self,
                       *,
                       gap_threshold: float = 0.5) -> np.ndarray:
        """

        """
        gap_idx = self.alphabet.index("gap") if "gap" in self.alphabet else len(self.alphabet) - 1
        post = self.to_one_hot() # (L, |A|)

        if post.shape[1] != len(self.alphabet):
            raise ValueError("posterior columns != alphabet length")

        keep_mask = post[:, gap_idx] <= gap_threshold
        filtered = post[keep_mask, :gap_idx]
        if gap_idx < post.shape[1] - 1:
            filtered = np.hstack([filtered, post[keep_mask, gap_idx + 1 :]])

        # renormalise row–wise
        filtered /= filtered.sum(axis=1, keepdims=True)
        return filtered

class BinarySequence(BaseNumpySequence):
    """
    Binary {0,1} sequence with bit‑wise Hamming distance optimisation.
    
    """

    def __init__(self, sequence: _SeqConvertible):
        arr = _to_numpy(sequence).astype(int)
        if not set(arr).issubset({0, 1}):
            raise ValueError("BinarySequence accepts only 0/1 symbols")
        super().__init__(arr, alphabet=[0, 1], moltype=None)
        self._packed = np.packbits(arr.astype(bool))

    # Override for fast bitwise Hamming
    def distance(self, other: _SeqConvertible, *, metric="hamming") -> float:  # type: ignore[override]
        if metric != "hamming":  # fall back to parent
            return super().distance(other, metric=metric)
        if isinstance(other, BinarySequence) and len(self) == len(other):
            xor = np.bitwise_xor(self._packed, other._packed)
            return int(np.sum(np.unpackbits(xor)[: len(self)]))
        return super().distance(other, metric="hamming")


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
    """
    if isinstance(seq1, BaseNumpySequence):
        return seq1.distance(seq2, metric=metric)
    elif isinstance(seq2, BaseNumpySequence):
        return seq2.distance(seq1, metric=metric)
    else:
        array1 = np.asarray(seq1)
        array2 = np.asarray(seq2)
        return _compute_distance(array1, array2, metric)
