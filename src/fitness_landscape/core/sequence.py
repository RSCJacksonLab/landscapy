from __future__ import annotations
from typing import TYPE_CHECKING, Any, Iterable, List, Sequence as _SeqLike, Union, Literal, Mapping
from .._const import PROT_20
from .._optional import require_optional
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from cogent3.core.moltype import MolType
    from cogent3.core.sequence import Sequence as _C3Sequence
else:
    MolType = Any
    _C3Sequence = Any

# Helper utilities
_SeqConvertible = Union["BaseNumpySequence", _SeqLike[int], np.ndarray, _C3Sequence]


def _require_positive_length(length: int) -> int:
    """Return a validated positive sequence length."""
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
        raise TypeError("length must be an integer")
    if length <= 0:
        raise ValueError("length must be positive")
    return int(length)


def _validated_alphabet(alphabet: Iterable, *, name: str = "alphabet") -> list:
    """Return a non-empty alphabet with no duplicate values."""
    values = list(alphabet)
    if not values:
        raise ValueError(f"{name} must not be empty")
    for index, value in enumerate(values):
        if any(value == previous for previous in values[:index]):
            raise ValueError(f"{name} values must be unique")
    return values


def _is_cogent3_sequence(value: object) -> bool:
    if not type(value).__module__.startswith("cogent3."):
        return False
    sequence_module = require_optional(
        "cogent3.core.sequence",
        extra="phylogeny",
        purpose="Cogent3 sequence interoperability",
    )
    return isinstance(value, sequence_module.Sequence)

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
    if _is_cogent3_sequence(x):
        return np.array(list(str(x)))
    # Ensure plain strings are split into characters rather than treated as scalars
    if isinstance(x, str):
        return np.array(list(x))
    return np.asarray(x).ravel()

class BaseNumpySequence:
    """Represent a biological or symbolic sequence as a NumPy array.

    Parameters
    ----------
    sequence : BaseNumpySequence, sequence of scalar, ndarray, or cogent3.Sequence
        Sequence values in positional order.
    sequence_id : str, optional
        Stable identifier for the sequence. If omitted, a representation of the
        sequence values is used.
    alphabet : iterable, optional
        Ordered set of permitted symbols. If omitted, infer it from ``sequence``.
    moltype : str or cogent3.core.moltype.MolType, optional
        Cogent3 molecular type used to construct an interoperable sequence.

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
                 sequence_id: str = None,
                 *,
                 alphabet: Union[Iterable, None] = None,
                 moltype: Union[str, MolType, None] = None) -> None:

        is_c3_seq = _is_cogent3_sequence(sequence) and not isinstance(sequence, BaseNumpySequence)

        self._np: np.ndarray = np.array(_to_numpy(sequence), copy=True)
        if self._np.size == 0:
            raise ValueError("sequence must not be empty")
        self._c3_seq = sequence if is_c3_seq else None

        # Correctly handle sequence ID from cogent3 objects
        if sequence_id is not None:
            self.id = sequence_id
            self._has_explicit_id = True
        elif is_c3_seq and hasattr(sequence, 'name'):
            self.id = sequence.name
            self._has_explicit_id = sequence.name is not None
        else:
            self.id = str(self._np)
            self._has_explicit_id = False
        
        # If an alphabet is explicitly provided, ALWAYS use it.
        if alphabet is not None:
            self._alphabet = tuple(map(str, _validated_alphabet(alphabet)))
        
        # If no alphabet is given, try to get it from the cogent3 object's moltype.
        elif is_c3_seq:
            self._alphabet = tuple(map(str, self._c3_seq.moltype.alphabet))
        
        # Infer the alphabet from the sequence data itself.
        else:
            self._alphabet = tuple(sorted(set(map(str, self._np))))

        # Standardize gap character ONLY for string arrays, preserving numeric arrays.
        # Ensure dtype can hold "gap" (not truncated to 'g').
        if self._np.dtype.kind in 'US' and "gap" in self.alphabet and "-" not in self.alphabet:
            try:
                # Upcast to a wider Unicode dtype if needed
                if self._np.dtype.itemsize < 12:  # 'U3' is 12 bytes
                    self._np = self._np.astype('U3')
            except Exception:
                # Fallback to object dtype for safety
                self._np = self._np.astype(object)
            self._np[self._np == "-"] = "gap"

        if moltype and not self._c3_seq:
            try:
                get_moltype = require_optional(
                    "cogent3",
                    extra="phylogeny",
                    purpose="MolType-backed sequence construction",
                ).get_moltype
                moltype_obj = get_moltype(moltype) if isinstance(moltype, str) else moltype

                # Translate "gap" back to "-" for cogent3 compatibility
                seq_str = "".join(map(str, self._np)).replace("gap", "-")
                try:
                    self._c3_seq = moltype_obj.make_seq(seq=seq_str)
                except TypeError:
                    self._c3_seq = moltype_obj.make_seq(data=seq_str)
            except (ValueError, TypeError, KeyError):
                self._c3_seq = None

        # Sequence objects are hashable and are used as landscape lookup keys.
        # Keep the owned storage immutable after all constructor normalization.
        self._np.setflags(write=False)

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
    def alphabet(self) -> list:
        """Return a defensive copy of the ordered sequence alphabet."""
        return list(self._alphabet)

    @property
    def ungapped_arr(self) -> np.ndarray:
        """Return one-hot probabilities with the gap column removed."""
        one_hot = np.asarray(self.to_one_hot(), dtype=np.float64)  # (L, |A|)

        if "gap" in self.alphabet or "-" in self.alphabet:
            gap_idx = self.alphabet.index("gap") if "gap" in self.alphabet else self.alphabet.index("-")
            keep = [i for i in range(one_hot.shape[1]) if i != gap_idx]
            one_hot = one_hot[:, keep]

        return one_hot
    
    @property
    def sequence(self) -> np.ndarray:
        """Return a read-only view of the sequence array."""
        view = self._np.view()
        view.setflags(write=False)
        return view

    @property
    def ndarray(self) -> np.ndarray:
        """Return a read-only view of the sequence array."""
        view = self._np.view()
        view.setflags(write=False)
        return view

    def to_array(self) -> np.ndarray:
        """Return a copy of the sequence array.

        Returns
        -------
        ndarray
            Independent one-dimensional sequence array.
        """
        return self._np.copy()

    def distance(self, other: _SeqConvertible, *, metric: Literal["hamming", "euclidean"] = "hamming") -> float:
        """Compute Hamming or Euclidean distance to another sequence.

        Parameters
        ----------
        other : BaseNumpySequence, sequence of scalar, ndarray, or cogent3.Sequence
            Sequence with the same length as this sequence.
        metric : {'hamming', 'euclidean'}, default='hamming'
            Distance definition. Hamming distance counts unequal sites; Euclidean
            distance is defined only for values convertible to floating point.

        Returns
        -------
        float
            Distance between the two sequences.

        Raises
        ------
        ValueError
            If lengths differ, the metric is unsupported, or Euclidean distance
            is requested for non-numeric values.
        """
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
            seed: int = None) -> "BaseNumpySequence":
        """
        Create a mutated copy of the sequence.

        Parameters
        ----------
        positions : int or iterable of int, optional
            Sites to mutate. If omitted, select one site uniformly at random.
        values : iterable, optional
            Replacement values corresponding to ``positions``. If omitted,
            sample a different symbol from the sequence alphabet at each site.
        seed : int, optional
            Seed for random site or symbol selection.

        Returns
        -------
        BaseNumpySequence
            Mutated sequence of the same concrete class.

        Raises
        ------
        ValueError
            If the numbers of positions and replacement values differ.
        """
        rng = np.random.default_rng(seed)
        new_np = self._np.copy()

        if positions is None:
            positions = [rng.integers(0, len(self))]
        elif isinstance(positions, int):
            positions = [positions]
        else:
            positions = list(positions)

        if any(isinstance(position, (bool, np.bool_)) for position in positions):
            raise TypeError("mutation positions must be integers")
        if any(not isinstance(position, (int, np.integer)) for position in positions):
            raise TypeError("mutation positions must be integers")
        positions = [int(position) for position in positions]
        if any(position < 0 or position >= len(self) for position in positions):
            raise IndexError("mutation position out of range")

        if values is None:
            choices = [
                [a for a in self.alphabet if a != str(new_np[position])]
                for position in positions
            ]
            if any(not available for available in choices):
                raise ValueError("mutation requires at least two alphabet values")
            values = [rng.choice(available) for available in choices]
        elif not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]

        if len(values) != len(positions):
            raise ValueError("Length of values must equal length of positions.")

        for p, v in zip(positions, values):
            new_np[p] = v

        return self._construct_mutated(new_np)

    def _mutation_sequence_id(self) -> str | None:
        """Return the identifier retained by a mutated sequence."""
        return self.id if self._has_explicit_id else None

    def _construct_mutated(self, sequence: np.ndarray) -> "BaseNumpySequence":
        """Construct a mutation while preserving base-sequence metadata."""
        current_moltype = self._c3_seq.moltype if self._c3_seq else None
        return self.__class__(
            sequence,
            sequence_id=self._mutation_sequence_id(),
            alphabet=self.alphabet,
            moltype=current_moltype,
        )
    
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
            base_mapping = {sym: i for i, sym in enumerate(self.alphabet)}
        else:
            base_mapping = dict(mapping)

        # Build a separate lookup map for robustness without changing class count
        lookup_map = dict(base_mapping)
        # Bridge '-' and 'gap' if only one is present
        if 'gap' in lookup_map and '-' not in lookup_map:
            lookup_map['-'] = lookup_map['gap']
        if '-' in lookup_map and 'gap' not in lookup_map:
            lookup_map['gap'] = lookup_map['-']
        # Add lowercase/uppercase aliases for alphabetic symbols
        for k, v in list(lookup_map.items()):
            if isinstance(k, str) and k and k not in {'gap', '-'}:
                lookup_map.setdefault(k.lower(), v)
                lookup_map.setdefault(k.upper(), v)

        # Look up indices with normalization for string symbols using lookup_map
        def _lookup(sym) -> int:
            s = str(sym)
            # Try direct, then case-insensitive
            if s in lookup_map:
                return lookup_map[s]
            s_up = s.upper()
            if s_up in lookup_map:
                return lookup_map[s_up]
            s_lo = s.lower()
            if s_lo in lookup_map:
                return lookup_map[s_lo]
            # Common gap alt
            if s == '.' and 'gap' in lookup_map:
                return lookup_map['gap']
            raise KeyError(s)

        idxs = np.array([_lookup(s) for s in self._np], dtype=int)
        num_classes = int(max(base_mapping.values())) + 1 if base_mapping else 0
        mat = np.zeros((len(self), num_classes), dtype=int)
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

        # Mirror the normalization used in to_one_hot
        if isinstance(mapping, dict):
            norm_map = dict(mapping)
            if 'gap' in norm_map and '-' not in norm_map:
                norm_map['-'] = norm_map['gap']
            if '-' in norm_map and 'gap' not in norm_map:
                norm_map['gap'] = norm_map['-']
            for k, v in list(norm_map.items()):
                if isinstance(k, str) and k and k not in {'gap', '-'}:
                    norm_map.setdefault(k.lower(), v)
                    norm_map.setdefault(k.upper(), v)
            mapping = norm_map

        def _lookup(sym) -> int:
            s = str(sym)
            if s in mapping:
                return mapping[s]
            s_up = s.upper()
            if s_up in mapping:
                return mapping[s_up]
            s_lo = s.lower()
            if s_lo in mapping:
                return mapping[s_lo]
            if s == '.' and 'gap' in mapping:
                return mapping['gap']
            raise KeyError(s)

        return np.array([_lookup(s) for s in self.ndarray], dtype=int)

    def to_str(self) -> str:
        """Convert the sequence to a concatenated string.

        Returns
        -------
        str
            The sequence in string format.
        """
        return "".join(map(str, self._np))
    
    def remove_gap_arr(self,
                       *,
                       gap_threshold: float = 0.5) -> np.ndarray:
        """Remove sites whose gap posterior exceeds a threshold.

        Parameters
        ----------
        gap_threshold : float, default=0.5
            Maximum retained posterior probability of a gap.

        Returns
        -------
        ndarray
            Renormalized non-gap posterior rows for retained sites.

        Raises
        ------
        ValueError
            If the one-hot width differs from the alphabet size.
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
        filtered = filtered / filtered.sum(axis=1, keepdims=True)
        return filtered
    
    # Factory constructor methods
    @classmethod
    def from_string(cls,
                    s: str,
                    *,
                    alphabet: Iterable = PROT_20,
                    moltype: str | None = None,
                    sequence_id: str | None = None) -> "BaseNumpySequence":
        """
        Build from a plain string like 'ACDE' or '0101'.

        Parameters
        ----------
        s : str
            The sequence string. 
        
        alphabet : Iterable, default=`PROT_20`
            The alphabet. Defaults to the canonical 20 amino acids.
        
        moltype : str, optional
            The (optional) moltype.
        
        sequence_id : str, optional
            The (optional) sequence ID. If `None`, the sequence is used
            as the id. 

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.
        """
        arr = np.array(list(s))
        
        return cls(arr,
                   sequence_id=sequence_id,
                   alphabet=alphabet,
                   moltype=moltype)

    @classmethod
    def from_iterable(cls,
                      it: _SeqLike | np.ndarray,
                      *,
                      alphabet: Iterable  = PROT_20,
                      moltype: str | None = None,
                      sequence_id: str | None = None) -> "BaseNumpySequence":
        """
        Build from any 1D iterable/array of symbols/numbers.
        
        Parameters
        ----------
        it : Iterable
            The sequence iterable. 
        
        alphabet : Iterable, default=`PROT_20`
            The alphabet. Defaults to the canonical 20 amino acids.
        
        moltype : str, optional
            The (optional) moltype.
        
        sequence_id : str, optional
            The (optional) sequence ID. If `None`, the sequence is used
            as the id. 

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.
        """
        arr = np.asarray(list(it)).ravel()
        
        return cls(arr,
                   sequence_id=sequence_id,
                   alphabet=alphabet,
                   moltype=moltype)

    @classmethod
    def from_cogent3(cls,
                     seq: _C3Sequence,
                     *,
                     sequence_id: str | None = None) -> "BaseNumpySequence":
        """
        Build directly from a cogent3 Sequence (moltype auto-carried).

        Parameters
        ----------
        seq : Sequence
            The cogent3 Sequence object. 
        
        sequence_id : str, optional
            The (optional) sequence ID. If `None`, the sequence is used
            as the id. 

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.
        """
        return cls(seq,
                   sequence_id=sequence_id,
                   alphabet=None,
                   moltype=seq.moltype)

    @classmethod
    def from_one_hot(cls,
                     one_hot: np.ndarray,
                     *,
                     alphabet: Iterable = None,
                     sequence_id: str | None = None) -> "BaseNumpySequence":
        """
        Build from (L, |A|) one-hot (argmax per row). No stochastic
        tie-break.

        Parameters
        ----------
        one_hot : np.ndarray
            The sequence one-hot array. 
        
        alphabet : Iterable, default=`None`
            The alphabet. Defaults to computation on the fly based on
            the size of the array.
        
        sequence_id : str, optional
            The (optional) sequence ID. If `None`, the sequence is used
            as the id. 

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.
        """
        one_hot = np.asarray(one_hot)
        if one_hot.ndim != 2:
            raise ValueError("one_hot must be 2D (L, |A|)")
        idx = np.argmax(one_hot, axis=1)
        if alphabet is not None:
            # Auto compute alphabet if not provided.
            alphabet = list(alphabet)
        else: 
            alphabet = list(range(one_hot.shape[1]))
        arr = np.array([alphabet[i] for i in idx], dtype=object)
        
        return cls(arr,
                   sequence_id=sequence_id,
                   alphabet=alphabet)

    @classmethod
    def from_integer(cls,
                     ints: _SeqLike[int] | np.ndarray,
                     *,
                     alphabet: Iterable = None,
                     sequence_id: str | None = None) -> "BaseNumpySequence":
        """
        Build from integer indices into a provided alphabet.

        Parameters
        ----------
        ints : SeqLike
            The int list or iterable.
        
        alphabet : Iterable, default=`None`
            The alphabet. Defaults to computation on the fly based on
            the size on the maximum int.
        
        sequence_id : str, optional
            The (optional) sequence ID. If `None`, the sequence is used
            as the id.

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.
        """
        raw = np.asarray(ints)
        if raw.size == 0:
            raise ValueError("integer sequence must not be empty")
        try:
            numeric = raw.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("integer sequence values must be finite integers") from error
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("integer sequence values must be finite integers")
        idx = numeric.astype(int).ravel()
        
        # Auto compute alphabet if not provided.
        if alphabet is not None:
            alphabet = list(alphabet)
        else:
            alphabet = list(range(max(idx)+1))

        if np.any((idx < 0) | (idx >= len(alphabet))):
            raise ValueError("integer indices out of range for given alphabet")
        arr = np.array([alphabet[i] for i in idx], dtype=object)
        
        return cls(arr,
                   sequence_id=sequence_id,
                   alphabet=alphabet)

    @classmethod
    def random(cls,
               length: int,
               *,
               alphabet: Iterable = PROT_20, 
               seed: int = None,
               sequence_id: str | None = None) -> "BaseNumpySequence":
        """
        Uniform random sequence of given length and alphabet.

        Parameters
        ----------
        length : int
            The length of the random sequences. 
        
        alphabet : Iterable, default=`PROT_20`
            The alphabet for the random sequence. Defaults to the 20
            canonical amino acids.
        
        seed : int, default=`None`
            The random state initialisation seed. 
        
        sequence_id : str, default=`None`
            The (optional) sequence ID. If `None`, the sequence is used
            as the id.

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.            
            
        """
        length = _require_positive_length(length)
        rng = np.random.default_rng(seed)
        A = _validated_alphabet(alphabet)
        arr = np.array(rng.choice(A, size=length), dtype=object)
        
        return cls(arr,
                   sequence_id=sequence_id,
                   alphabet=A)


# BinarySequence can now be much simpler
class BinarySequence(BaseNumpySequence):
    """Represent a sequence whose symbols are restricted to zero and one.

    Parameters
    ----------
    sequence : BaseNumpySequence, sequence of int, or ndarray
        Binary values in positional order.
    sequence_id : str, optional
        Stable identifier for the sequence.

    Raises
    ------
    ValueError
        If any value is not zero or one.
    """
    def __init__(self,
                 sequence: _SeqConvertible,
                 sequence_id: str | None = None) -> None:
        raw = np.asarray(_to_numpy(sequence))
        if raw.size == 0:
            raise ValueError("sequence must not be empty")
        try:
            numeric = raw.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("BinarySequence accepts only finite 0/1 symbols") from error
        if not np.all(np.isfinite(numeric)) or not np.all(
            (numeric == 0.0) | (numeric == 1.0)
        ):
            raise ValueError("BinarySequence accepts only 0/1 symbols")
        arr = numeric.astype(np.int8)

        super().__init__(
            arr,
            sequence_id=sequence_id,
            alphabet=['0', '1'],
            moltype=None,
        )

    def _construct_mutated(self, sequence: np.ndarray) -> "BinarySequence":
        """Construct a binary mutation while preserving an explicit ID."""
        return self.__class__(
            sequence,
            sequence_id=self._mutation_sequence_id(),
        )

    @classmethod
    def from_bits(cls,
                  bits: _SeqLike[int] | np.ndarray,
                  *,
                  sequence_id: str | None = None) -> "BinarySequence":
        """
        Build from a 0/1 iterable/array.
        
        Parameters
        ----------
        bits : _SeqLike[int] or np.ndarray
            The bits to construct the sequence from. 
        
        sequence_id : str, default=`None`
            The (optional) sequence ID. If `None`, the sequence is used
            as the id. 
        
        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.  
        """
        return cls(bits, sequence_id=sequence_id)

    @classmethod
    def from_integer_bits(cls,
                          value: int,
                          *,
                          length: int,
                          msb_first: bool = True,
                          sequence_id: str | None = None) -> "BinarySequence":
        """
        Build from an integer's bit pattern clipped/padded to length.

        Parameters
        ----------
        value : int
            The integer value to use
        
        length : int
            The length to truncate the bits to. 
        
        msb_first : bool, default=`True`
            Boolean to mask the first bit. 
        
        sequence_id : str, default=`None`
            The (optional) sequence ID. If `None`, the sequence is used
            as the id.

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.  
        """
        if value < 0:
            raise ValueError("value must be non-negative")
        length = _require_positive_length(length)
        bits = np.array(list(np.binary_repr(value, width=length)), dtype=int)
        if not msb_first:
            bits = bits[::-1]
        return cls(bits, sequence_id=sequence_id)

    @classmethod
    def random(cls,
               length: int,
               *,
               p_one: float = 0.5,
               seed: int = None,
               sequence_id: str | None = None) -> "BinarySequence":
        
        """
        Construct random sequenceo of bits. 

        Parameters
        ----------
        length : int
            The length of the sequence. 
        
        p_one : float, default=0.5
            The probability of a value being clamped to `1`.
        
        seed : int, default=`None`
            The random state initialisation seed. 
        
        sequence_id : str, default=`None`
            The (optional) sequence ID. If `None`, the sequence is used
            as the id. 

        Returns
        -------
        BaseNumpySequence
            The constructed BaseNumpySequence object.  
        """
        length = _require_positive_length(length)
        if not np.isfinite(p_one) or not 0.0 <= p_one <= 1.0:
            raise ValueError("p_one must be finite and between zero and one")
        rng = np.random.default_rng(seed)
        arr = rng.random(length) < p_one
        return cls(arr.astype(int), sequence_id=sequence_id)


class MultialleleSequence(BaseNumpySequence):
    """Represent a sequence over an explicitly supplied alphabet.

    Parameters
    ----------
    sequence : BaseNumpySequence, sequence of scalar, or ndarray
        Alleles in positional order.
    alphabet : iterable
        Permitted allele values.
    sequence_id : str, optional
        Stable identifier for the sequence.

    Raises
    ------
    ValueError
        If ``sequence`` contains a value outside ``alphabet``.
    """

    def __init__(self,
                 sequence: _SeqConvertible,
                 alphabet: Iterable,
                 sequence_id: str | None = None):
        arr = _to_numpy(sequence)
        if arr.size == 0:
            raise ValueError("sequence must not be empty")
        alphabet = _validated_alphabet(alphabet)
        if not set(arr).issubset(set(alphabet)):
            raise ValueError("MultialleleSequence accepts only symbols from the alphabet")
        super().__init__(
            arr,
            sequence_id=sequence_id,
            alphabet=alphabet,
            moltype=None,
        )

    def _construct_mutated(self, sequence: np.ndarray) -> "MultialleleSequence":
        """Construct a multiallelic mutation with its alphabet and ID."""
        return self.__class__(
            sequence,
            alphabet=self.alphabet,
            sequence_id=self._mutation_sequence_id(),
        )
        
    @classmethod
    def random(cls,
               length: int,
               *,
               alphabet: Iterable,
               seed: int | None = None,
               sequence_id: str | None = None) -> "MultialleleSequence":
        """Generate a uniformly sampled multiallelic sequence.

        Parameters
        ----------
        length : int
            Number of sites.
        alphabet : iterable
            Values sampled independently at each site.
        seed : int, optional
            Random-number-generator seed.
        sequence_id : str, optional
            Sequence identifier. Reserved for API consistency; the current
            constructor derives the identifier from sequence values.

        Returns
        -------
        MultialleleSequence
            Sampled sequence.
        """
        length = _require_positive_length(length)
        rng = np.random.default_rng(seed)
        A = _validated_alphabet(alphabet)
        arr = np.array(rng.choice(A, size=length), dtype=object)
        return cls(arr, alphabet=A, sequence_id=sequence_id)

    @classmethod
    def from_string(cls,
                    s: str,
                    *,
                    alphabet: Iterable,
                    sequence_id: str | None = None) -> "MultialleleSequence":
        """Construct a multiallelic sequence from a string.

        Parameters
        ----------
        s : str
            String whose characters become sites.
        alphabet : iterable
            Permitted character values.
        sequence_id : str, optional
            Sequence identifier. Reserved for API consistency; the current
            constructor derives the identifier from sequence values.

        Returns
        -------
        MultialleleSequence
            Constructed sequence.
        """
        return cls(list(s), alphabet=alphabet, sequence_id=sequence_id)


class SoftSequence(BaseNumpySequence):
    """Represent a sequence as normalized categorical posteriors.

    Parameters
    ----------
    aa_posterior : np.ndarray
        Finite, non-negative array of shape ``(L, A)``. Each row must
        sum to one within an absolute tolerance of ``1e-8``.
    alphabet : iterable
        Non-empty, unique ordered alphabet corresponding to posterior columns.
    hard_rule : {'argmax', 'sample'}, default 'argmax'
        How to derive the proxy hard sequence used by existing code.
    gap_posterior : np.ndarray, optional
        Finite array of shape ``(L, 1)`` containing gap probabilities, or a
        normalized ``(L, 2)`` matrix ordered as gap and non-gap. When provided,
        append a gap channel to the conditional amino-acid posterior.
    seed : int, optional
        Random-number-generator seed.
    sequence_id : str, optional
        Stable identifier retained when the sequence is resampled or mutated.

    Raises
    ------
    ValueError
        If posterior or alphabet invariants are violated.
    """
    _NORMALIZATION_ATOL = 1e-8

    def __init__(self,
                 aa_posterior: np.ndarray,
                 *,
                 alphabet: Iterable,
                 hard_rule: str = "argmax",
                 gap_posterior: np.ndarray = None,
                 seed: int = None,
                 sequence_id: str | None = None):

        core_alphabet = _validated_alphabet(alphabet)
        aa_posterior = self._validate_posterior(
            aa_posterior,
            width=len(core_alphabet),
            name="aa_posterior",
        )
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Determine the alphabet to be used for the hard sequence proxy
        hard_sequence_alphabet = core_alphabet

        if gap_posterior is not None:
            if "gap" in core_alphabet or "-" in core_alphabet:
                raise ValueError("alphabet must not already contain a gap channel")
            gap_posterior = self._validate_gap_posterior(
                gap_posterior,
                length=aa_posterior.shape[0],
            )
            posterior = self.compute_conditional_gap_dist(
                aa_post_dist=aa_posterior,
                gap_post_dist=gap_posterior,
            )
            # Use an extended alphabet that includes the gap character
            hard_sequence_alphabet = core_alphabet + ['gap']
        else:
            posterior = aa_posterior.copy()

        if hard_rule == "argmax":
            hard = posterior.argmax(axis=1)
        elif hard_rule == "sample":
            # Use the full posterior for sampling
            hard = [self._rng.choice(posterior.shape[1], p=row) for row in posterior]
        else:
            raise ValueError("hard_rule must be 'argmax' or 'sample'")

        self._posterior = np.array(posterior, dtype=float, copy=True)
        self._posterior.setflags(write=False)
        super().__init__(
            [hard_sequence_alphabet[i] for i in hard],
            sequence_id=sequence_id,
            alphabet=hard_sequence_alphabet,
        )

    @classmethod
    def _validate_posterior(
        cls,
        posterior: np.ndarray,
        *,
        width: int,
        name: str,
    ) -> np.ndarray:
        """Return a defensive copy of a validated probability matrix."""
        try:
            result = np.array(posterior, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numeric probabilities") from error
        if result.ndim != 2:
            raise ValueError(f"{name} must be 2D with shape (L, A)")
        if result.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one site")
        if result.shape[1] != width:
            raise ValueError(f"{name} width must match alphabet length")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values")
        if np.any(result < 0.0):
            raise ValueError(f"{name} must contain non-negative values")
        if not np.allclose(
            result.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=cls._NORMALIZATION_ATOL,
        ):
            raise ValueError(f"{name} rows must sum to one")
        return result

    @staticmethod
    def _validate_gap_posterior(
        gap_posterior: np.ndarray,
        *,
        length: int,
    ) -> np.ndarray:
        """Return a validated ``(L, 1)`` gap-probability matrix."""
        try:
            result = np.array(gap_posterior, dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError("gap_posterior must contain numeric probabilities") from error
        if result.shape not in {(length, 1), (length, 2)}:
            raise ValueError("gap_posterior must have shape (L, 1) or (L, 2)")
        if not np.all(np.isfinite(result)):
            raise ValueError("gap_posterior must contain only finite values")
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError("gap_posterior values must lie between zero and one")
        if result.shape[1] == 2:
            if not np.allclose(
                result.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=SoftSequence._NORMALIZATION_ATOL,
            ):
                raise ValueError("two-state gap_posterior rows must sum to one")
            result = result[:, :1]
        return result

    @property
    def posterior(self) -> np.ndarray:
        """Return a read-only view of the posterior matrix."""
        view = self._posterior.view()
        view.setflags(write=False)
        return view

    @property
    def ungapped_arr(self) -> np.ndarray:
        """
        Return a (L, A) probabilistic array with the gap channel
        marginalised out. If the posterior already has no gap channel,
        return it as-is.
        """
        P = self._posterior
        _, C = P.shape

        # If a gap symbol exists in this sequence's alphabet, assume the
        # posterior includes it (same order) and renormalise by (1 - p_gap).
        if "gap" in self.alphabet:
            gap_idx = self.alphabet.index("gap")
            if not (0 <= gap_idx < C):
                raise ValueError(
                    "SoftSequence posterior width does not match alphabet including gap"
                )
            aa = np.delete(P, gap_idx, axis=1)      # (L, A)
            p_gap = P[:, gap_idx:gap_idx + 1]       # (L, 1)
            denom = np.clip(1.0 - p_gap, 1e-12, None)
            return aa / denom

        # No explicit gap channel — posterior already (L, A)
        return P.copy()

    def remove_gap_arr(self, *, gap_threshold: float = 0.5) -> np.ndarray:
        """
        For SoftSequence, prefer probabilistic gap removal by
        marginalisation, not thresholding. Ignores gap_threshold and
        returns the renormalised amino-acid posterior of shape (L, A).
        """
        return self.ungapped_arr

    def map_values(self) -> np.ndarray:
        """
        Method to return the maximum posterior value for each site.

        Returns
        -------
        np.ndarray
            The maximum posterior value for each site.
        """
        return self.posterior.max(axis=1)

    def entropy(self) -> np.ndarray:
        """
        Method to compute the entropy of the posterior distribution.

        Returns
        -------
        np.ndarray
            The entropy for each site in the sequence.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(self.posterior > 0, np.log(self.posterior), 0.0)
        return -np.sum(self.posterior * logp, axis=1)

    def resample(self) -> "SoftSequence":
        """
        Generate a new hard proxy by sampling each posterior row.
        """
        return self.__class__(
            self._posterior,
            alphabet=self.alphabet,
            hard_rule="sample",
            seed=self._seed,
            sequence_id=self._mutation_sequence_id(),
        )

    def _construct_mutated(self, sequence: np.ndarray) -> "SoftSequence":
        """Construct a soft mutation by making changed sites deterministic."""
        posterior = self._posterior.copy()
        alphabet = self.alphabet
        for index, (old_value, new_value) in enumerate(zip(self._np, sequence)):
            if str(old_value) == str(new_value):
                continue
            try:
                allele_index = alphabet.index(str(new_value))
            except ValueError as error:
                raise ValueError(
                    f"mutation value {new_value!r} is not in the alphabet"
                ) from error
            posterior[index] = 0.0
            posterior[index, allele_index] = 1.0
        return self.__class__(
            posterior,
            alphabet=alphabet,
            hard_rule="argmax",
            seed=self._seed,
            sequence_id=self._mutation_sequence_id(),
        )

    @classmethod
    def from_posteriors(cls,
                        aa_posterior: np.ndarray,
                        *,
                        alphabet: Iterable = PROT_20,
                        gap_posterior: np.ndarray | None = None,
                        hard_rule: Literal["argmax", "sample"] = "argmax",
                        seed: int = None,
                        sequence_id: str | None = None) -> "SoftSequence":
        """
        Alias around __init__ for clarity / parity with other classes.

        Parameters
        ----------
        posterior : np.ndarray
            Shape (L, A). Rows are sites, columns are alphabet symbols.
        
        alphabet : Iterable, default=`PROT_20`
            The ordered alphabet corresponding to columns of `posterior`.
        
        hard_rule : {'argmax', 'sample'}, default 'argmax'
            How to derive the proxy hard sequence used by existing code.
        
        seed : int, default=`None`
            The seed for random state initialisation.
        
        Returns
        -------
        SoftSequence
            The constructed soft sequence.
        """
        return cls(aa_posterior,
                   alphabet=alphabet,
                   gap_posterior=gap_posterior,
                   hard_rule=hard_rule,
                   seed=seed,
                   sequence_id=sequence_id)

    @staticmethod
    def compute_conditional_gap_dist(aa_post_dist: np.ndarray,       
                                     gap_post_dist: np.ndarray) -> np.ndarray: 
    
        """
        Compute the conditional distribution of amino acids given gaps.
        
        Parameters
        ----------
        aa_post_dist : np.ndarray
            The posterior distribution of amino acids, shape (L, 20).
        gap_post_dist : np.ndarray
            The posterior distribution of gaps, shape (L, 1).   
        
        Returns
        -------
        np.ndarray
            The conditional distribution of amino acids given gaps, shape (L, 21).
        """
        if aa_post_dist.shape[0] != gap_post_dist.shape[0]:
            raise ValueError("aa_post_dist and gap_post_dist must share length L")

        d = gap_post_dist[:, 0:1]
        num_sites, num_alleles = aa_post_dist.shape
        # Create the conditional matrix with the correct dimensions
        cond = np.empty((num_sites, num_alleles + 1), dtype=float)
        # Fill the allele probabilities
        cond[:, :num_alleles] = aa_post_dist * (1.0 - d)
        # Fill the gap probability
        cond[:, num_alleles]  = d[:, 0]
        return cond

def make_sequence(sequence: _SeqConvertible,
                  *,
                  binary: bool | None = None,
                  alphabet: Iterable | None = None,
                  moltype: str | None = None) -> BaseNumpySequence:
    """
    Function to create a sequence object from various input types.
    
    Parameters
    ----------
    sequence : _SeqConvertible
        The sequence data, which can be a BaseNumpySequence, sequence-like,
        numpy array, or cogent3 Sequence.
    binary : bool, optional
        If True, create a BinarySequence. If None, infer from the sequence data.
    alphabet : Iterable, optional
        The alphabet of the sequence. If None, inferred from the sequence data.
    moltype : str, optional
        The moltype of the sequence. If None, no moltype is set.
    
    Returns
    -------
    BaseNumpySequence
        A BaseNumpySequence or BinarySequence object based on the input.
    """
    if isinstance(sequence, BaseNumpySequence) and alphabet is None:
        return sequence

    seq_id = getattr(sequence, 'id', getattr(sequence, 'name', None))
    seq_np = _to_numpy(sequence)
    
    # If an alphabet wasn't passed, but the original object had one, use it.
    if alphabet is None and isinstance(sequence, BaseNumpySequence):
        alphabet = sequence.alphabet

    if binary is True or (binary is None and set(seq_np).issubset({0, 1})):
        return BinarySequence(seq_np, sequence_id=seq_id)
    
    return BaseNumpySequence(seq_np, sequence_id=seq_id, alphabet=alphabet, moltype=moltype)

# Batch factory functions
def as_sequences(items: Iterable[Union[str, _SeqLike, np.ndarray, BaseNumpySequence, _C3Sequence]],
                 *,
                 binary: bool = False,
                 alphabet: Iterable | None = PROT_20,
                 moltype: str | None = None) -> List[BaseNumpySequence]:
    """
    Coerce a heterogeneous iterable of inputs into BaseNumpySequence
    (or BinarySequence). Respects existing `make_sequence` behavior.

    Parameters
    ----------
    items : Iterable
        Iterable of heterogenous (or homogenous) input types.
    
    binary : bool, default=`False`
        Boolean for whether sequences are binary.
    
    alphabet : iterable, default=`PROT_20`
        The sequence alphabet.
    
    moltype : str, default=`None`
        The (optional) moltype.
    
    Returns
    -------
    List[BaseNumpySequence]
        List of constructed `BaseNumpySequence` objects.
    """
    out: list[BaseNumpySequence] = []
    for x in items:
        if isinstance(x, str) and not binary:
            if set(x).issubset({'0', '1'}):
                out.append(BinarySequence.from_bits([int(ch) for ch in x]))
            else:
                out.append(BaseNumpySequence.from_string(x, alphabet=alphabet, moltype=moltype))
        else:
            inferred_flag = True if binary is True else None
            out.append(make_sequence(x, binary=inferred_flag, alphabet=alphabet, moltype=moltype))
    return out


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

        return [BaseNumpySequence([s], alphabet=alphabet) for s in alphabet]
    sequences = []
    for s in alphabet:
        for sub_sequence in generate_sequences(length - 1, alphabet):
            # Pass the alphabet to the constructor
            sequences.append(BaseNumpySequence([s] + sub_sequence.to_array().tolist(), alphabet=alphabet))
    return sequences

def read_from_fasta(filepath: Path,
                    moltype: str = "protein") -> List[BaseNumpySequence]:
    """
    Reads sequences from a FASTA file and returns them as a list of
    BaseNumpySequence objects.

    Parameters
    ----------
    filepath : Path
        The path to the FASTA file.
    moltype : str, default='protein'
        The molecular type of the sequences (e.g., 'protein', 'dna',
        'rna'). This is passed to cogent3's sequence loader.

    Returns
    -------
    List[BaseNumpySequence]
        A list of BaseNumpySequence objects from the FASTA file.
    """
    # Use cogent3 to load the sequences from the FASTA file
    cogent3 = require_optional(
        "cogent3",
        extra="phylogeny",
        purpose="loading sequence files",
    )
    seq_collection = cogent3.load_unaligned_seqs(filepath, moltype=moltype)

    moltype_obj = cogent3.get_moltype(moltype)
    alph = list(moltype_obj.alphabet)

    numpy_sequences = []
    for seq in seq_collection.iter_seqs():
        numpy_seq = make_sequence(seq, moltype=moltype, alphabet=alph)
        numpy_sequences.append(numpy_seq)
    return numpy_sequences
