from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
import operator
from typing import TYPE_CHECKING, Dict, Literal, List, Any, Tuple, Union, Mapping, Callable, Sequence
import numpy as np
from scipy import stats
from .._optional import require_optional

if TYPE_CHECKING:
    import torch


def _torch_module():
    return require_optional(
        "torch",
        extra="ml",
        purpose="PyTorch fitness tensors",
    )


def _is_torch_tensor(value: object) -> bool:
    if not type(value).__module__.startswith("torch"):
        return False
    return isinstance(value, _torch_module().Tensor)


def _validated_categories(categories: Sequence[str]) -> list[str]:
    """Return a non-empty defensive category list with unique values."""
    values = list(categories)
    if not values:
        raise ValueError("categories must not be empty")
    for index, value in enumerate(values):
        if any(value == previous for previous in values[:index]):
            raise ValueError("categories must contain unique values")
    return values


def _as_float_matrix(value: object, *, name: str) -> np.ndarray:
    """Return a defensive two-dimensional floating-point matrix."""
    raw = value.detach().cpu().numpy() if _is_torch_tensor(value) else value
    try:
        matrix = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix")
    return matrix


# Fitness oeprates as a `layer` over the fitness landscape object.
class BaseFitnessLayer(ABC):
    """
    Base class for fitness layers in a fitness landscape. 

    Attributes
    ----------
    name : str
        The name of the fitness layer.
    metadata : Dict, optional
        Additional metadata associated with the fitness layer.
    """
    def __init__(self, name: str,
                 metadata: Dict = None) -> None:
        
        self.name = name
        self.metadata = metadata or {}

    @property
    @abstractmethod
    def dtype(self) -> Literal['numeric', 'categorical']:
        """
        Returns the data type of the fitness layer.
        """
        pass

    @abstractmethod
    def get_tensor(self) -> torch.Tensor:
        """
        Method to return the fitness layer as a PyTorch tensor.        
        """
        pass

    @abstractmethod
    def to_scalar(self,
                  **kwargs) -> np.ndarray:
        """
        Method to convert the fitness layer to a scalar representation.
        """
        pass

    @abstractmethod
    def get_value(self,
                  sequence_index: int) -> Any:
        """
        Retrieves the native fitness value(s) for a single sequence.
        """
        pass
    
    def _validate_length(self,
                         expected: int, *,
                         name: str = "") -> None:
        n = len(self)
        if n != expected:
            lab = f" for layer '{self.name}'" if getattr(self, 'name', None) else ""
            raise ValueError(f"Layer length mismatch{lab}: got {n}, expected {expected}. {name}")

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"
    
    @abstractmethod
    def __len__(self):
        pass

    
class NumericFitness(BaseFitnessLayer):
    """Represent scalar or replicated numeric fitness measurements.

    Parameters
    ----------
    name : str
        Layer name.
    values : sequence of float, sequence of sequence of float, ndarray, or torch.Tensor
        One scalar per sequence or one collection of replicate measurements per
        sequence.
    metadata : dict, optional
        Free-form layer metadata.

    Attributes
    ----------
    name : str
        The name of the fitness layer.
    values : Sequence[float] | Sequence[Sequence[float]]
        Either a 1-D sequence of scalar fitness values, or a sequence
        where each inner sequence contains replicate fitness values for
        a sequence.
    metadata : Dict, optional
        Additional metadata associated with the fitness layer.
    """
    def __init__(self,
                 name: str,
                 values: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray, torch.Tensor],
                 metadata: Dict = None) -> None:
        
        super().__init__(name=name, metadata=metadata)

        self._replicates = self._normalize_values(values)
        # For each sequence, create a normal distribution based on its replicates
        self._distributions = [
            stats.norm(loc=np.mean(r), scale=np.std(r)) if len(r) > 1 else stats.norm(loc=r[0], scale=0)
            for r in self._replicates
        ]

    @staticmethod
    def _normalize_values(
        values: Union[Sequence[float], Sequence[Sequence[float]], np.ndarray, torch.Tensor]
    ) -> List[List[float]]:
        err = (
            "Input 'values' must be a 1-D sequence of scalars or a 2-D "
            "sequence of per-sequence replicate values."
        )

        if _is_torch_tensor(values):
            raw_values = values.detach().cpu().numpy().tolist()
        elif isinstance(values, np.ndarray):
            if values.ndim == 0 or values.ndim > 2:
                raise TypeError(err)
            raw_values = values.tolist()
        else:
            try:
                raw_values = list(values)
            except TypeError as exc:
                raise TypeError(err) from exc

        if not raw_values:
            return []

        normalized: List[List[float]] = []
        for row in raw_values:
            if _is_torch_tensor(row):
                row = row.detach().cpu().numpy()

            if isinstance(row, np.ndarray):
                arr = np.asarray(row, dtype=float)
                if arr.ndim == 0:
                    normalized.append([float(arr.item())])
                    continue
                if arr.ndim > 1:
                    raise TypeError(err)
                row_values = arr.tolist()
            elif isinstance(row, (list, tuple)):
                arr = np.asarray(row, dtype=float)
                if arr.ndim > 1:
                    raise TypeError(err)
                row_values = arr.tolist()
            else:
                try:
                    normalized.append([float(row)])
                except (TypeError, ValueError) as exc:
                    raise TypeError(err) from exc
                continue

            if len(row_values) == 0:
                row_values = [float("nan")]

            normalized.append([float(x) for x in row_values])

        return normalized

    @property
    def dtype(self) -> Literal['numeric']:
        """Return the layer kind, ``'numeric'``."""
        return 'numeric'

    def get_tensor(self) -> torch.Tensor:
        """
        Method to convert the fitness layer to a PyTorch tensor.
        
        Returns
        -------
        torch.Tensor
            A tensor representation of the fitness layer's replicate values.
            Each sequence's replicates are padded to the maximum length found
            across all sequences to ensure consistent tensor shape.
            This tensor will have shape (num_sequences, max_replicates),
        """
        max_reps = max(len(r) for r in self._replicates) if self._replicates else 0
        
        padded_reps = [
            r + [np.nan] * (max_reps - len(r)) for r in self._replicates
        ]
        torch = _torch_module()
        return torch.tensor(padded_reps, dtype=torch.float32)

    def to_scalar(self,
                  aggregate_func=np.mean) -> np.ndarray:
        """
        Method to convert the fitness layer to a scalar real value.

        Parameters
        ----------
        aggregate_func : callable, optional
            A function to aggregate the replicate values into a single
            scalar. Default is `np.mean`, but could be `np.median`,
            `np.max`.

        Returns
        -------
        np.ndarray
            An array of scalar values, one for each sequence, computed
            using the specified aggregation function (default is mean).
        """
        return np.array([aggregate_func(r) for r in self._replicates])

    def get_value(self,
                  sequence_index: int) -> List[float]:
        """Return replicate values for one sequence.
        
        Parameters
        ----------
        sequence_index : int
            Positional sequence index.

        Returns
        -------
        list of float
            Replicate measurements for the sequence.
        """
        return self._replicates[sequence_index]
    
    def __len__(self):
        return len(self._replicates)
    
    @classmethod
    def from_scalars(cls,
                     name: str,
                     values: Union[List[float], np.ndarray],
                     *,
                     metadata: Dict | None = None) -> "NumericFitness":
        """
        Constructor method to build fitness layer with one scalar per
        sequence.

        Parameters
        ----------
        name : str
            The layer name.
        values : List[floar] OR np.ndarray
            The scalar values for each sequence.
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.
    
        Returns
        -------
        NumericFitness
            An instance of the NumericFitness class initialized with
            the provided parameters.
        """
        v = np.asarray(values, dtype=float).ravel().tolist()
        reps = [[float(x)] for x in v]
        return cls(name=name, values=reps, metadata=metadata)

    @classmethod
    def from_replicates(cls,
                        name: str,
                        replicates: List[Union[List[float], np.ndarray]],
                        *,
                        metadata: Dict | None = None,
                        coerce_numeric: bool = True) -> "NumericFitness":
        

        """
        Constructor method to build fitness from a list of replicates
        sequence.

        Parameters
        ----------
        name : str
            The layer name.
        
        replicates : List[float[List] OR np.ndarray]
            The replicate values for each sequence.
        
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.
        
        coerce_numeric : bool, default=True
            If `True`, will convert all values to float.
    
        Returns
        -------
        NumericFitness
            An instance of the NumericFitness class initialized with
            the provided parameters.
        
        """
        vals: List[List[float]] = []
        for r in replicates:
            arr = np.asarray(r, dtype=float if coerce_numeric else None).tolist()
            if len(arr) == 0:
                # keep a NaN placeholder so shape stays consistent
                arr = [float("nan")]
            vals.append(arr)
        return cls(name=name, values=vals, metadata=metadata)

    @classmethod
    def from_tensor(cls,
                    name: str,
                    tensor: Union[np.ndarray, torch.Tensor],
                    *,
                    pad_strategy: Literal["keep", "trim_tail_nans"] = "keep",
                    metadata: Dict | None = None) -> "NumericFitness":
        """
        Build from a (num_sequences, num_replicates) matrix.
        NaNs are preserved as missing replicates.

        Constructor method to build fitness layer from a tensor matrix
        of (num_sequences, num_replicates).

        Parameters
        ----------
        name : str
            The layer name.
        
        tensor : np.ndarray OR torch.Tensor
            a 2-D array of shape (num_sequences, num_replicates).
        
        pad_strategy : str, default=`keep`
            Strategy for handling missing values. Options are:
            - "keep": Keep all values, including NaNs.
            - "trim_tail_nans": Trim only trailing NaNs from each sequence,
              keeping leading/middle NaNs (safer for padding artifacts).

        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        NumericFitness
            An instance of the NumericFitness class initialized with
            the provided parameters.
        """
        arr = tensor.detach().cpu().numpy() if _is_torch_tensor(tensor) else np.asarray(tensor)
        if arr.ndim != 2:
            raise ValueError("NumericFitness.from_tensor expects a 2-D array (num_sequences, num_replicates)")
        vals: List[List[float]] = []
        for row in arr:
            r = row.tolist()
            if pad_strategy == "trim_tail_nans":
                # drop only trailing NaNs, keep leading/middle (safer for padding artifacts)
                while r and (r[-1] is None or (isinstance(r[-1], float) and np.isnan(r[-1]))):
                    r.pop()
                if not r:
                    r = [float("nan")]
            vals.append([float(x) for x in r])
        return cls(name=name, values=vals, metadata=metadata)

    @classmethod
    def from_index_map(cls,
                       name: str,
                       mapping: Mapping[int, Union[float, List[float]]],
                       *,
                       length: int,
                       fill: float | None = float("nan"),
                       metadata: Dict | None = None) -> "NumericFitness":
        """Build a numeric layer from values indexed by sequence position.

        Parameters
        ----------
        name : str
            Layer name.
        mapping : mapping of int to float or list of float
            Scalar or replicate values keyed by zero-based sequence index.
        length : int
            Total number of sequences in the output layer.
        fill : float, optional
            Value assigned to missing indices. ``None`` is represented as NaN.
        metadata : dict, optional
            Free-form layer metadata.

        Returns
        -------
        NumericFitness
            Constructed numeric layer.
        """
        vals: List[List[float]] = []
        for i in range(length):
            v = mapping.get(i, None)
            if v is None:
                vals.append([fill] if fill is not None else [float("nan")])
            else:
                if isinstance(v, (list, tuple, np.ndarray)):
                    vv = np.asarray(v, dtype=float).tolist()
                    if not vv:
                        vv = [float("nan")]
                    vals.append(vv)
                else:
                    vals.append([float(v)])
        return cls(name=name, values=vals, metadata=metadata)

    @classmethod
    def random(cls,
               name: str,
               *,
               length: int,
               reps: int = 1,
               dist: Literal["normal", "uniform"] = "normal",
               loc: float = 0.0,
               scale: float = 1.0,
               low: float = 0.0,
               high: float = 1.0,
               seed: int | None = None,
               metadata: Dict | None = None) -> "NumericFitness":
        """
        Constrctor method for a random numeric fitness layer.
        
        Parameters
        ----------
        name : str
            The layer name.
        
        length : int
            The number of sequences described by the fitness layer.
        
        reps : int, default=`1`
            The number of replicates per sequence. Each sequence will have
            `reps` values.
        
        dist : str, default=`"normal"`
            The distribution to sample from. Options are:
            - "normal": Samples from a normal distribution with specified `loc` and `scale`.
            - "uniform": Samples from a uniform distribution between `low` and `high`.
        
        loc : float, default=`0.0`
            Mean for normal distribution (ignored for uniform).
        
        scale : float, default=`1.0`
            Standard deviation for normal distribution (ignored for uniform).
        
        low : float, default=`0.0`
            Lower bound for uniform distribution (ignored for normal).
        
        high : float, default=`1.0`
            Upper bound for uniform distribution (ignored for normal).
        
        seed : int, optional
            Random seed for reproducibility.
        
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        NumericFitness
            An instance of the NumericFitness class initialized with
            the provided parameters.
        """
        rng = np.random.default_rng(seed)
        if dist == "normal":
            mat = rng.normal(loc=loc, scale=scale, size=(length, reps))
        elif dist == "uniform":
            mat = rng.uniform(low=low, high=high, size=(length, reps))
        else:
            raise ValueError(f"Unknown dist: {dist}")
        return cls.from_tensor(name=name, tensor=mat, metadata=metadata)


class CategoricalFitness(BaseFitnessLayer):
    """Represent one categorical fitness value per sequence.

    Parameters
    ----------
    name : str
        Layer name.
    values : list of str
        Category assigned to each sequence.
    categories : list of str, optional
        Ordered allowed categories. If omitted, sorted unique values are used.
    metadata : dict, optional
        Free-form layer metadata.

    Attributes
    ----------
    name : str
        The name of the fitness layer.
    values : List[str]
        A list of categorical values representing the fitness of each
        sequence.
    categories : List[str], optional
        A list of unique categories that the values can take. If not
        provided, it will be derived from the values.
    metadata : Dict, optional
        Additional metadata associated with the fitness layer.
    """
    def __init__(self,
                 name: str,
                 values: List[str],
                 categories: List[str] = None,
                 metadata: Dict = None) -> None:
        super().__init__(name=name,
                         metadata=metadata)
        self._values = list(values)

        if categories is None:
            try:
                inferred_categories = sorted(set(self._values))
            except TypeError as error:
                raise ValueError("categorical values must be hashable") from error
            self._categories = tuple(_validated_categories(inferred_categories))
        else:
            self._categories = tuple(_validated_categories(categories))

        self.category_map = {cat: i for i, cat in enumerate(self._categories)}

        if not all(v in self.category_map for v in self._values):
            raise ValueError("All fitness 'values' must be present in the 'categories' list.")

    @property
    def categories(self) -> list[str]:
        """Return a defensive copy of the ordered categories."""
        return list(self._categories)

    @property
    def dtype(self) -> Literal['categorical']:
        """Return the layer kind, ``'categorical'``."""
        return 'categorical'

    def get_tensor(self) -> torch.Tensor:
        """
        Method to convert the fitness layer to a one-hot encoded
        PyTorch tensor.

        Returns
        -------
        torch.Tensor
            A one-hot encoded tensor representation of the categorical
            fitness values. Each sequence's categorical value is represented
            as a one-hot vector, where the length of the vector is equal to
            the number of unique categories. The tensor will have shape
            (num_sequences, num_categories).
        """
        num_classes = len(self.categories)
        torch = _torch_module()
        one_hot = torch.zeros(len(self._values), num_classes, dtype=torch.float32)
        
        for i, val in enumerate(self._values):
            one_hot[i, self.category_map[val]] = 1.0
            
        return one_hot

    def to_scalar(self,
                  rank_map: Dict[str, int] = None) -> np.ndarray:
        """
        Method to convert the categorical fitness layer to a scalar
        representation based on a provided rank map.

        Parameters
        ----------
        rank_map : Dict[str, int], optional
            A mapping from category names to integer ranks. If not provided,
            the default category order will be used, which is the order in
            which categories were defined in the layer.
        
        Returns
        -------
        np.ndarray
            An array of integer ranks corresponding to the categorical values.
            Each value in the layer is replaced by its rank according to the
            provided rank map. If a value is not found in the rank map, it will
            raise a ValueError.
        """
        _rank_map = rank_map or self.category_map
        if not all(c in _rank_map for c in self.categories):
            raise ValueError("The provided rank_map does not cover all categories.")
            
        return np.array([_rank_map[v] for v in self._values], dtype=int)
    
    def get_value(self,
                  sequence_index: int) -> str:
        """Return the category assigned to one sequence.
        
        Parameters
        ----------
        sequence_index : int
            Positional sequence index.

        Returns
        -------
        str
            Assigned category.
        """
        return self._values[sequence_index]
    
    def __len__(self):
        return len(self._values)

    @classmethod
    def from_values(cls,
                    name: str,
                    values: Union[List[str], np.ndarray],
                    *,
                    categories: List[str] | None = None,
                    metadata: Dict | None = None) -> "CategoricalFitness":
        
        """
        Constructor method to build categorical fitness layer from a
        list of values. One value per sequence is expected.

        Parameters
        ----------
        name : str
            The layer name.
        
        values : List[str] OR np.ndarray
            The categorical values of each sequence provided by the
            fitness layer.
        
        categories : List[str], default=`None`
            A list of unique categories that values can take. If not
            provided, categories are derived from the `values` list.

        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        CategoricalFitness
            An instance of the CategoricalFitness class initialized
            with the provided parameters.
        """
        vals = [str(v) for v in np.asarray(values, dtype=object).ravel().tolist()]
        return cls(name=name, values=vals, categories=categories, metadata=metadata)

    @classmethod
    def from_one_hot(cls,
                     name: str,
                     one_hot: Union[np.ndarray, torch.Tensor],
                     *,
                     categories: List[str],
                     metadata: Dict | None = None) -> "CategoricalFitness":
        
        """
        Constructor method to build categorical fitness layer from a
        ohe-hot encoded 2D array or tensor. Each row corresponds to a
        sequence and each column corresponds to a category.

        Parameters
        ----------
        name : str
            The layer name.
        
        one_hot : np.ndarray OR torch.Tensor
            A 2-D array or tensor of shape (num_sequences,
            num_categories) representing one-hot encoded categorical
            values. Each row corresponds to a sequence and each column
            corresponds to a category.
        
        categories : List[str], default=`None`
            A list of unique categories that values can take. If not
            provided, categories are derived from the `values` list.

        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        CategoricalFitness
            An instance of the CategoricalFitness class initialized
            with the provided parameters.
        """
        categories = _validated_categories(categories)
        mat = _as_float_matrix(one_hot, name="one_hot")
        if mat.shape[1] != len(categories):
            raise ValueError("one_hot width must match categories")
        if not np.all(np.isfinite(mat)):
            raise ValueError("one_hot must contain only finite values")
        if not np.all((mat == 0.0) | (mat == 1.0)):
            raise ValueError("one_hot must contain only zero and one")
        if not np.all(mat.sum(axis=1) == 1.0):
            raise ValueError("one_hot rows must contain exactly one active category")
        idx = np.argmax(mat, axis=1)
        vals = [categories[i] for i in idx]
        return cls(name=name, values=vals, categories=categories, metadata=metadata)

    @classmethod
    def from_index_map(cls,
                       name: str,
                       mapping: Mapping[int, str],
                       *,
                       length: int,
                       default: str | None = None,
                       categories: List[str] | None = None,
                       metadata: Dict | None = None) -> "CategoricalFitness":
        """Build a categorical layer from values indexed by sequence position.

        Parameters
        ----------
        name : str
            Layer name.
        mapping : mapping of int to str
            Categories keyed by zero-based sequence index.
        length : int
            Total number of sequences in the output layer.
        default : str, optional
            Category assigned to indices absent from ``mapping``.
        categories : list of str, optional
            Ordered allowed categories. If omitted, infer them from values.
        metadata : dict, optional
            Free-form layer metadata.

        Returns
        -------
        CategoricalFitness
            Constructed categorical layer.

        Raises
        ------
        ValueError
            If an index is missing and ``default`` is not provided.
        """
        vals: List[str] = []
        for i in range(length):
            v = mapping.get(i, default)
            if v is None:
                raise ValueError(f"Missing category at index {i} and no default provided")
            vals.append(str(v))
        return cls(name=name, values=vals, categories=categories, metadata=metadata)

    @classmethod
    def random(cls,
               name: str,
               *,
               length: int,
               categories: List[str],
               p: List[float] | None = None,
               seed: int | None = None,
               metadata: Dict | None = None) -> "CategoricalFitness":
        """
        Constructor method for a random categorical fitness layer.

        Parameters
        ----------
        name : str
            The layer name.
        
        length : int
            The number of sequences described by the fitness layer.

        categories : List[str], default=`None`
            A list of unique categories that values can take. If not
            provided, categories are derived from the `values` list.

        p : List[float], default=`None`
            A list of probabilities for each category. If not provided,
            uniform distribution is assumed (equal probability for each
            category).

        seed : int, optional
            Random seed for reproducibility.

        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        CategoricalFitness
            An instance of the CategoricalFitness class initialized
            with the provided parameters.
        """
        rng = np.random.default_rng(seed)
        cats = _validated_categories(categories)
        if p is None:
            p = [1.0 / len(cats)] * len(cats)
        probabilities = np.asarray(p, dtype=float)
        if probabilities.shape != (len(cats),):
            raise ValueError("p must provide one probability per category")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("p must contain finite non-negative probabilities")
        if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("p probabilities must sum to one")
        idx = rng.choice(len(cats), size=length, p=probabilities)
        vals = [cats[i] for i in idx]
        return cls(name=name, values=vals, categories=cats, metadata=metadata)

class ProbabilisticCategoricalFitness(BaseFitnessLayer):
    """Represent a categorical probability distribution per sequence.

    Parameters
    ----------
    name : str
        Layer name.
    probabilities : ndarray
        Finite non-negative matrix with shape ``(n_sequences, n_categories)``.
        Rows must sum to one within an absolute tolerance of ``1e-8``.
    categories : list of str
        Ordered category names corresponding to matrix columns.
    metadata : dict, optional
        Free-form layer metadata.

    Attributes
    ----------
    name : str
        The name of the fitness layer.
    probabilities : np.ndarray
        A 2D array of shape (num_sequences, num_categories) where each row
        is a probability distribution over the categories.
    categories : List[str]
        The ordered list of all possible categories.
    metadata : Dict, optional
        Additional metadata associated with the fitness layer.
    """
    _NORMALIZATION_ATOL = 1e-8

    def __init__(self,
                 name: str,
                 probabilities: np.ndarray,
                 categories: List[str],
                 metadata: Dict = None) -> None:
        super().__init__(name=name, metadata=metadata)

        categories = _validated_categories(categories)
        probabilities = _as_float_matrix(probabilities, name="probabilities")
        if probabilities.shape[1] != len(categories):
            raise ValueError("Shape of probabilities matrix must match the number of categories.")
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probabilities must contain only finite values")
        if np.any(probabilities < 0.0):
            raise ValueError("probabilities must contain non-negative values")
        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=self._NORMALIZATION_ATOL,
        ):
            raise ValueError("Rows in the probabilities matrix must sum to 1.")

        self._probabilities = probabilities
        self._probabilities.setflags(write=False)
        self._categories = tuple(categories)
        self.category_map = {cat: i for i, cat in enumerate(self._categories)}

    @property
    def probabilities(self) -> np.ndarray:
        """Return a read-only view of the probability matrix."""
        view = self._probabilities.view()
        view.setflags(write=False)
        return view

    @property
    def categories(self) -> list[str]:
        """Return a defensive copy of the ordered categories."""
        return list(self._categories)

    @property
    def dtype(self) -> Literal['categorical']:
        """Return the layer kind, ``'categorical'``."""
        return 'categorical'

    def get_tensor(self) -> torch.Tensor:
        """
        Method to convert the fitness layer to a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            A tensor representation of the probabilities, where each row
            corresponds to a sequence and each column corresponds to a
            category. The tensor will have shape (num_sequences,
            num_categories).
        """
        torch = _torch_module()
        return torch.tensor(self.probabilities, dtype=torch.float32)

    def get_value(self,
                  sequence_index: int) -> Dict[str, float]:
        """
        Returns the full probability distribution for a single sequence.
        
        Parameters
        ----------
        sequence_index : int
            The index of the sequence for which to retrieve the
            probability distribution.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping each category to its probability for
            the specified sequence.
        """
        return {cat: self.probabilities[sequence_index, i] for i, cat in enumerate(self.categories)}

    def to_scalar(self,
                  rank_map: Dict[str, int] = None) -> np.ndarray:
        """
        Converts to a scalar by returning the integer rank of the most
        probable category (the mode of the posterior).

        Parameters
        ----------
        rank_map : Dict[str, int], optional
            A mapping from category names to integer ranks. If not provided,
            the default category order will be used, which is the order in
            which categories were defined in the layer.

        Returns
        -------
        np.ndarray
            An array of integer ranks corresponding to the most likely
            category for each sequence. Each value in the layer is replaced
            by its rank according to the provided rank map. If a value is not
            found in the rank map, it will raise a ValueError.
        """
        _rank_map = rank_map or self.category_map
        # Find the index of the most likely category for each sequence
        most_likely_indices = np.argmax(self.probabilities, axis=1)
        # Convert these indices back to category names
        most_likely_categories = [self.categories[i] for i in most_likely_indices]
        # Map the most likely categories to their ranks
        return np.array([_rank_map[cat] for cat in most_likely_categories])
    
    def __len__(self):
        return int(self.probabilities.shape[0])
    
    @classmethod
    def from_probabilities(cls,
                           name: str,
                           probabilities: Union[np.ndarray, torch.Tensor],
                           *,
                           categories: List[str],
                           metadata: Dict | None = None) -> "ProbabilisticCategoricalFitness":
        """
        Constructor method to build a probabilistic categorical fitness
        layer from a 2D array or tensor of probabilities.

        Parameters
        ----------
        name : str
            The layer name.
        
        probabilities : np.ndarray OR torch.Tensor
            A 2D array or tensor of shape (num_sequences,
            num_categories). Each row corresponds to a sequence and
            each column corresponds to a category.

        categories : List[str]
            A list of unique categories that values can take. The order
            of categories must match the columns of the probabilities
            matrix.
            
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        ProbabilisticCategoricalFitness
            An instance of the ProbabilisticCategoricalFitness class.
        """
        arr = _as_float_matrix(probabilities, name="probabilities")
        return cls(
            name=name,
            probabilities=arr,
            categories=categories,
            metadata=metadata,
        )

    @classmethod
    def from_logits(cls,
                    name: str,
                    logits: Union[np.ndarray, torch.Tensor],
                    *,
                    categories: List[str],
                    metadata: Dict | None = None) -> "ProbabilisticCategoricalFitness":
        
        """
        Constructor method to build a probabilistic categorical fitness
        layer from an array of logits. Functional alias to
        `from_probabilities` that computes probabilities from logits.

        Parameters
        ----------
        name : str
            The layer name.
        
        logits : np.ndarray OR torch.Tensor
            A 2D array or tensor of shape (num_sequences,
            num_categories). Each row corresponds to a sequence and
            each column corresponds to a category.

        categories : List[str]
            A list of unique categories that values can take. The order
            of categories must match the columns of the probabilities
            matrix.
            
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        ProbabilisticCategoricalFitness
            An instance of the ProbabilisticCategoricalFitness class.
        """
        categories = _validated_categories(categories)
        z = _as_float_matrix(logits, name="logits")
        if z.shape[1] != len(categories):
            raise ValueError("logits width must match categories")
        if not np.all(np.isfinite(z)):
            raise ValueError("logits must contain only finite values")
        z = z - z.max(axis=1, keepdims=True)  # numerical stability
        exp = np.exp(z)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return cls.from_probabilities(name=name, probabilities=probs, categories=categories, metadata=metadata)

    @classmethod
    def from_counts(cls,
                    name: str,
                    counts: Union[np.ndarray, torch.Tensor],
                    *,
                    categories: List[str],
                    alpha: float = 0.0,
                    metadata: Dict | None = None) -> "ProbabilisticCategoricalFitness":
        """
        Constructor method to build a probabilistic categorical fitness
        layer from a counts matrix. Each row corresponds to a sequence
        and each column corresponds to a category. Optionally applies
        Laplace smoothing with parameter `alpha`.

        Parameters
        ----------
        name : str
            The layer name.
        
        counts : np.ndarray OR torch.Tensor
            A 2D array or tensor of shape (num_sequences,
            num_categories) representing counts of each category for
            each sequence. Each row corresponds to a sequence and each
            column corresponds to a category.

        categories : List[str]
            A list of unique categories that values can take. The order
            of categories must match the columns of the probabilities
            matrix.

        alpha : float, default=`0.0`
            Laplace smoothing parameter. If greater than 0, adds `alpha`
            to each count before normalizing to probabilities.
            
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        ProbabilisticCategoricalFitness
            An instance of the ProbabilisticCategoricalFitness class.
        """
        categories = _validated_categories(categories)
        c = _as_float_matrix(counts, name="counts")
        if c.shape[1] != len(categories):
            raise ValueError("counts width must match categories")
        if not np.all(np.isfinite(c)):
            raise ValueError("counts must contain only finite values")
        if np.any(c < 0.0):
            raise ValueError("counts must contain non-negative values")
        if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be a finite non-negative scalar")
        c = c + float(alpha)
        row_sums = c.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0.0):
            raise ValueError("count rows must have positive mass after smoothing")
        c = c / row_sums
        return cls.from_probabilities(name=name, probabilities=c, categories=categories, metadata=metadata)

    @classmethod
    def from_samples(cls,
                     name: str,
                     samples: List[List[str]],
                     *,
                     categories: List[str],
                     metadata: Dict | None = None) -> "ProbabilisticCategoricalFitness":
        """
        Constructor method to build a probabilistic categorical fitness
        layer from a list of samples. Each sample is a list of categories
        for a sequence, and the counts of each category are computed.

        Parameters
        ----------
        name : str
            The layer name.
        
        samples : List[List[str]]
            A list of samples where each sample is a list of categories
            for a sequence. Each inner list corresponds to a sequence.

        categories : List[str]
            A list of unique categories that values can take. The order
            of categories must match the columns of the probabilities
            matrix.
            
        metadata : Dict, optional
            Additional metadata associated with the fitness layer.

        Returns
        -------
        ProbabilisticCategoricalFitness
            An instance of the ProbabilisticCategoricalFitness class.
        """
        cats = _validated_categories(categories)
        idx_map = {c: i for i, c in enumerate(cats)}
        num_seq = len(samples)
        num_cat = len(cats)
        C = np.zeros((num_seq, num_cat), dtype=float)
        for i, row in enumerate(samples):
            for s in row:
                if s not in idx_map:
                    raise ValueError(f"Unknown category '{s}'")
                C[i, idx_map[s]] += 1.0
        return cls.from_counts(name=name, counts=C, categories=cats, metadata=metadata)


# TODO: wrapper classes for fitness layers modifiers.

class BaseFitnessWrapper(BaseFitnessLayer):
    """
    """
    def __init__(self,
                 layer: BaseFitnessLayer,
                 **kwargs):
        super().__init__(name=kwargs.get('name', layer.name),
                         metadata=kwargs.get('metadata',layer.metadata))
        # TODO: Add callable transofmrmations to the wrapper.
        self._wrapped_layer = layer

    @property
    def dtype(self):
        return self._wrapped_layer.dtype

    # Delegate core methods to the wrapped layer
    def get_tensor(self):
        return self._wrapped_layer.get_tensor()

    def to_scalar(self, **kwargs):
        return self._wrapped_layer.to_scalar(**kwargs)
    
# Modifier interfaces

class BaseFitnessModifier(ABC):
    """
    Base class for objects that transform one fitness layer into
    another. Implementors only need to override `apply`; validation and
    naming defaults are handled here.
    """

    # tuple of acceptable input dtypes reported by BaseFitnessLayer.dtype
    input_dtypes: tuple[str, ...] = ("numeric", "categorical")
    # short identifier appended to default output names
    modifier_name: str = "modifier"

    def __init__(self, *, name: str | None = None):
        self._custom_name = name

    def default_name(self, source_name: str) -> str:
        return f"{source_name}_{self.modifier_name}"

    def __call__(self, layer: BaseFitnessLayer, *, name: str | None = None) -> BaseFitnessLayer:
        if layer.dtype not in self.input_dtypes:
            raise TypeError(
                f"{self.__class__.__name__} only accepts layers with dtype in {self.input_dtypes}, "
                f"got '{layer.dtype}'."
            )
        target_name = name or self._custom_name or self.default_name(layer.name)
        return self.apply(layer, name=target_name)

    @abstractmethod
    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        """
        Transform `layer` into a new BaseFitnessLayer. Implementations
        must set the returned layer's name to `name`.
        """
        raise NotImplementedError


FitnessModifierLike = Union[
    BaseFitnessModifier,
    Callable[[BaseFitnessLayer], BaseFitnessLayer],
]

def apply_fitness_modifier(layer: BaseFitnessLayer,
                           modifier: FitnessModifierLike,
                           *,
                           name: str | None = None) -> BaseFitnessLayer:
    """
    Run a modifier (object or simple callable) on a fitness layer and
    return the transformed layer.
    """
    if isinstance(modifier, BaseFitnessModifier):
        return modifier(layer, name=name)

    if callable(modifier):
        result = modifier(layer)
        if not isinstance(result, BaseFitnessLayer):
            raise TypeError("Fitness modifiers must return a BaseFitnessLayer instance.")
        if name is not None:
            result.name = name
        return result

    raise TypeError(f"Unsupported modifier type: {type(modifier)!r}")


class EntropyFitnessModifier(BaseFitnessModifier):
    """
    Convert a probabilistic categorical fitness layer into a numeric
    fitness layer where each value is the entropy of the input
    distribution.
    """

    modifier_name = "entropy"
    input_dtypes = ("categorical",)

    def __init__(self, *, base: float | None = None, name: str | None = None):
        """
        Parameters
        ----------
        base : float, optional
            Logarithm base for entropy. When None, natural logarithm is
            used (consistent with scipy.stats.entropy default).
        name : str, optional
            Optional explicit name for the output layer.
        """
        super().__init__(name=name)
        self.base = base

    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        if not isinstance(layer, ProbabilisticCategoricalFitness):
            raise TypeError(
                "EntropyFitnessModifier expects a ProbabilisticCategoricalFitness layer."
            )
        probs = layer.probabilities
        ent = stats.entropy(probs, axis=1, base=self.base)
        meta = dict(layer.metadata) if getattr(layer, "metadata", None) else {}
        meta.update(
            {
                "modifier": "entropy",
                "source_layer": layer.name,
                "input_categories": list(layer.categories),
                "base": self.base,
            }
        )
        return NumericFitness.from_scalars(name=name, values=ent, metadata=meta)


class ProbabilitySliceFitnessModifier(BaseFitnessModifier):
    """
    Extract the probability of a specific category/index from a
    probabilistic categorical fitness layer and emit it as a numeric
    fitness layer.
    """

    modifier_name = "probability"
    input_dtypes = ("categorical",)

    def __init__(self, category: int | str, *, name: str | None = None):
        """
        Parameters
        ----------
        category : int or str
            Category index (int) or category label (str) to extract.
        name : str, optional
            Optional explicit name for the output layer.
        """
        super().__init__(name=name)
        self.category = category

    def _resolve_index(self, layer: ProbabilisticCategoricalFitness) -> tuple[int, str]:
        if isinstance(self.category, int):
            idx = int(self.category)
            if idx < 0 or idx >= len(layer.categories):
                raise IndexError(
                    f"Category index {idx} out of range for {len(layer.categories)} categories."
                )
            return idx, layer.categories[idx]

        label = str(self.category)
        if label not in layer.category_map:
            raise KeyError(f"Unknown category label '{label}' for layer '{layer.name}'.")
        return layer.category_map[label], label

    def default_name(self, source_name: str) -> str:
        if isinstance(self.category, str):
            return f"{source_name}_prob_{self.category}"
        return f"{source_name}_prob_{int(self.category)}"

    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        if not isinstance(layer, ProbabilisticCategoricalFitness):
            raise TypeError(
                "ProbabilitySliceFitnessModifier expects a ProbabilisticCategoricalFitness layer."
            )
        idx, label = self._resolve_index(layer)
        probs = layer.probabilities[:, idx]
        meta = dict(layer.metadata) if getattr(layer, "metadata", None) else {}
        meta.update(
            {
                "modifier": "probability_slice",
                "source_layer": layer.name,
                "target_index": idx,
                "target_category": label,
                "input_categories": list(layer.categories),
            }
        )
        return NumericFitness.from_scalars(name=name, values=probs, metadata=meta)


def _numeric_to_scalar(layer: BaseFitnessLayer,
                       aggregate_func: Callable | None = None) -> np.ndarray:
    if aggregate_func is None:
        return layer.to_scalar()
    try:
        return layer.to_scalar(aggregate_func=aggregate_func)
    except TypeError:
        return layer.to_scalar()


class GaussianNoiseFitnessModifier(BaseFitnessModifier):
    """
    Add Gaussian noise to a numeric fitness layer.
    """

    modifier_name = "gaussian_noise"
    input_dtypes = ("numeric",)

    def __init__(self,
                 *,
                 scale: float = 1.0,
                 loc: float = 0.0,
                 seed: int | None = None,
                 name: str | None = None):
        """
        Parameters
        ----------
        scale : float, default=`1.0`
            Standard deviation of the Gaussian noise.
        loc : float, default=`0.0`
            Mean of the Gaussian noise.
        seed : int, optional
            Random seed for reproducibility.
        name : str, optional
            Optional explicit name for the output layer.
        """
        super().__init__(name=name)
        if scale < 0:
            raise ValueError("scale must be non-negative.")
        self.scale = float(scale)
        self.loc = float(loc)
        self.seed = seed

    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        rng = np.random.default_rng(self.seed)
        meta = dict(layer.metadata) if getattr(layer, "metadata", None) else {}
        meta.update(
            {
                "modifier": "gaussian_noise",
                "source_layer": layer.name,
                "loc": self.loc,
                "scale": self.scale,
                "seed": self.seed,
            }
        )

        if isinstance(layer, NumericFitness):
            reps: List[List[float]] = []
            for i in range(len(layer)):
                r = np.asarray(layer.get_value(i), dtype=float)
                if r.size == 0:
                    r = np.array([float("nan")])
                noise = rng.normal(loc=self.loc, scale=self.scale, size=r.shape)
                mask = np.isnan(r)
                r = r + noise
                if mask.any():
                    r[mask] = np.nan
                reps.append(r.tolist())
            return NumericFitness.from_replicates(name=name, replicates=reps, metadata=meta)

        values = _numeric_to_scalar(layer)
        noise = rng.normal(loc=self.loc, scale=self.scale, size=len(values))
        return NumericFitness.from_scalars(name=name, values=values + noise, metadata=meta)


class GaussianDistributionFitnessModifier(BaseFitnessModifier):
    """
    Convert scalar values into Gaussian replicate distributions.
    """

    modifier_name = "gaussian_distribution"
    input_dtypes = ("numeric",)

    def __init__(self,
                 *,
                 scale: float,
                 reps: int = 10,
                 seed: int | None = None,
                 aggregate_func: Callable = np.mean,
                 name: str | None = None):
        """
        Parameters
        ----------
        scale : float
            Standard deviation of the Gaussian distribution.
        reps : int, default=`10`
            Number of replicates to sample per sequence.
        seed : int, optional
            Random seed for reproducibility.
        aggregate_func : callable, optional
            Aggregator for input replicates when reducing to scalars.
        name : str, optional
            Optional explicit name for the output layer.
        """
        super().__init__(name=name)
        if scale < 0:
            raise ValueError("scale must be non-negative.")
        if reps <= 0:
            raise ValueError("reps must be a positive integer.")
        if not callable(aggregate_func):
            raise TypeError("aggregate_func must be callable.")
        self.scale = float(scale)
        self.reps = int(reps)
        self.seed = seed
        self.aggregate_func = aggregate_func

    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        values = np.asarray(_numeric_to_scalar(layer, self.aggregate_func), dtype=float).ravel()
        rng = np.random.default_rng(self.seed)
        samples = rng.normal(loc=values[:, None], scale=self.scale, size=(len(values), self.reps))
        meta = dict(layer.metadata) if getattr(layer, "metadata", None) else {}
        meta.update(
            {
                "modifier": "gaussian_distribution",
                "source_layer": layer.name,
                "scale": self.scale,
                "reps": self.reps,
                "seed": self.seed,
                "aggregate_func": getattr(self.aggregate_func, "__name__", repr(self.aggregate_func)),
            }
        )
        return NumericFitness.from_tensor(name=name, tensor=samples, metadata=meta)


class ResampleFitnessModifier(BaseFitnessModifier):
    """
    Resample values from a distribution defined by numeric replicates.
    """

    modifier_name = "resample"
    input_dtypes = ("numeric",)

    def __init__(self,
                 *,
                 reps: int = 1,
                 seed: int | None = None,
                 name: str | None = None):
        """
        Parameters
        ----------
        reps : int, default=`1`
            Number of resampled replicates per sequence.
        seed : int, optional
            Random seed for reproducibility.
        name : str, optional
            Optional explicit name for the output layer.
        """
        super().__init__(name=name)
        if reps <= 0:
            raise ValueError("reps must be a positive integer.")
        self.reps = int(reps)
        self.seed = seed

    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        if not isinstance(layer, NumericFitness):
            raise TypeError("ResampleFitnessModifier expects a NumericFitness layer.")

        rng = np.random.default_rng(self.seed)
        reps: List[List[float]] = []
        for i in range(len(layer)):
            r = np.asarray(layer.get_value(i), dtype=float)
            if r.size == 0:
                reps.append([float("nan")] * self.reps)
                continue
            if len(r) > 1:
                loc = float(np.mean(r))
                scale = float(np.std(r))
            else:
                loc = float(r[0])
                scale = 0.0
            samples = rng.normal(loc=loc, scale=scale, size=self.reps)
            reps.append(samples.tolist())

        meta = dict(layer.metadata) if getattr(layer, "metadata", None) else {}
        meta.update(
            {
                "modifier": "resample",
                "source_layer": layer.name,
                "reps": self.reps,
                "seed": self.seed,
            }
        )
        return NumericFitness.from_replicates(name=name, replicates=reps, metadata=meta)


class ArithmeticFitnessModifier(BaseFitnessModifier):
    """
    Perform arithmetic operations between numeric fitness layers.
    """

    modifier_name = "arithmetic"
    input_dtypes = ("numeric",)

    def __init__(self,
                 other_layers: Sequence[BaseFitnessLayer] | BaseFitnessLayer,
                 *,
                 op: str | Callable = "add",
                 aggregate_func: Callable = np.mean,
                 name: str | None = None):
        """
        Parameters
        ----------
        other_layers : BaseFitnessLayer or sequence of BaseFitnessLayer
            Additional layers to combine with the source layer.
        op : str or callable, default=`"add"`
            Arithmetic operation to apply. Built-ins: "add", "sub",
            "mul", "div". If callable, it should accept arrays from each
            layer as positional arguments and return an array.
        aggregate_func : callable, optional
            Aggregator for input replicates when reducing to scalars.
        name : str, optional
            Optional explicit name for the output layer.
        """
        super().__init__(name=name)
        if isinstance(other_layers, BaseFitnessLayer):
            layers = [other_layers]
        else:
            layers = list(other_layers)
        if not layers:
            raise ValueError("ArithmeticFitnessModifier requires at least one other layer.")
        if not callable(aggregate_func):
            raise TypeError("aggregate_func must be callable.")

        self.other_layers = layers
        self.op = op
        self.aggregate_func = aggregate_func

    def _resolve_operation(self) -> Callable:
        if callable(self.op):
            return self.op
        ops = {
            "add": operator.add,
            "sub": operator.sub,
            "subtract": operator.sub,
            "mul": operator.mul,
            "multiply": operator.mul,
            "div": operator.truediv,
            "divide": operator.truediv,
        }
        if isinstance(self.op, str) and self.op in ops:
            return ops[self.op]
        raise ValueError(f"Unsupported operation: {self.op!r}")

    def apply(self, layer: BaseFitnessLayer, *, name: str) -> BaseFitnessLayer:
        for other in self.other_layers:
            if other.dtype not in self.input_dtypes:
                raise TypeError(
                    f"ArithmeticFitnessModifier only accepts layers with dtype in {self.input_dtypes}, "
                    f"got '{other.dtype}'."
                )
            other._validate_length(len(layer), name="arithmetic modifier")

        base_values = np.asarray(_numeric_to_scalar(layer, self.aggregate_func), dtype=float).ravel()
        other_values = [
            np.asarray(_numeric_to_scalar(other, self.aggregate_func), dtype=float).ravel()
            for other in self.other_layers
        ]

        op = self._resolve_operation()
        if callable(self.op) and not isinstance(self.op, str):
            result = op(base_values, *other_values)
        else:
            result = reduce(op, other_values, base_values)

        result = np.asarray(result, dtype=float).ravel()
        if result.shape[0] != len(layer):
            raise ValueError(
                "ArithmeticFitnessModifier expects operation output length "
                f"{len(layer)}, got {result.shape[0]}."
            )

        meta = dict(layer.metadata) if getattr(layer, "metadata", None) else {}
        if isinstance(self.op, str):
            op_label: str | None = self.op
        else:
            op_label = getattr(self.op, "__name__", repr(self.op))
        meta.update(
            {
                "modifier": "arithmetic",
                "source_layer": layer.name,
                "other_layers": [l.name for l in self.other_layers],
                "operation": op_label,
                "aggregate_func": getattr(self.aggregate_func, "__name__", repr(self.aggregate_func)),
            }
        )
        return NumericFitness.from_scalars(name=name, values=result, metadata=meta)
    
# Batch factory functions

if TYPE_CHECKING:
    FitnessLike = Union[
        BaseFitnessLayer,
        List[float],
        List[List[float]],
        np.ndarray,
        torch.Tensor,
    ]
else:
    FitnessLike = Any

def make_fitness_layer(name: str,
                       obj: FitnessLike,
                       *,
                       dtype: Literal["numeric", "categorical", "auto"] = "auto",
                       categories: List[str] | None = None,
                       metadata: Dict | None = None) -> BaseFitnessLayer:
    """
    Factory function to coerce an object into a BaseFitnessLayer.

    Parameters
    ----------
    name : str
        The name of the fitness layer.
    
    obj : FitnessLike  
        The object to coerce into a fitness layer. Can be:
        - BaseFitnessLayer instance
        - 1-D numeric list or array (scalars)
        - 2-D numeric list or array (matrix)
        - 2-D probabilities with categories provided
        - 2-D one-hot encoded with categories
        - List of lists of floats (replicates)

    dtype : str, default=`"auto"`
        The expected data type of the fitness layer. Options are:
        - "numeric": Coerce to NumericFitness.
        - "categorical": Coerce to CategoricalFitness or ProbabilisticCategoricalFitness.
        - "auto": Infer based on the input structure.

    categories : List[str], optional
        A list of unique categories for categorical layers. Required if
        `dtype` is "categorical" or "auto" and the input is 2-D.

    metadata : Dict, optional
        Additional metadata associated with the fitness layer.

    Returns
    -------
    BaseFitnessLayer
        An instance of a fitness layer class based on the input object.
    """
    if isinstance(obj, BaseFitnessLayer):
        return obj

    is_tensor = _is_torch_tensor(obj)

    # Fast-path: replicate lists (ragged)
    if (
        dtype != "categorical"
        and isinstance(obj, list)
        and obj
        and isinstance(obj[0], (list, tuple, np.ndarray))
    ):
        return NumericFitness.from_replicates(name, obj, metadata=metadata)

    # Try to coerce numerically
    try:
        arr_num = obj.detach().cpu().numpy() if is_tensor else np.asarray(obj, dtype=float)
    except (TypeError, ValueError):
        arr_num = None

    if arr_num is not None:
        if arr_num.ndim == 1:
            return NumericFitness.from_scalars(name, arr_num, metadata=metadata)

        if arr_num.ndim == 2:
            if dtype == "categorical":
                if categories is None:
                    raise ValueError("categories required for categorical 2-D inputs")
                
                is_binary = np.all((arr_num == 0) | (arr_num == 1))
                row_sum = arr_num.sum(axis=1)
                is_one_hot = bool(is_binary and np.allclose(row_sum, 1.0))

                if is_one_hot:
                    return CategoricalFitness.from_one_hot(
                        name, arr_num, categories=categories, metadata=metadata
                    )

                if np.allclose(row_sum, 1.0, atol=1e-6):
                    return ProbabilisticCategoricalFitness.from_probabilities(
                        name, arr_num, categories=categories, metadata=metadata
                    )

                raise ValueError(
                    "Ambiguous 2-D categorical tensor; provide probabilities (rows sum≈1) "
                    "or a strict one-hot matrix."
                )

            # numeric matrix
            if dtype in ("numeric", "auto"):
                return NumericFitness.from_tensor(name, arr_num, metadata=metadata)

    # Fallback for non-numeric objects
    arr_obj = obj.detach().cpu().numpy() if is_tensor else np.asarray(obj, dtype=object)
    if arr_obj.ndim == 2 and dtype == "categorical":
        if categories is None:
            raise ValueError("categories required for categorical 2-D inputs")
        row_sum = np.asarray(arr_obj, dtype=float).sum(axis=1)
        if np.allclose(row_sum, 1.0, atol=1e-6):
            return ProbabilisticCategoricalFitness.from_probabilities(
                name, np.asarray(arr_obj, dtype=float), categories=categories, metadata=metadata
            )

    raise TypeError(f"Cannot infer fitness layer from object of type {type(obj)} with dtype='{dtype}'")

def as_fitness_layers(layers: Mapping[str, FitnessLike],
                      *,
                      categories: Mapping[str, List[str]] | None = None,
                      metadata: Mapping[str, Dict] | None = None) -> Dict[str, BaseFitnessLayer]:
    """
    Factory function to convert a mapping of layer names to
    FitnessLike objects into a mapping of layer names to
    BaseFitnessLayer instances.

    Parameters
    ----------
    layers : Mapping[str, FitnessLike]
        A mapping of layer names to objects that can be coerced into
        fitness layers. Each object can be a BaseFitnessLayer instance,
        a numeric list, a 2-D numeric array, or a categorical tensor.

    categories : Mapping[str, List[str]], optional
        A mapping of layer names to lists of unique categories for
        categorical layers. Required if the input is 2-D and `dtype` is
        "categorical" or "auto".

    metadata : Mapping[str, Dict], optional
        A mapping of layer names to additional metadata dictionaries
        associated with each fitness layer.

    Returns
    -------
    Dict[str, BaseFitnessLayer]
        A mapping of layer names to BaseFitnessLayer instances.
        Each object in the input mapping is coerced into a fitness layer
        based on its type and structure.
    """
    out: Dict[str, BaseFitnessLayer] = {}
    for name, obj in layers.items():
        cats = categories.get(name) if categories else None
        meta = metadata.get(name) if metadata else None
        # Heuristic: if `cats` provided, prefer categorical coercion
        dtype = "categorical" if cats is not None else "auto"
        out[name] = make_fitness_layer(name, obj, dtype=dtype, categories=cats, metadata=meta)
    return out
    
# TODO: Non linear composition modifiers.
# TODO: Temporal seascape modifiers / iterator.
