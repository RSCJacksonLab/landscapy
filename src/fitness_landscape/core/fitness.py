from abc import ABC, abstractmethod
from typing import Dict, Literal, List, Any
import torch
import numpy as np
from scipy import stats

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

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"

    
class NumericFitness(BaseFitnessLayer):
    """
    Fitness layer that represents numeric fitness values as scalars and
    distributions based on replicate data.

    Attributes
    ----------
    name : str
        The name of the fitness layer.
    values : List[List[float]]
        A list of lists where each inner list contains replicate
        fitness values for a sequence.
    metadata : Dict, optional
        Additional metadata associated with the fitness layer.
    """
    def __init__(self,
                 name: str,
                 values: List[List[float]],
                 metadata: Dict = None) -> None:
        
        super().__init__(name=name, metadata=metadata)
        
        if not all(isinstance(r, list) for r in values):
            raise TypeError("Input 'values' must be a list of lists.")
        
        self._replicates = values
        # For each sequence, create a normal distribution based on its replicates
        self._distributions = [
            stats.norm(loc=np.mean(r), scale=np.std(r)) if len(r) > 1 else stats.norm(loc=r[0], scale=0)
            for r in self._replicates
        ]

    @property
    def dtype(self) -> Literal['numeric']:
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
                  sequence_index: int) -> Dict[str, float]:
        """
        Returns the full set of values for a single sequence.
        
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
        return self._replicates[sequence_index]


class CategoricalFitness(FitnessLayer):
    """
    Fitness layer that represents categorical fitness values.

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
        self._values = values
        
        if categories is None:
            self.categories = list(set(values))  # Unique categories from values
        else:
            self.categories = categories

        self.category_map = {cat: i for i, cat in enumerate(self.categories)}

        if not all(v in self.category_map for v in self._values):
            raise ValueError("All fitness 'values' must be present in the 'categories' list.")

    @property
    def dtype(self) -> Literal['categorical']:
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
                  sequence_index: int) -> Dict[str, float]:
        """
        Returns the full set of values for a single sequence.
        
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
        return self._values[sequence_index]
    


class ProbabilisticCategoricalFitness(BaseFitnessLayer):
    """
    Categorical fitness layer that represents probabilities
    of each category for each sequence.

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
    def __init__(self,
                 name: str,
                 probabilities: np.ndarray,
                 categories: List[str],
                 metadata: Dict = None) -> None:
        super().__init__(name=name, metadata=metadata)
        
        if probabilities.shape[1] != len(categories):
            raise ValueError("Shape of probabilities matrix must match the number of categories.")
        if not np.allclose(np.sum(probabilities, axis=1), 1.0):
            raise ValueError("Rows in the probabilities matrix must sum to 1.")
            
        self.probabilities = probabilities
        self.categories = categories
        self.category_map = {cat: i for i, cat in enumerate(self.categories)}

    @property
    def dtype(self) -> Literal['categorical']:
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
    
# TODO: Non linear composition modifiers.
# TODO: Temporal seascape modifiers / iterator.