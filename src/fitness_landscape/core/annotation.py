from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Hashable

import numpy as np
import pandas as pd


class AnnotationLayer:
    """
    Container that stores arbitrary per-sequence annotation data.

    Parameters
    ----------
    name :
        Layer identifier.
    data :
        Annotation records provided either as a pandas DataFrame or as a
        dictionary mapping column names to sequences of values. The number of
        rows must match the number of sequences associated with the landscape
        the layer will be attached to.
    metadata :
        Optional free-form metadata associated with the annotation layer.
    """

    def __init__(
        self,
        name: str,
        data: pd.DataFrame | Mapping[str, Sequence[Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.metadata = dict(metadata) if metadata else {}
        self._frame = self._coerce_to_dataframe(data)
        self._frame.columns = [str(c) for c in self._frame.columns]
        self._frame = self._frame.reset_index(drop=True)

    @staticmethod
    def _coerce_to_dataframe(
        data: pd.DataFrame | Mapping[str, Sequence[Any]]
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("AnnotationLayer cannot be built from an empty DataFrame.")
            return data.copy(deep=True)

        if not isinstance(data, Mapping):
            raise TypeError("`data` must be a pandas DataFrame or a mapping of columns to values.")

        if not data:
            raise ValueError("AnnotationLayer cannot be constructed from an empty mapping.")

        # Detect dictionary-of-dictionaries (records) versus columnar data.
        sample = next(iter(data.values()))
        if isinstance(sample, Mapping):
            frame = pd.DataFrame.from_dict(data, orient="index")
        else:
            frame = pd.DataFrame(data)

        if frame.empty:
            raise ValueError("Provided data produced an empty annotation frame.")

        # Ensure columnar inputs have equal length to avoid silent broadcasting.
        lengths: set[int] = set()
        for key, values in frame.items():
            lengths.add(len(values))
            if len(lengths) > 1:
                raise ValueError(
                    f"Column '{key}' length does not match previous columns in annotation data."
                )

        return frame.copy(deep=True)

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def columns(self) -> list[str]:
        return list(self._frame.columns)

    def to_dataframe(self, copy: bool = True) -> pd.DataFrame:
        """
        Return the annotation layer as a pandas DataFrame.
        """
        return self._frame.copy(deep=True) if copy else self._frame

    def get_record(self, index: int) -> dict[str, Any]:
        """
        Retrieve annotation values for a specific sequence index.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Annotation index {index} outside valid range 0..{len(self) - 1}."
            )
        return self._frame.iloc[index].to_dict()

    def validate_length(self, expected: int, *, context: str = "") -> None:
        """
        Ensure the layer length matches an expected number of sequences.
        """
        if len(self) != expected:
            label = f" for layer '{self.name}'" if self.name else ""
            extra = f" ({context})" if context else ""
            raise ValueError(
                f"Annotation length mismatch{label}: got {len(self)}, expected {expected}{extra}."
            )

    def query(
        self,
        criteria: Mapping[str, Any] | None = None,
        *,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Filter annotations by matching column values against a dictionary of criteria.

        The criteria mapping supports scalar equality and membership checks
        (when the criterion value is an iterable such as list, tuple, set,
        numpy array, or pandas Series).
        """
        if not criteria:
            return self.to_dataframe(copy=copy)

        mask = pd.Series(True, index=self._frame.index, dtype=bool)
        for column, requirement in criteria.items():
            if column not in self._frame.columns:
                raise KeyError(
                    f"Column '{column}' is not present in annotation layer '{self.name}'."
                )

            if _is_iterable(requirement):
                mask &= self._frame[column].isin(list(requirement))
            else:
                mask &= self._frame[column] == requirement

        result = self._frame.loc[mask]
        return result.copy(deep=True) if copy else result

    def matching_indices(self, criteria: Mapping[str, Any] | None = None) -> list[int]:
        """
        Return positional indices of records that satisfy the provided criteria.
        """
        filtered = self.query(criteria, copy=False)
        return filtered.index.to_list()


def _is_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (Sequence, set, pd.Series, np.ndarray))


def register_auto_annotation(
    graph,
    layer_name: str,
    records: Mapping[Hashable, Mapping[str, Any] | Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Attach per-node annotation specifications so that Landscape.build
    can materialise them later as full AnnotationLayer objects.
    """
    store = graph.graph.setdefault("_auto_annotations", {})
    formatted = {}
    for node, record in records.items():
        if isinstance(record, Mapping):
            formatted[node] = {str(k): v for k, v in record.items()}
        else:
            formatted[node] = {layer_name: record}
    store[layer_name] = {
        "records": formatted,
        "metadata": dict(metadata) if metadata else {},
    }
