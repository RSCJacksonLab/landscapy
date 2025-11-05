from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

import pandas as pd

from ...core.landscape import FitnessLandscape
from ...core.annotation import AnnotationLayer
from ..registry import AnnotationRegistry, PaletteStore


def import_pct_annotations(
    landscape: FitnessLandscape,
    clusters: pd.DataFrame,
    *,
    annotation_name: str = "pct_clusters",
    sequence_column: str = "Entry",
    registry: AnnotationRegistry | None = None,
    palette_store: PaletteStore | None = None,
    palette: Mapping[str, Any] | None = None,
    allow_missing: bool = True,
) -> AnnotationLayer:
    """
    Attach ProteinClusterTools cluster annotations to a landscape.

    Parameters
    ----------
    landscape :
        Target landscape.
    clusters :
        DataFrame containing at least a sequence identifier column (default
        ``"Entry"``) and any number of cluster level columns.
    annotation_name :
        Name assigned to the resulting annotation layer.
    sequence_column :
        Column in ``clusters`` containing sequence identifiers matching
        ``sequence.id`` in the landscape.
    registry :
        Optional annotation registry to update.
    palette_store :
        Optional palette store to register palettes for cross-tool consistency.
    palette :
        Optional palette payload (e.g., output of ProteinClusterTools'
        ``ColorAnnot``). Stored verbatim under a derived palette key.
    allow_missing :
        Whether to allow sequences in the landscape with no corresponding PCT
        annotation. Missing entries are filled with ``None``.
    """
    if sequence_column not in clusters.columns:
        raise KeyError(f"Column '{sequence_column}' missing from clusters DataFrame.")

    df = clusters.set_index(sequence_column).copy()

    layer = landscape.attach_annotation(
        name=annotation_name,
        data=df,
        map_by="name",
        allow_missing=allow_missing,
    )

    descriptor = None
    if registry is not None:
        descriptor = registry.register(annotation_name, layer, source="proteinclustertools")

    if palette is not None and palette_store is not None:
        palette_key = register_pct_palette(
            palette_store,
            annotation_name,
            palette,
        )
        if descriptor is not None:
            registry.update_palette(annotation_name, palette_key)

    return layer


def register_pct_palette(
    palette_store: PaletteStore,
    annotation_name: str,
    palette: Mapping[str, Any],
    *,
    palette_suffix: str = "pct",
) -> str:
    """
    Store a ProteinClusterTools palette dictionary in the palette store.

    Returns
    -------
    palette_key :
        Key under which the palette was saved.
    """
    key = f"{annotation_name}:{palette_suffix}"
    palette_store.register_palette(key, dict(palette))
    return key
