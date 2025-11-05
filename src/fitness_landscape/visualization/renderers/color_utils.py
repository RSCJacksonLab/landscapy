from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..dataset import VisualizationDataset

RGBA = Tuple[float, float, float, float]


def resolve_node_colours(
    dataset: VisualizationDataset,
    *,
    annotation_field: str | None,
    palette_key: str | None,
    cmap: str,
) -> tuple[Sequence, list[tuple[str, RGBA]] | None, bool]:
    """
    Determine node colours for a dataset.

    Returns
    -------
    colours : Sequence
        Either numeric values (for continuous colouring) or RGBA tuples.
    legend : list[(label, RGBA)] | None
        Legend labels for categorical colouring.
    is_continuous : bool
        Flag indicating whether the colour mapping is numeric.
    """
    if dataset.annotation_values:
        field = annotation_field or next(iter(dataset.annotation_values))
        values = dataset.annotation_values[field]
        palette = dataset.palettes.get(palette_key) if palette_key else None
        colours, legend = _categorical_colours(values, palette=palette, cmap=cmap)
        return colours, legend, False

    if dataset.fitness_values is not None:
        return dataset.fitness_values, None, True

    default_colour: RGBA = (0.1216, 0.4667, 0.7059, 1.0)  # matplotlib default blue
    return [default_colour] * len(dataset.nodes), [("nodes", default_colour)], False


def _categorical_colours(
    values: Sequence,
    *,
    palette: Mapping | None,
    cmap: str,
) -> tuple[List[RGBA], list[tuple[str, RGBA]]]:
    if palette and "categories" in palette:
        categories_map = {
            label: _normalize_colour(colour)
            for label, colour in palette["categories"].items()
        }
        unknown_colour = _normalize_colour(palette.get("other", "#cccccc"))
        colours = [categories_map.get(val, unknown_colour) for val in values]
        legend = [(label, categories_map[label]) for label in categories_map]
        if any(val not in categories_map for val in values):
            legend.append(("Other", unknown_colour))
        return colours, legend

    uniques = [val for val in dict.fromkeys(values)]
    cmap_obj = plt.get_cmap(cmap, len(uniques) if uniques else 1)
    colour_map = {val: _to_rgba_tuple(cmap_obj(idx)) for idx, val in enumerate(uniques)}
    colours = [colour_map[val] for val in values]
    legend = [(str(label), colour_map[label]) for label in uniques]
    return colours, legend


def _normalize_colour(colour) -> RGBA:
    if isinstance(colour, str):
        lower = colour.lower()
        if lower.startswith("rgba(") and lower.endswith(")"):
            payload = lower[5:-1]
            parts = [float(p.strip()) for p in payload.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid RGBA colour specification: {colour!r}")
            scale = 255.0 if any(v > 1 for v in parts[:3]) else 1.0
            alpha_scale = 255.0 if parts[3] > 1 else 1.0
            return tuple(  # type: ignore[arg-type]
                [
                    parts[0] / scale,
                    parts[1] / scale,
                    parts[2] / scale,
                    parts[3] / alpha_scale,
                ]
            )
        if lower.startswith("rgb(") and lower.endswith(")"):
            payload = lower[4:-1]
            parts = [float(p.strip()) for p in payload.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid RGB colour specification: {colour!r}")
            scale = 255.0 if any(v > 1 for v in parts) else 1.0
            return tuple([parts[0] / scale, parts[1] / scale, parts[2] / scale, 1.0])  # type: ignore[arg-type]
        from matplotlib.colors import to_rgba

        return _to_rgba_tuple(to_rgba(colour))

    if isinstance(colour, (tuple, list)) and len(colour) in (3, 4):
        arr = np.asarray(colour, dtype=float)
        scale = 255.0 if np.any(arr > 1) else 1.0
        if len(arr) == 3:
            arr = np.append(arr, 1.0)
        return tuple(arr / scale)  # type: ignore[return-value]

    raise TypeError(f"Unsupported colour specification: {colour!r}")


def _to_rgba_tuple(colour) -> RGBA:
    arr = np.asarray(colour, dtype=float)
    if arr.shape[-1] == 3:
        arr = np.append(arr, 1.0)
    return tuple(arr.tolist())  # type: ignore[return-value]


def rgba_to_plotly(colour: RGBA) -> str:
    r, g, b, a = colour
    return f"rgba({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))},{a})"
