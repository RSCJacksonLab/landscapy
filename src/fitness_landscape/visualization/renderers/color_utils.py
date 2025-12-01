from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple, Literal

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
    categorical_cmap: str = "Set2",
    color_by: Literal["auto", "annotation", "fitness"] = "auto",
    palette: Mapping[str, Any] | None = None,
) -> tuple[Sequence, list[tuple[str, RGBA]] | None, bool]:
    """
    Determine node colours for a dataset.

    Parameters
    ----------
    annotation_field :
        Optional annotation column to colour by when ``color_by="annotation"``.
    palette_key, palette :
        Palette lookup key in ``dataset.palettes`` or a direct palette mapping.
    cmap :
        Colormap used for continuous colouring.
    categorical_cmap :
        Colormap used when generating categorical colours without an explicit palette.
    color_by :
        Strategy for selecting colouring source: ``"annotation"``, ``"fitness"``, or
        ``"auto"`` (annotation preferred when available).

    Returns
    -------
    colours : Sequence
        Either numeric values (for continuous colouring) or RGBA tuples.
    legend : list[(label, RGBA)] | None
        Legend labels for categorical colouring.
    is_continuous : bool
        Flag indicating whether the colour mapping is numeric.
    """
    palette_obj = palette
    if palette_obj is None and palette_key:
        palette_obj = dataset.palettes.get(palette_key)

    mode = _resolve_colour_mode(dataset, color_by=color_by)

    if mode == "annotation" and dataset.annotation_values:
        field = annotation_field or next(iter(dataset.annotation_values))
        values = dataset.annotation_values[field]
        colours, legend = _categorical_colours(values, palette=palette_obj, cmap=cmap)
        return colours, legend, False

    if mode == "fitness" and (dataset.fitness_kind or dataset.fitness_values is not None):
        return _fitness_colours(
            dataset,
            palette=palette_obj,
            cmap=cmap,
            categorical_cmap=categorical_cmap,
        )

    return _default_colour_map(len(dataset.nodes))


def _resolve_colour_mode(
    dataset: VisualizationDataset, *, color_by: Literal["auto", "annotation", "fitness"]
) -> Literal["annotation", "fitness", "default"]:
    if color_by == "annotation":
        if dataset.annotation_values:
            return "annotation"
        if dataset.fitness_kind or dataset.fitness_values is not None:
            return "fitness"
        return "default"
    if color_by == "fitness":
        if dataset.fitness_kind or dataset.fitness_values is not None:
            return "fitness"
        if dataset.annotation_values:
            return "annotation"
        return "default"

    if dataset.annotation_values:
        return "annotation"
    if dataset.fitness_kind or dataset.fitness_values is not None:
        return "fitness"
    return "default"


def _fitness_colours(
    dataset: VisualizationDataset,
    *,
    palette: Mapping[str, Any] | None,
    cmap: str,
    categorical_cmap: str,
) -> tuple[Sequence, list[tuple[str, RGBA]] | None, bool]:
    kind = dataset.fitness_kind
    if kind is None and dataset.fitness_values is not None:
        kind = "numeric"

    if kind == "numeric":
        if dataset.fitness_values is not None:
            return dataset.fitness_values, None, True
        return _default_colour_map(len(dataset.nodes))

    if kind == "categorical":
        values = dataset.fitness_labels
        if values is None and dataset.fitness_values is not None:
            try:
                values = dataset.fitness_values.tolist()
            except AttributeError:
                values = list(dataset.fitness_values)
        if values is None:
            return _default_colour_map(len(dataset.nodes))
        colours, legend = _categorical_colours(values, palette=palette, cmap=categorical_cmap)
        return colours, legend, False

    if kind == "probabilistic":
        probs = dataset.fitness_probabilities
        categories = dataset.fitness_categories or []
        if probs is None or probs.shape[0] == 0 or not categories:
            return _default_colour_map(len(dataset.nodes))

        palette_map, legend = _palette_for_categories(categories, palette=palette, cmap=categorical_cmap)
        colours = _mix_probability_colours(probs, categories, palette_map)
        return colours, legend, False

    return _default_colour_map(len(dataset.nodes))


def _categorical_colours(
    values: Sequence,
    *,
    palette: Mapping | None,
    cmap: str,
) -> tuple[List[RGBA], list[tuple[str, RGBA]]]:
    palette_map: Mapping | None = None
    unknown_colour: RGBA | None = None
    if palette and isinstance(palette, Mapping):
        if "categories" in palette:
            palette_map = palette["categories"]
            unknown_colour = _normalize_colour(palette.get("other", "#cccccc"))
        else:
            palette_map = palette
            if hasattr(palette, "get"):  # type: ignore[truthy-bool]
                try:
                    unknown_colour = _normalize_colour(palette.get("other", "#cccccc"))  # type: ignore[call-arg]
                except Exception:
                    unknown_colour = _normalize_colour("#cccccc")
            else:
                unknown_colour = _normalize_colour("#cccccc")

    if palette_map is not None:
        categories_map = {
            label: _normalize_colour(colour)
            for label, colour in palette_map.items()
            if label != "other"
        }
        colours = [categories_map.get(val, unknown_colour or _normalize_colour("#cccccc")) for val in values]
        legend = [(str(label), categories_map[label]) for label in categories_map]
        if any(val not in categories_map for val in values):
            legend.append(("Other", unknown_colour or _normalize_colour("#cccccc")))
        return colours, legend

    uniques = [val for val in dict.fromkeys(values)]
    try:
        cmap_obj = plt.get_cmap(cmap, len(uniques) if uniques else 1)
    except ValueError:
        cmap_obj = plt.get_cmap(str(cmap).lower(), len(uniques) if uniques else 1)
    colour_map = {val: _to_rgba_tuple(cmap_obj(idx)) for idx, val in enumerate(uniques)}
    colours = [colour_map[val] for val in values]
    legend = [(str(label), colour_map[label]) for label in uniques]
    return colours, legend


def _palette_for_categories(
    categories: Sequence[str],
    *,
    palette: Mapping[str, Any] | None,
    cmap: str,
) -> tuple[dict[str, RGBA], list[tuple[str, RGBA]]]:
    """
    Build a deterministic mapping from category -> RGBA colour.
    """
    colours, legend = _categorical_colours(categories, palette=palette, cmap=cmap)
    colour_map = {cat: col for cat, col in zip(categories, colours)}
    # Ensure palette entries not referenced in categories are preserved in legend
    for label, colour in legend:
        colour_map.setdefault(label, colour)
    return colour_map, legend


def _mix_probability_colours(
    probabilities: np.ndarray,
    categories: Sequence[str],
    palette_map: Mapping[str, RGBA],
) -> List[RGBA]:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != len(categories):
        raise ValueError(
            f"Probability matrix shape {probs.shape} does not match number of categories ({len(categories)})"
        )
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = probs / row_sums
    base = np.zeros((probs.shape[0], 4), dtype=float)
    for idx, cat in enumerate(categories):
        colour = palette_map.get(cat)
        if colour is None:
            continue
        weight = probs[:, idx][:, None]
        base[:, :3] += weight * np.asarray(colour[:3], dtype=float)
        base[:, 3] += probs[:, idx] * colour[3]
    base[:, :3] = np.clip(base[:, :3], 0.0, 1.0)
    base[:, 3] = np.clip(base[:, 3], 0.0, 1.0)
    return [tuple(row.tolist()) for row in base]  # type: ignore[return-value]


def _default_colour_map(n: int) -> tuple[list[RGBA], list[tuple[str, RGBA]], bool]:
    default_colour: RGBA = (0.1216, 0.4667, 0.7059, 1.0)  # matplotlib default blue
    return [default_colour] * n, [("nodes", default_colour)], False


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
