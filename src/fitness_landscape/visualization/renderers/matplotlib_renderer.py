from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..dataset import VisualizationDataset


def plot_landscape_matplotlib(
    dataset: VisualizationDataset,
    *,
    annotation_field: str | None = None,
    palette_key: str | None = None,
    cmap: str = "viridis",
    node_size: int = 120,
    edge_color: str = "#bdbdbd",
    edge_alpha: float = 0.4,
    edge_linewidth: float = 1.0,
    ax: plt.Axes | None = None,
    show: bool = False,
):
    """
    Render a :class:`VisualizationDataset` using matplotlib.

    Parameters
    ----------
    dataset :
        Visualization dataset created by :class:`VisualizationDatasetBuilder`.
    annotation_field :
        Optional annotation column to use for categorical colouring. If not
        provided, defaults to the first available annotation column.
    palette_key :
        Optional key referencing a stored palette within ``dataset.palettes``.
        When provided and the palette contains a ``"categories"`` mapping, the
        associated colours are used for categorical rendering.
    cmap :
        Matplotlib colormap name used for continuous fitness colouring or for
        categorical fallback.
    node_size :
        Size of node markers passed to ``Axes.scatter``.
    edge_color, edge_alpha, edge_linewidth :
        Styling parameters for the drawn edges.
    ax :
        Optional matplotlib axes to draw on. When omitted, a figure and axes are
        created with ``subplots``.
    show :
        If ``True``, call ``plt.show()`` at the end of the rendering routine.

    Returns
    -------
    figure, axes :
        Tuple containing the matplotlib figure and axes used for rendering.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    positions = dataset.positions
    x = positions[:, 0]
    y = positions[:, 1]

    # Draw edges
    for u, v in dataset.edges:
        try:
            i = dataset.nodes.index(u)
            j = dataset.nodes.index(v)
        except ValueError:
            continue
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            color=edge_color,
            alpha=edge_alpha,
            linewidth=edge_linewidth,
            zorder=1,
        )

    colours, legend_labels = _resolve_node_colours(
        dataset,
        annotation_field=annotation_field,
        palette_key=palette_key,
        cmap=cmap,
    )

    scatter = ax.scatter(
        x,
        y,
        c=colours,
        s=node_size,
        cmap=cmap if np.issubdtype(np.asarray(colours).dtype, np.number) else None,
        edgecolors="black",
        linewidths=0.2,
        zorder=2,
    )

    if dataset.fitness_values is not None and legend_labels is None and scatter.cmap is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(dataset.fitness_name or "fitness")

    if legend_labels:
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=color,
                markeredgecolor="black",
                markersize=np.sqrt(node_size / np.pi),
            )
            for _, color in legend_labels
        ]
        ax.legend(handles, [label for label, _ in legend_labels], loc="best", frameon=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(dataset.metadata.get("title", "Fitness Landscape"))

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _resolve_node_colours(
    dataset: VisualizationDataset,
    *,
    annotation_field: str | None,
    palette_key: str | None,
    cmap: str,
) -> tuple[Sequence, list[tuple[str, str]] | None]:
    """
    Determine node colours and legends based on dataset contents.
    """
    if dataset.annotation_values:
        field = annotation_field or next(iter(dataset.annotation_values))
        values = dataset.annotation_values[field]
        palette = dataset.palettes.get(palette_key) if palette_key else None
        return _categorical_colours(values, palette=palette, cmap=cmap)

    if dataset.fitness_values is not None:
        return dataset.fitness_values, None

    return ["#1f77b4"] * len(dataset.nodes), None


def _categorical_colours(
    values: Sequence,
    *,
    palette: Mapping | None,
    cmap: str,
) -> tuple[List[str], list[tuple[str, str]]]:
    """
    Produce categorical colours (and legend labels) for annotation values.
    """
    if palette and "categories" in palette:
        categories_map = {
            label: _normalize_colour(colour) for label, colour in palette["categories"].items()
        }
        unknown_colour = _normalize_colour(palette.get("other", "#cccccc"))
        colours = [categories_map.get(val, unknown_colour) for val in values]
        legend_labels = [(label, categories_map[label]) for label in categories_map]
        if any(val not in categories_map for val in values):
            legend_labels.append(("Other", unknown_colour))
        return colours, legend_labels

    uniques = [val for val in dict.fromkeys(values)]
    cmap_obj = plt.get_cmap(cmap, len(uniques) if uniques else 1)
    colour_map = {val: cmap_obj(idx) for idx, val in enumerate(uniques)}
    colours = [colour_map[val] for val in values]
    legend_labels = [(str(label), colour_map[label]) for label in uniques]
    return colours, legend_labels


def _normalize_colour(colour) -> str | tuple[float, float, float, float]:
    """
    Convert various colour specifications (including ProteinClusterTools rgba strings)
    into a matplotlib-friendly representation.
    """
    if isinstance(colour, str):
        lower = colour.lower()
        if lower.startswith("rgba(") and lower.endswith(")"):
            payload = lower[5:-1]
            parts = [float(p.strip()) for p in payload.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid RGBA colour specification: {colour!r}")
            # In PCT, components are often 0-255 integers.
            scale = 255.0 if any(v > 1 for v in parts[:3]) else 1.0
            return tuple([parts[0] / scale, parts[1] / scale, parts[2] / scale, parts[3] / scale])
        if lower.startswith("rgb(") and lower.endswith(")"):
            payload = lower[4:-1]
            parts = [float(p.strip()) for p in payload.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid RGB colour specification: {colour!r}")
            scale = 255.0 if any(v > 1 for v in parts) else 1.0
            return tuple([parts[0] / scale, parts[1] / scale, parts[2] / scale, 1.0])
        return colour
    if isinstance(colour, (tuple, list)) and len(colour) in (3, 4):
        arr = np.asarray(colour, dtype=float)
        scale = 255.0 if np.any(arr > 1) else 1.0
        if len(arr) == 3:
            arr = np.append(arr, 1.0)
        return tuple(arr / scale)
    return colour
