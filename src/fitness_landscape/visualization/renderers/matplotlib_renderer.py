from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Literal

import matplotlib.pyplot as plt
import numpy as np

from ..dataset import VisualizationDataset
from .color_utils import resolve_node_colours


def plot_landscape_matplotlib(
    dataset: VisualizationDataset,
    *,
    annotation_field: str | None = None,
    palette_key: str | None = None,
    palette: Mapping[str, Any] | None = None,
    cmap: str = "viridis",
    categorical_cmap: str = "Set2",
    color_by: Literal["auto", "annotation", "fitness"] = "auto",
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
    palette :
        Direct palette mapping used for categorical colouring. Overrides
        ``palette_key`` when provided.
    cmap :
        Matplotlib colormap name used for continuous fitness colouring or for
        categorical fallback.
    categorical_cmap :
        Colormap to use for categorical colouring when no explicit palette is
        provided (defaults to ``"Set2"`` for fitness colouring).
    color_by :
        Select colouring source: ``"annotation"``, ``"fitness"``, or ``"auto"``.
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

    colours, legend_labels, is_continuous = resolve_node_colours(
        dataset,
        annotation_field=annotation_field,
        palette_key=palette_key,
        cmap=cmap,
        categorical_cmap=categorical_cmap,
        color_by=color_by,
        palette=palette,
    )

    scatter = ax.scatter(
        x,
        y,
        c=colours,
        s=node_size,
        cmap=cmap if is_continuous else None,
        edgecolors="black",
        linewidths=0.2,
        zorder=2,
    )

    if is_continuous and scatter.cmap is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(dataset.fitness_name or "fitness")

    if legend_labels and not is_continuous:
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
