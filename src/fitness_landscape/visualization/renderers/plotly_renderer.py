from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Literal

import numpy as np

from ..dataset import VisualizationDataset
from .color_utils import resolve_node_colours, rgba_to_plotly


def plot_landscape_plotly(
    dataset: VisualizationDataset,
    *,
    annotation_field: str | None = None,
    palette_key: str | None = None,
    palette: Mapping[str, Any] | None = None,
    cmap: str = "Viridis",
    categorical_cmap: str = "Set2",
    color_by: Literal["auto", "annotation", "fitness"] = "auto",
    node_size: int = 10,
    edge_color: str = "rgba(189,189,189,0.4)",
    edge_width: float = 1.0,
    show: bool = False,
):
    """
    Render a landscape using plotly.

    Parameters
    ----------
    annotation_field :
        Optional annotation column to use for categorical colouring.
    palette_key, palette :
        Palette lookup key in ``dataset.palettes`` or a direct palette mapping.
    cmap :
        Colormap used for continuous colouring.
    categorical_cmap :
        Colormap used for categorical colouring when no explicit palette is supplied.
    color_by :
        Select colouring source: ``"annotation"``, ``"fitness"``, or ``"auto"``.
    """

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("plotly is required for interactive plotting.") from exc

    positions = dataset.positions
    x = positions[:, 0]
    y = positions[:, 1]

    colours, legend_labels, is_continuous = resolve_node_colours(
        dataset,
        annotation_field=annotation_field,
        palette_key=palette_key,
        palette=palette,
        cmap=cmap,
        categorical_cmap=categorical_cmap,
        color_by=color_by,
    )

    fig = go.Figure()

    if dataset.edges:
        edge_x: list[float] = []
        edge_y: list[float] = []
        node_index = {node: idx for idx, node in enumerate(dataset.nodes)}
        for u, v in dataset.edges:
            if u not in node_index or v not in node_index:
                continue
            edge_x.extend([x[node_index[u]], x[node_index[v]], None])
            edge_y.extend([y[node_index[u]], y[node_index[v]], None])
        fig.add_trace(
            go.Scattergl(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color=edge_color, width=edge_width),
                hoverinfo="skip",
                showlegend=False,
                name="edges",
            )
        )

    if is_continuous:
        colorscale = cmap
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=node_size,
                    color=np.asarray(colours, dtype=float),
                    colorscale=colorscale,
                    colorbar=dict(title=dataset.fitness_name or "fitness"),
                    line=dict(color="black", width=0.5),
                ),
                name="nodes",
                hoverinfo="text",
                text=[str(node) for node in dataset.nodes],
            )
        )
    else:
        colour_strings = [rgba_to_plotly(colour) for colour in colours]
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=node_size, color=colour_strings, line=dict(color="black", width=0.5)),
                name="nodes",
                hoverinfo="text",
                text=[str(node) for node in dataset.nodes],
                showlegend=False,
            )
        )

        for label, colour in legend_labels or []:
            fig.add_trace(
                go.Scattergl(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=node_size, color=rgba_to_plotly(colour), line=dict(color="black", width=0.5)),
                    name=str(label),
                    hoverinfo="skip",
                )
            )

    use_fitness_legend = color_by == "fitness" or (
        not dataset.annotation_values and color_by in {"auto", "annotation", "fitness"}
    )
    legend_title = ""
    if not is_continuous:
        legend_title = (dataset.fitness_name or "Fitness") if use_fitness_legend else (dataset.annotation_name or "Annotations")

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=dataset.metadata.get("title", "Fitness Landscape"),
        legend=dict(title=legend_title),
    )

    if show:
        fig.show()
    return fig
