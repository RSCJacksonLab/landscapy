from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..dataset import VisualizationDataset
from .color_utils import resolve_node_colours, rgba_to_plotly


def plot_landscape_plotly(
    dataset: VisualizationDataset,
    *,
    annotation_field: str | None = None,
    palette_key: str | None = None,
    cmap: str = "Viridis",
    node_size: int = 10,
    edge_color: str = "rgba(189,189,189,0.4)",
    edge_width: float = 1.0,
    show: bool = False,
):
    """Render a landscape using plotly."""

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
        cmap=cmap,
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
        field_name = annotation_field or next(iter(dataset.annotation_values))
        value_lookup = dataset.annotation_values.get(field_name, [])
        legend_labels = legend_labels or []
        known_labels = {label for label, _ in legend_labels if label != "Other"}

        for label, colour in legend_labels:
            if label == "Other":
                mask = np.asarray([val not in known_labels for val in value_lookup], dtype=bool)
            else:
                mask = np.asarray([val == label for val in value_lookup], dtype=bool)
            if not np.any(mask):
                continue
            colour_str = rgba_to_plotly(colour)
            fig.add_trace(
                go.Scattergl(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    marker=dict(size=node_size, color=colour_str, line=dict(color="black", width=0.5)),
                    name=str(label),
                    hoverinfo="text",
                    text=[str(dataset.nodes[i]) for i, keep in enumerate(mask) if keep],
                )
            )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=dataset.metadata.get("title", "Fitness Landscape"),
        legend=dict(title="Annotations" if not is_continuous else ""),
    )

    if show:
        fig.show()
    return fig
