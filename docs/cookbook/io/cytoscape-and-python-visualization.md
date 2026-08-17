# Visualize a landscape with external tools

Landscapy exports graph and analysis data; Cytoscape, NetworkX, and Matplotlib
provide visualization. A plot is a diagnostic or communication device, not a
separate statistical result.

## Input

Use an undirected landscape small enough to draw legibly. Select annotation
layers and scalar fitness values intentionally before export.

## Worked example

```python
# cookbook: test
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from fitness_landscape.core import AnnotationLayer, BinarySequence, FitnessLandscape, NumericFitness
from fitness_landscape.analysis import calculate_ruggedness_dirichlet_energy

sequences = [
    BinarySequence(f"{value:03b}", sequence_id=f"s{value}") for value in range(8)
]
fitness = NumericFitness.from_scalars("assay", np.linspace(0.0, 1.0, 8))
annotations = AnnotationLayer(
    "design", pd.DataFrame({"split": ["train"] * 6 + ["test"] * 2})
)
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"assay": fitness},
    annotation_layers={"design": annotations},
)
landscape.view("assay")
energy = calculate_ruggedness_dirichlet_energy(landscape)

with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    xgmml = landscape.export_xgmml(
        tmp / "landscape.xgmml", annotation_layers=["design"]
    )
    text = xgmml.read_text()
    assert 'directed="0"' in text
    assert "fitness::assay" in text and "design::split" in text

    positions = nx.spring_layout(landscape.graph, seed=11)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    nx.draw_networkx(
        landscape.graph,
        positions,
        node_color=landscape.view("assay").to_scalar(),
        cmap="viridis",
        ax=axes[0],
    )
    axes[0].set_axis_off()
    axes[1].bar(["global", "per node"], [
        energy["global_dirichlet_energy"], energy["total_dirichlet_energy"]
    ])
    fig.tight_layout()
    figure = tmp / "diagnostic.png"
    fig.savefig(figure, dpi=100)
    plt.close(fig)
    assert figure.stat().st_size > 0

print(landscape.graph.number_of_nodes(), energy["global_dirichlet_energy"])
```

The XGMML file carries selected node attributes for Cytoscape. The Python plot
uses ordinary external APIs and returned numeric results; it is not a Landscapy
plotting interface. For large graphs, plot component/density summaries or use
an externally computed layout rather than drawing every edge.

## Common failures

- Every annotation is exported, including sensitive or irrelevant metadata.
- A stochastic layout is shown without a seed.
- Dense overplotting is interpreted as biological clustering.
- Node colours use a different row order from the graph.
- A visualization is presented without graph definition, scale, or denominator.
