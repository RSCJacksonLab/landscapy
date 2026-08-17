# Compare ruggedness across graph views

Before comparing values, establish matched nodes, component eligibility,
fitness scaling, edge count, and weight semantics for every graph.

## Input

This example uses the same six rows and a predeclared standardization for both
views. Standardization is appropriate only because the same assay and node set
are compared; it would not make unrelated phenotypes commensurate.

## Worked example

```python
# cookbook: test
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import calculate_ruggedness_dirichlet_energy, graph_properties
from fitness_landscape.core import FitnessLandscape, NumericFitness, create_hamming_graph, create_knn_graph

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
).iloc[:6]
sequences = [BinarySequence(text, sequence_id=f"s{i}") for i, text in enumerate(table["sequence"])]
fitness = table["fitness"].to_numpy(dtype=float)
standardized = (fitness - fitness.mean()) / fitness.std(ddof=0)

graphs = {
    "hamming": create_hamming_graph(sequences),
    "ohe_knn_k3": create_knn_graph(
        sequences,
        k=3,
        embedding_domain="ohe",
        backend="balltree",
        tie_policy="all",
    ),
}
rows = []
for name, graph in graphs.items():
    landscape = FitnessLandscape(
        sequences,
        graph,
        fitness_layers={"z_assay": NumericFitness.from_scalars("z_assay", standardized)},
    )
    landscape.view("z_assay")
    properties = graph_properties(graph)
    energy = calculate_ruggedness_dirichlet_energy(landscape)
    rows.append(
        {
            "view": name,
            "nodes": len(landscape),
            "edges": graph.number_of_edges(),
            "components": properties["components"]["count"],
            "eligible_nodes": len(landscape),
            "global_energy": energy["global_dirichlet_energy"],
            "per_node_energy": energy["total_dirichlet_energy"],
            "weight_key": energy["weight_key"],
        }
    )

coverage = pd.DataFrame(rows)
assert coverage["nodes"].nunique() == 1
assert coverage["components"].eq(1).all()
assert coverage.set_index("view").loc["hamming", "edges"] == 7
assert coverage.set_index("view").loc["ohe_knn_k3", "edges"] == 13
assert coverage["global_energy"].nunique() == 2
print(coverage.to_dict(orient="records"))
```

The matched standardized signal produces different energy because the kNN view
contains more and different edges. The table makes that denominator visible.

## Common failures

- Graphs use different node sets but are compared as matched views.
- Standardization hides scientifically meaningful target-unit differences.
- Global energy is compared without density or component coverage.
- Weighted and unweighted estimands share one result label.
- A graph is selected post hoc because it gives the preferred conclusion.
