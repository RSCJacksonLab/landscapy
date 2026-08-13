# Audit evaluation support without target leakage

Before model fitting, determine whether each held-out node has graph support
from training nodes. Component membership and nearest-train distance distinguish
interpolation from graph extrapolation; neither alone proves predictive validity.

## Install and input

```bash
python -m pip install "landscapy[ml]"
```

Construct the graph from sequence or other target-independent inputs. This
worked example preregisters `111` as outside the represented adjacency support,
so it remains an isolated query rather than being silently dropped.

## Worked example

```python
# cookbook: test
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.core import AnnotationLayer, FitnessLandscape, NumericFitness, create_hamming_graph

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
sequences = [BinarySequence(text, sequence_id=f"toy-{i}") for i, text in enumerate(table.sequence)]
graph = create_hamming_graph(sequences)  # target values have not been loaded
graph.remove_edges_from(list(graph.edges(7)))
design = table[["split", "background"]].copy()
design["evaluation_cell"] = np.where(table.sequence == "111", "graph_extrapolation", "interpolation")
landscape = FitnessLandscape(
    sequences,
    graph,
    fitness_layers={"measured": NumericFitness.from_scalars("measured", table.fitness)},
    annotation_layers={"design": AnnotationLayer("design", design)},
)

split = landscape.get_annotation_layer("design").to_dataframe()["split"].to_numpy()
train_nodes = {
    landscape.node_for_sequence_index(i) for i, value in enumerate(split) if value == "train"
}
component_for_node = {}
component_report = []
for component_id, component in enumerate(nx.connected_components(landscape.graph)):
    for node in component:
        component_for_node[node] = component_id
    component_report.append(
        {
            "component": component_id,
            "nodes": len(component),
            "train": len(component & train_nodes),
            "test": sum(split[landscape.sequence_index_for_node(node)] == "test" for node in component),
        }
    )

test_report = []
for i, value in enumerate(split):
    if value != "test":
        continue
    node = landscape.node_for_sequence_index(i)
    component = nx.node_connected_component(landscape.graph, node)
    supported_train = component & train_nodes
    distance = min(
        (nx.shortest_path_length(landscape.graph, node, source) for source in supported_train),
        default=None,
    )
    test_report.append(
        {
            "sequence_id": sequences[i].id,
            "component": component_for_node[node],
            "nearest_train_distance": distance,
            "status": "estimable" if distance is not None else "unreachable",
        }
    )

assert sum(row["test"] for row in component_report) == 4
assert [row["status"] for row in test_report].count("unreachable") == 1
assert next(row for row in test_report if row["sequence_id"] == "toy-7")["nearest_train_distance"] is None
assert landscape.graph.degree[landscape.node_for_sequence_index(7)] == 0
print(component_report, test_report)
```

Report the unreachable row as a non-estimable evaluation cell. A random split
mostly measures interpolation; a support or scaffold split asks a different
question and should be reported separately, with uncertainty and sample count.

## Common failures

- Fitness values influence graph construction, split assignment, or feature selection.
- Test nodes outside training components are silently removed from metrics.
- Infinite graph distance is encoded as a large finite number.
- Random interpolation and graph-extrapolation scores are pooled.
- Component support is interpreted as evidence of biological generalization.
