# Audit reachability of high-fitness variants

Combine an explicit fitness threshold, annotations, graph reachability, and
monotone-path checks to separate reachable, long-path, and unreachable targets. This workflow can be useful in studying what paths exist in the fitness landscape from a sequence to a maxima. 
## Input

Declare the wild type or training set, graph constructor, target threshold,
fitness units, tie rule, and component support before inspecting paths.

## Worked example

```python
# cookbook: test
import networkx as nx
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import find_greedy_accessible_paths
from fitness_landscape.core import AnnotationLayer, FitnessLandscape, NumericFitness

labels = ["wild_type", "bridge", "reachable_high", "remote_high", "remote_peer"]
sequences = [BinarySequence(f"{value:03b}", sequence_id=label) for value, label in enumerate(labels)]
graph = nx.Graph([("wild_type", "bridge"), ("bridge", "reachable_high"), ("remote_high", "remote_peer")])
for label, sequence in zip(labels, sequences):
    graph.nodes[label]["sequence"] = sequence

fitness = [0.1, 0.5, 0.9, 0.95, 0.8]
annotations = AnnotationLayer(
    "design",
    pd.DataFrame(
        {
            "split": ["train", "train", "test", "test", "test"],
            "is_wild_type": [True, False, False, False, False],
        }
    ),
)
landscape = FitnessLandscape(
    sequences,
    graph,
    fitness_layers={"assay": NumericFitness.from_scalars("assay", fitness)},
    annotation_layers={"design": annotations},
)
landscape.view("assay")

threshold = 0.8
source = "wild_type"
high_nodes = [
    node for node in graph.nodes
    if landscape.get_signal()[landscape.sequence_index_for_node(node)] >= threshold
]
audit = []
for node in high_nodes:
    reachable = nx.has_path(graph, source, node)
    distance = nx.shortest_path_length(graph, source, node) if reachable else None
    target_index = landscape.sequence_index_for_node(node)
    greedy = find_greedy_accessible_paths(
        landscape, sequences[0], sequences[target_index]
    )
    audit.append(
        {
            "target": node,
            "reachable": reachable,
            "shortest_observed_distance": distance,
            "strictly_increasing_paths": greedy["path_count"],
        }
    )

by_target = {row["target"]: row for row in audit}
assert by_target["reachable_high"] == {
    "target": "reachable_high",
    "reachable": True,
    "shortest_observed_distance": 2,
    "strictly_increasing_paths": 1,
}
assert by_target["remote_high"]["reachable"] is False
assert by_target["remote_peer"]["reachable"] is False
print(audit)
```

The audit distinguishes a reachable two-edge target from targets in another
observed component. Repeat it across predeclared graph views; graph sensitivity
is part of the result.

