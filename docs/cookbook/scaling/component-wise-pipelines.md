# Run component-wise pipelines with honest denominators

Inventory every connected component, declare eligibility before analysis, and
store skip reasons. Largest-component-only analysis changes the target
population whenever excluded nodes differ systematically.

## Input

The example has a four-node component, a two-node component, and an isolate.
The isolate is a held-out row with no training support.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import calculate_ruggedness_dirichlet_energy
from fitness_landscape.core import AnnotationLayer, FitnessLandscape, NumericFitness

node_order = ["main-0", "main-1", "main-2", "main-3", "minor-0", "minor-1", "isolate"]
sequences = [BinarySequence(f"{i:03b}", sequence_id=node) for i, node in enumerate(node_order)]
graph = nx.Graph()
for node, sequence in zip(node_order, sequences):
    graph.add_node(node, sequence=sequence)
graph.add_edges_from([
    ("main-0", "main-1"), ("main-1", "main-2"), ("main-2", "main-3"),
    ("minor-0", "minor-1"),
])
fitness = [0.0, 1.0, 2.0, 3.0, 10.0, 14.0, 100.0]
split = ["train", "train", "test", "test", "train", "test", "test"]
landscape = FitnessLandscape(
    sequences,
    graph,
    fitness_layers={"assay": NumericFitness.from_scalars("assay", fitness)},
    annotation_layers={"design": AnnotationLayer("design", {"split": split})},
)
landscape.view("assay")

inventory = []
eligible_values = []
for component_id, component in enumerate(landscape.get_components()):
    n_nodes = component.graph.number_of_nodes()
    n_edges = component.graph.number_of_edges()
    frame = component.get_annotation_layer("design").to_dataframe()
    train_count = int((frame["split"] == "train").sum())
    test_count = int((frame["split"] == "test").sum())
    eligible = n_nodes >= 2 and n_edges >= 1
    row = {
        "component": component_id,
        "nodes": n_nodes,
        "edges": n_edges,
        "train": train_count,
        "test": test_count,
        "eligible": eligible,
        "skip_reason": None if eligible else "fewer than 2 nodes or no represented edge",
        "test_support": "supported" if train_count > 0 else "no_train_node_in_component",
    }
    if eligible:
        energy = calculate_ruggedness_dirichlet_energy(component)
        row["global_energy"] = energy["global_dirichlet_energy"]
        eligible_values.extend(component.get_signal().tolist())
    inventory.append(row)

largest = landscape.get_components()[0].get_signal()
assert [row["nodes"] for row in inventory] == [4, 2, 1]
assert [row["eligible"] for row in inventory] == [True, True, False]
assert inventory[-1]["test_support"] == "no_train_node_in_component"
assert sum(row["nodes"] for row in inventory if row["eligible"]) == 6
all_eligible_mean = float(np.mean(eligible_values))
largest_only_mean = float(np.mean(largest))
assert all_eligible_mean != largest_only_mean
print(inventory, {"eligible_mean": all_eligible_mean, "largest_only_mean": largest_only_mean, "included": "6/7"})
```

Aggregate only compatible quantities with stated weights and denominators. The
largest component here excludes the high-valued minor component, while the
isolate is a non-estimable test cell rather than a zero-energy observation.

## Common failures

- Only the largest component is retained without redefining the population.
- Isolates receive zero distances or edge-based statistics.
- Components with no training nodes are included in interpolation metrics.
- Skipped components disappear from the output table.
- Component means are averaged without node counts or a common measurement scale.
