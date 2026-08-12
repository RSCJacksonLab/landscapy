# Analyse connected components

Split disconnected landscapes before applying methods that require finite
paths, then report which components were eligible and how many observations
were excluded.

## Install and input

```bash
python -m pip install landscapy
```

The graph must be undirected and each node must carry the matching `sequence`.
Fitness and annotation layers must align with the sequence order.

## Worked example

```python
# cookbook: test
import networkx as nx
import numpy as np

from fitness_landscape import BinarySequence, FitnessLandscape, NumericFitness
from fitness_landscape.core import AnnotationLayer

node_order = ["left-0", "left-1", "right-0", "right-1", "isolate"]
sequences = [BinarySequence(text, sequence_id=node) for node, text in zip(
    node_order, ["000", "001", "110", "111", "010"]
)]
graph = nx.Graph()
for node, sequence in zip(node_order, sequences):
    graph.add_node(node, sequence=sequence)
graph.add_edges_from([("left-0", "left-1"), ("right-0", "right-1")])

landscape = FitnessLandscape(
    sequences,
    graph,
    fitness_layers={"assay": NumericFitness("assay", [1.0, 2.0, 10.0, 12.0, 7.0])},
    annotation_layers={
        "context": AnnotationLayer(
            "context",
            {"background": ["left", "left", "right", "right", "isolated"]},
        )
    },
)
landscape.view("assay")
components = landscape.get_components()

eligibility = []
included_values = []
for component_id, component in enumerate(components):
    nodes = sorted(component.graph.nodes, key=str)
    n_nodes = len(nodes)
    n_edges = component.graph.number_of_edges()
    eligible = n_nodes >= 2 and n_edges > 0
    values = component.get_layer("assay").to_scalar()
    annotation_frame = component.get_annotation_layer("context").to_dataframe()
    backgrounds = {
        node: annotation_frame.iloc[component.node_to_sequence_index[node]]["background"]
        for node in nodes
    }
    eligibility.append(
        {
            "component": component_id,
            "nodes": nodes,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "eligible": eligible,
            "backgrounds": backgrounds,
        }
    )
    if eligible:
        included_values.extend(values.tolist())

assert [row["n_nodes"] for row in eligibility] == [2, 2, 1]
assert [row["eligible"] for row in eligibility] == [True, True, False]
assert eligibility[0]["backgrounds"] == {"left-0": "left", "left-1": "left"}
assert eligibility[1]["backgrounds"] == {"right-0": "right", "right-1": "right"}
assert sum(row["n_nodes"] for row in eligibility if row["eligible"]) == 4
assert set(components[0].fitness_layers) == {"assay"}
assert set(components[0].annotation_layers) == {"context"}
assert set().union(*(set(row["nodes"]) for row in eligibility)) == set(node_order)

# This pooled mean is valid only because all eligible values share the same
# assay scale. Report its 4/5 denominator; do not average component estimates
# that answer incompatible questions.
eligible_mean = float(np.mean(included_values))
assert eligible_mean == 6.25
print(eligibility)
print({"eligible_mean": eligible_mean, "included_nodes": 4, "total_nodes": 5})
```

The result contains components of sizes 2, 2, and 1. Layers, annotations, and
original node labels are retained. The singleton is visible but ineligible for
an edge-based analysis, so the example reports an included denominator of 4/5.

## Interpretation

Component-wise analysis prevents infinite cross-component relationships from
being converted to finite numbers. Recombination is justified only for
compatible estimands on a common scale; use node-weighted or otherwise stated
weights rather than an unlabelled mean of component means.

## Common failures

- Analysing only the largest component without reporting excluded nodes.
- Losing arbitrary node labels or layer alignment during manual subgraphing.
- Treating isolates as zero-distance observations instead of unsupported pairs.
- Combining component summaries with incompatible assay backgrounds.
- Omitting component eligibility criteria and skipped denominators.
