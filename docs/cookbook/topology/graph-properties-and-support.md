# Report graph properties and support

Run this audit before any topology-dependent analysis. Empty, singleton, and
disconnected graphs have legitimate but different domains of definition.

## Install and input

```bash
python -m pip install landscapy
```

`graph_properties` accepts an undirected NetworkX graph or a
`FitnessLandscape`. Graph nodes are the observed support; absent sequences are
not implicit nodes.

## Worked example

```python
# cookbook: test
import math

import networkx as nx

from fitness_landscape.analysis import graph_properties

graphs = {
    "empty": nx.Graph(),
    "singleton": nx.empty_graph(1),
    "connected": nx.path_graph(4),
    "disconnected": nx.Graph([(0, 1), (2, 3)]),
}

audit = {}
for name, graph in graphs.items():
    summary = graph_properties(graph)
    audit[name] = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": summary["density"],
        "degree": summary["degree"],
        "clustering": summary["clustering"],
        "path_length": summary["path_length"],
        "path_note": summary.get("path_length_note"),
        "components": summary["components"],
    }

assert math.isnan(audit["empty"]["degree"]["mean"])
assert math.isnan(audit["empty"]["path_length"])
assert audit["singleton"]["path_length"] == 0.0
assert audit["connected"]["components"]["count"] == 1
assert audit["disconnected"]["components"]["sizes"] == [2, 2]
assert audit["disconnected"]["path_note"] == "Calculated for largest connected component"

for name, row in audit.items():
    print(name, row["nodes"], row["edges"], row["components"], row["path_note"])
```

The empty graph reports undefined degree, clustering, and path summaries; the
singleton has zero degree and zero path length. The disconnected example has
two components of size two. Its finite path length applies only to one largest
component and must not be reported as a whole-graph average.

## Interpretation

Node and edge counts describe observed support, density normalizes edge count
by possible pairs, and component sizes determine which node pairs can interact
through the graph. These facts determine whether later estimates have a valid
domain. They do not distinguish biological isolation from incomplete sampling.

## Common failures

- Omitting the largest-component note makes a disconnected path summary appear
  global.
- Treating NaN as zero changes “undefined” into a measured absence.
- Comparing densities across very different node counts without the possible
  edge denominator can be misleading.
- Reporting only the largest component hides excluded nodes and categories.
- Interpreting fragmentation as intrinsic biology without auditing missing
  single-mutant neighbours confounds topology with empirical sampling.
