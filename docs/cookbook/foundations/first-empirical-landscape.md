# Build a first empirical landscape

Use this recipe when each row contains an aligned sequence and an empirical
numeric response.

## Input

The input table must have one unique sequence per row and a numeric `fitness`
column. This example reads the versioned [cookbook dataset](../data/README.md).
Its `sequence` column is forced to string so leading zeroes survive CSV parsing.

## Worked example

```python
# cookbook: test
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence, FitnessLandscape, NumericFitness

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"),
    dtype={"sequence": "string"},
)
assert table["sequence"].is_unique
assert table["fitness"].notna().all()

sequences = [
    BinarySequence([int(site) for site in text], sequence_id=f"toy-{i:03d}")
    for i, text in enumerate(table["sequence"])
]
assay = NumericFitness.from_scalars(
    "assay",
    table["fitness"].to_numpy(),
    metadata={"source": "cookbook toy data v1.0", "units": "arbitrary"},
)

landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"assay": assay},
)
active = landscape.view("assay")

assert len(landscape) == 8
assert landscape.graph.number_of_nodes() == 8
assert landscape.graph.number_of_edges() == 12
np.testing.assert_allclose(active.to_scalar(), table["fitness"])
assert landscape.active_layer_name == "assay"
assert [landscape.sequence_index_to_node[i] for i in range(8)] == list(
    landscape.graph.nodes
)
assert all(
    landscape.graph.nodes[node]["sequence"] == sequences[index]
    for index, node in landscape.sequence_index_to_node.items()
)
edge_schema = landscape.graph.graph["landscapy_edge_schema"]
assert edge_schema["constructor"] == "hamming-binary"

print(len(landscape), landscape.graph.number_of_edges())
print(landscape.active_layer_name, active.to_scalar().tolist())
print(edge_schema["constructor"])
```

Expected values are `8 12`, the active layer `assay` with values
`[0.1, 0.25, 0.35, 0.6, 0.2, 0.55, 0.7, 0.95]`, and constructor metadata
`hamming-binary`. The assertions are the minimal alignment audit:
row count equals node count, layer order equals sequence order, and the graph's
node-to-sequence mapping is reversible.

## Interpretation

The graph connects observed sequences that differ at exactly one site. Because
the toy table enumerates the complete binary cube, twelve edges are expected.
For empirical data, a missing edge can mean an unobserved neighbour rather than
a biological barrier. Constructing this object does not validate Hamming
adjacency as the correct biological representation and does not turn the
illustrative values into evidence.

## Common failures

- CSV readers may parse `000` as integer zero unless the column type is fixed.
- Duplicate sequences make sequence-keyed alignment ambiguous; resolve the
  experimental unit and replicate policy before construction.
- A fitness array with a different row order can be accepted as the wrong data;
  always audit sequence IDs and representative values.
- Unequal sequence lengths are invalid for Hamming construction.
