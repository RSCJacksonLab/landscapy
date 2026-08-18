# Work with fitness layers

A landscape can hold multiple phenotypes or representations of uncertainty.
The active view selects a layer for methods that require one. Changing the active layer does not alter
data in other fitness layers, annotations, sequences or graph connectivity.

## Input

Every ready-made layer must contain one row per sequence. Numeric rows may hold
one scalar or several replicates. Categorical values must occur in the declared
category list. Probabilistic rows must be finite, non-negative, and sum to one.

## Worked example

```python
# cookbook: test
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_landscape import (
    BinarySequence,
    CategoricalFitness,
    FitnessLandscape,
    NumericFitness,
    ProbabilisticCategoricalFitness,
)

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"),
    dtype={"sequence": "string"},
)
sequences = [BinarySequence(text, sequence_id=f"toy-{i:03d}") for i, text in enumerate(table.sequence)]
landscape = FitnessLandscape.build(sequences, graph="hamming", fitness_layers={})

scalar = NumericFitness.from_scalars("fitness", table.fitness.to_numpy())
replicates = NumericFitness.from_replicates(
    "replicates", table[["replicate_1", "replicate_2"]].to_numpy().tolist()
)
classes = CategoricalFitness(
    "activity", table.activity_class.tolist(), categories=["low", "mid", "high"]
)
probabilities = ProbabilisticCategoricalFitness(
    "activity_posterior",
    probabilities=np.array(
        [
            [0.80, 0.15, 0.05], [0.70, 0.25, 0.05],
            [0.20, 0.70, 0.10], [0.10, 0.75, 0.15],
            [0.75, 0.20, 0.05], [0.10, 0.75, 0.15],
            [0.05, 0.20, 0.75], [0.02, 0.08, 0.90],
        ]
    ),
    categories=["low", "mid", "high"],
)
for layer in (scalar, replicates, classes, probabilities):
    landscape.attach(layer)

np.testing.assert_allclose(
    landscape.get_layer("replicates").to_scalar(),
    table[["replicate_1", "replicate_2"]].mean(axis=1),
)
assert landscape.view("activity").to_scalar().tolist() == [0, 0, 1, 1, 0, 1, 2, 2]
assert landscape.active_layer_name == "activity"
assert landscape.get_layer("activity_posterior").to_scalar().tolist() == [0, 0, 1, 1, 0, 1, 2, 2]

# Raw values can instead be aligned by sequence key. Missing rows become NaN
# only when the policy is explicit.
landscape.attach(
    name="sparse_measurement",
    values={"000": 1.0, "111": 2.0},
    dtype="numeric",
    map_by="sequence",
    allow_missing=True,
)
sparse = landscape.get_layer("sparse_measurement").to_scalar()
assert sparse[0] == 1.0 and sparse[-1] == 2.0
assert np.isnan(sparse[1:-1]).all()

# Duplicate sequence rows require an explicit policy.
duplicate_landscape = FitnessLandscape.build(
    [BinarySequence("000"), BinarySequence("000"), BinarySequence("001")],
    graph="hamming",
    fitness_layers={},
)
duplicate_landscape.attach(
    name="mapped",
    values={"000": [1.0, 1.2], "001": 2.0},
    dtype="numeric",
    map_by="sequence",
    on_duplicates="all",
)
np.testing.assert_allclose(
    duplicate_landscape.get_layer("mapped").to_scalar(), [1.1, 1.1, 2.0]
)

print(sorted(landscape.fitness_layers))
print(landscape.active_layer_name)
```

Replicate-valued numeric layers scalarize to the row mean by default.
Categorical layers scalarize by declared category order, while probabilistic
categorical layers use the most probable category. 

## Mapping policy

Index mapping is safest after a row-order audit. Sequence mapping is useful
when tables are reordered, but duplicates require `error`, `first`, `all`, or
numeric `aggregate` policy. `allow_missing=True` is an explicit decision to
create missing values, not an imputation method.

Experimental phenotypes belong in fitness layers. Taxonomy, batch, background,
and train/test split normally belong in annotation layers. Converting metadata
to a scalar fitness signal can create a false biological ordering.

## Common failures

- Layer length differs from sequence count or table order was not verified.
- Probability rows do not sum to one or category order differs between data and
  labels.
- NaN replicate rows are scalarized without an explicit missing-data policy.
- Duplicate sequences are silently copied when the experimental unit should
  instead have been aggregated.
- The active layer is assumed rather than checked with `active_layer_name`.
