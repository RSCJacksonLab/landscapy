# Prepare aligned features, targets, and split annotations

Prepare features without using the target, attach the measured response as a
separate layer, and retain split and background fields as annotations. Fit every
data-dependent preprocessing step on training rows only.

## Install and input

```bash
python -m pip install "landscapy[ml]"
```

The fixture's primary key is `sequence`; `fitness` is the target and `split`
and `background` are design fields. OHE is appropriate for this complete
aligned binary example. For proteins, compute a versioned PLM cache with
`compute_plm_embeddings` and record model revision, pooling, device, and dtype.

## Worked example

```python
# cookbook: test
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.core import AnnotationLayer, FitnessLandscape, NumericFitness

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
sequences = [
    BinarySequence(text, sequence_id=f"toy-{row}")
    for row, text in enumerate(table["sequence"])
]
features = np.stack([sequence.to_one_hot().reshape(-1) for sequence in sequences])
target = table["fitness"].to_numpy(dtype=float)
design = AnnotationLayer(
    "design",
    table[["split", "background"]],
    metadata={"dataset": "toy_landscape", "version": "1.0"},
)
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",  # constructed from sequence identity, never target values
    fitness_layers={"measured_fitness": NumericFitness.from_scalars("measured_fitness", target)},
    annotation_layers={"design": design},
    embeddings={"ohe": features},
    embedding_domain="ohe",
)
landscape.view("measured_fitness")

split = landscape.get_annotation_layer("design").to_dataframe()["split"].to_numpy()
train = split == "train"
labelled = np.isfinite(landscape.get_signal())
train_mean = features[train].mean(axis=0)
train_scale = features[train].std(axis=0)
train_scale[train_scale == 0.0] = 1.0
normalized = (features - train_mean) / train_scale

assert landscape.active_embedding_domain == "ohe"
assert features.shape == (len(sequences), 3 * 2)
assert target.shape == (len(sequences),)
assert train.sum() == 4 and labelled.sum() == 8
assert np.isfinite(normalized).all()
for row, node in enumerate(landscape.graph.nodes()):
    assert landscape.sequence_index_for_node(node) == row
    np.testing.assert_array_equal(features[row], landscape.embeddings["ohe"][row])
print(features.shape, target.shape, train_mean, train_scale)
```

Export `normalized`, the retained row indices, and the fitted training
statistics together. Missing labels should produce an explicit mask; they must
not be converted to zero. The graph and feature transform support prediction,
not a claim that the representation captures biological mechanism.

## Common failures

- Fitness is used to choose graph edges, embeddings, or preprocessing hyperparameters.
- Normalization is fit on test rows before evaluation.
- Row position is assumed after an external table join without checking sequence IDs.
- Missing labels are imputed as negative examples.
- PLM model revision, pooling rule, or cached sequence order is omitted.
