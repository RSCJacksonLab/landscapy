# Import and export a landscape table with CSV

CSV is useful for flat interchange, but it is not a complete landscape
serialization. This recipe uses Pandas because `read_csv_landscape` and
`to_csv_landscape` are not in the supported [0.9 API](../../public_api.md).

## Install and input

```bash
python -m pip install landscapy
```

The versioned toy table has one unique aligned sequence and one scalar response
per row. Replicate, category, and probability columns are added with an
explicit schema so their meanings are not inferred from column names.

## Worked example

```python
# cookbook: test
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fitness_landscape.core import (
    BinarySequence,
    CategoricalFitness,
    FitnessLandscape,
    NumericFitness,
    ProbabilisticCategoricalFitness,
)

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
table["replicate_1"] = table["fitness"] - 0.02
table["replicate_2"] = table["fitness"] + 0.02
table["class"] = np.where(table["fitness"] >= 0.5, "high", "low")
table["p_high"] = table["fitness"]
table["p_low"] = 1.0 - table["p_high"]

sequences = [
    BinarySequence([int(site) for site in text], sequence_id=f"toy-{i:03d}")
    for i, text in enumerate(table["sequence"])
]
layers = {
    "assay": NumericFitness.from_scalars("assay", table["fitness"]),
    "replicates": NumericFitness.from_replicates(
        "replicates", table[["replicate_1", "replicate_2"]].to_numpy()
    ),
    "class": CategoricalFitness(
        "class", table["class"].tolist(), categories=["low", "high"]
    ),
    "class_probability": ProbabilisticCategoricalFitness.from_probabilities(
        "class_probability",
        table[["p_low", "p_high"]].to_numpy(),
        categories=["low", "high"],
    ),
}
landscape = FitnessLandscape.build(
    sequences, graph="hamming", fitness_layers=layers
)
landscape.view("assay")

export = pd.DataFrame(
    {
        "sequence_id": [sequence.id for sequence in landscape.sequences],
        "sequence": ["".join(map(str, sequence.to_array())) for sequence in sequences],
        "assay": landscape.get_layer("assay").to_scalar(),
        "class": landscape.get_layer("class").to_scalar(),
    }
)
with TemporaryDirectory() as tmp:
    path = Path(tmp) / "landscape.csv"
    export.to_csv(path, index=False)
    restored = pd.read_csv(path, dtype={"sequence": "string"})
    pd.testing.assert_frame_equal(restored, export, check_dtype=False)

assert list(landscape.fitness_layers) == [
    "assay", "replicates", "class", "class_probability"
]
assert export["sequence_id"].is_unique
print(export.shape, list(landscape.fitness_layers))
```

The flat export has eight rows and four columns. The in-memory object retains
all four layer types, but the exported table deliberately selects one scalar
value per layer.

## Interpretation and limits

CSV preserves only the columns and conventions written by the user. It does
not preserve graph edges or constructor provenance, annotation schemas,
embedding arrays, replicate axes, probability category order, active views, or
checksums. Use a [portable bundle](portable-directory-bundles.md) when those
objects matter.

## Common failures

- Leading zeroes are lost because the sequence column was parsed as an integer.
- Replicates are averaged without retaining the aggregation rule.
- Probability columns are reloaded in a different category order.
- Row order changes without a stable sequence identifier and alignment audit.
- A CSV round trip is described as a complete landscape round trip.
