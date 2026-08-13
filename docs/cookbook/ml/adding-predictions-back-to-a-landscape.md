# Add predictions without replacing measurements

Store predictions under new, model-specific layer names and preserve the
measured layer unchanged. Ensemble members are replicate values of the
prediction layer; uncertainty and split membership have separate meanings.

## Install and input

```bash
python -m pip install "landscapy[ml,parquet]"
```

Predictions in this example exist only for held-out rows. The model values are
already on the assay scale; real pipelines must record any inverse transform.

## Worked example

```python
# cookbook: test
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.core import AnnotationLayer, FitnessLandscape, NumericFitness

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
sequences = [BinarySequence(text, sequence_id=f"toy-{i}") for i, text in enumerate(table.sequence)]
split = table.split.to_numpy()
ensemble = [
    [] if cell == "train" else [float(table.fitness.iloc[i] - 0.03), float(table.fitness.iloc[i] + 0.01)]
    for i, cell in enumerate(split)
]
uncertainty = [np.nan if cell == "train" else 0.02 for cell in split]
model_provenance = {
    "model": "external-ridge-example",
    "model_version": "demo-1",
    "feature_domain": "ohe",
    "training_split": "design.split == train",
    "output_scale": "original assay units; no inverse transform",
    "seed": 41,
}

landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"measured_fitness": NumericFitness.from_scalars("measured_fitness", table.fitness)},
    annotation_layers={"design": AnnotationLayer("design", table[["split", "background"]])},
)
landscape.attach(
    NumericFitness("prediction_external_ridge_demo1", ensemble, metadata=model_provenance)
)
landscape.attach(
    NumericFitness.from_scalars(
        "prediction_sd_external_ridge_demo1", uncertainty, metadata={"units": "assay scale"}
    )
)

measured_before = landscape.fitness_layers["measured_fitness"].to_scalar().copy()
with TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "prediction_bundle"
    landscape.save_bundle_dir(
        bundle,
        metadata={"dataset": "toy_landscape", "version": "1.0", "model": model_provenance},
    )
    restored = FitnessLandscape.load_bundle_dir(bundle)
    np.testing.assert_allclose(restored.fitness_layers["measured_fitness"].to_scalar(), measured_before)
    prediction = restored.fitness_layers["prediction_external_ridge_demo1"]
    np.testing.assert_allclose(prediction.to_scalar()[split == "test"], table.fitness[split == "test"] - 0.01)
    assert np.isnan(prediction.to_scalar()[split == "train"]).all()
    assert len(prediction.get_value(3)) == 2
    assert restored.get_annotation_layer("design").to_dataframe()["split"].tolist() == split.tolist()
    assert prediction.metadata["model_version"] == "demo-1"
print(list(restored.fitness_layers))
```

See [portable directory bundles](../io/portable-directory-bundles.md) for the
artifact schema and checksum audit. A prediction layer is not a replacement
measurement and must not become the default empirical response accidentally.

## Common failures

- Predictions overwrite the measured fitness layer.
- Ensemble members, predictive standard deviation, and assay replicates are conflated.
- Training-row fitted values are labelled as held-out predictions.
- Transformed predictions are stored without scale and inverse-transform metadata.
- The bundle round trip checks file existence but not layer values and split alignment.
