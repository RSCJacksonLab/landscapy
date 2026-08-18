# Save and load a portable directory bundle

Portable bundles preserve graph structure, sequences, layers, annotations,
embeddings, active views, metadata, and checksums without requiring pickle.

## Input

`pyarrow` enables Parquet payloads. Without it, Landscapy writes its
deterministic JSON-table fallback. Both are valid portable bundles.

## Worked example

```python
# cookbook: test
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fitness_landscape.core import (
    AnnotationLayer,
    BinarySequence,
    FitnessLandscape,
    NumericFitness,
)
from fitness_landscape.io import BundleValidationError, ChecksumMismatchError

sequences = [
    BinarySequence(f"{value:03b}", sequence_id=f"s{value}") for value in range(8)
]
fitness = NumericFitness.from_scalars("assay", np.linspace(0.0, 1.0, 8))
annotations = AnnotationLayer(
    "design", pd.DataFrame({"split": ["train"] * 6 + ["test"] * 2})
)
ohe = np.stack([sequence.to_one_hot().reshape(-1) for sequence in sequences])
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"assay": fitness},
    annotation_layers={"design": annotations},
    embeddings={"ohe": ohe},
    embedding_domain="ohe",
)
landscape.view("assay")

with TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "toy_bundle"
    landscape.save_bundle_dir(
        bundle,
        metadata={
            "dataset_name": "cookbook-toy",
            "assay_type": "synthetic",
            "version": "1.0",
            "provenance": {"license": "CC0-1.0"},
        },
        overwrite=False,
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    restored = FitnessLandscape.load_bundle_dir(bundle)

    assert manifest["node_count"] == 8
    assert manifest["edge_count"] == 12
    assert manifest["graph"]["storage_backend"] in {"parquet", "json-table-v1"}
    assert restored.active_layer_name == "assay"
    assert restored.active_embedding_domain == "ohe"
    assert list(restored.annotation_layers) == ["design"]
    assert restored.graph.number_of_edges() == landscape.graph.number_of_edges()
    np.testing.assert_allclose(restored.view("assay").to_scalar(), fitness.to_scalar())
    np.testing.assert_allclose(restored.embeddings["ohe"], ohe)
    assert getattr(restored, "_bundle_metadata")["dataset_name"] == "cookbook-toy"

    try:
        landscape.save_bundle_dir(bundle)
    except FileExistsError:
        pass
    else:
        raise AssertionError("overwrite=False must protect an existing bundle")

print(
    manifest["graph"]["storage_backend"],
    manifest["node_count"],
    manifest["edge_count"],
)
```

Loading verifies every recorded payload checksum before reconstructing the
landscape. `BundleValidationError` means the schema or content is invalid;
`ChecksumMismatchError`, a subclass, means a payload differs from the manifest.
Do not suppress either exception: obtain a verified copy or regenerate the
artifact from its recorded source.

