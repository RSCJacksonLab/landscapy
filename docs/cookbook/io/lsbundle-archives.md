# Export a deterministic `.lsbundle` archive

An `.lsbundle` is a deterministic ZIP container around the portable bundle
schema. It is convenient for a release asset or data-repository deposit.

## Input

The input must be a valid undirected `FitnessLandscape`. Metadata must be
JSON-compatible and should identify the deposited dataset and pipeline.

## Worked example

```python
# cookbook: test
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=3, K=1, seed=17)
metadata = {
    "dataset_name": "seeded NK validation fixture",
    "assay_type": "simulation",
    "version": "1.0",
    "provenance": {"license": "CC0-1.0", "pipeline": "cookbook-io-v1"},
}

with TemporaryDirectory() as tmp:
    first = Path(tmp) / "first.lsbundle"
    second = Path(tmp) / "second.lsbundle"
    landscape.export_lsbundle(first, metadata=metadata, backend="portable")
    landscape.export_lsbundle(second, metadata=metadata, backend="portable")

    first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
    assert first_hash == second_hash

    with ZipFile(first) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        stored_metadata = json.loads(archive.read("metadata.json"))
    assert names == sorted(names)
    assert manifest["serializer_backend"] == "portable"
    assert stored_metadata["dataset_name"] == metadata["dataset_name"]
    assert stored_metadata["provenance"] == metadata["provenance"]

print(first_hash, manifest["serializer_backend"])
```

Identical inputs and metadata produce identical archive bytes. Deposit the
archive together with its SHA-256 digest, citation, license, and a human-readable
method description. Determinism supports integrity checks; it does not prove
that the underlying graph or analysis is scientifically appropriate.

## Portable versus pickle

`backend="portable"` is the publication format. `backend="pickle"` exists only
for legacy interoperability and can execute code during deserialization. Never
load a pickle bundle from an untrusted or unverifiable source.

## Common failures

- The archive is renamed without preserving its checksum or metadata.
- Non-deterministic user metadata, such as a current timestamp, changes bytes.
- A legacy pickle is treated as safe because it has an `.lsbundle` suffix.
- The archive is deposited without package version, license, or node-order data.
