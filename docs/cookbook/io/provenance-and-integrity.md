# Record provenance and verify integrity

A reproducible bundle needs scientific provenance in addition to structural
checksums. This recipe creates a compact metadata record and independently
audits every manifest payload.

## Install and input

```bash
python -m pip install landscapy
```

Record source and license, preprocessing and alignment, graph parameters,
software versions, seeds, target definitions, node order, and the user pipeline
version before starting the analysis.

## Worked example

```python
# cookbook: test
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=3, K=1, seed=23)
node_order = [
    {"index": index, "id": sequence.id, "sequence": sequence.to_str()}
    for index, sequence in enumerate(landscape.sequences)
]
node_fingerprint = hashlib.sha256(
    json.dumps(node_order, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
metadata = {
    "dataset_name": "seeded NK fixture",
    "assay_type": "simulation",
    "source_name": "landscapy cookbook",
    "version": "1.0",
    "provenance": {
        "license": "CC0-1.0",
        "preprocessing": {"alignment": "not applicable", "duplicates": "forbidden"},
        "graph": {"constructor": "NK complete Hamming cube", "N": 3, "K": 1},
        "software": {
            "landscapy": importlib.metadata.version("landscapy"),
            "python": platform.python_version(),
        },
        "random_seeds": {"landscape": 23},
        "targets": {"fitness": {"units": "model units", "kind": "simulated"}},
        "node_order_sha256": node_fingerprint,
        "pipeline": {"name": "cookbook-integrity", "version": "1.0"},
    },
}

with TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "bundle"
    landscape.save_bundle_dir(bundle, metadata=metadata)
    manifest = json.loads((bundle / "manifest.json").read_text())
    stored_metadata = json.loads((bundle / "metadata.json").read_text())

    for relative_path, payload in manifest["files"].items():
        path = bundle / relative_path
        assert path.stat().st_size == payload["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["sha256"]

    restored = landscape.load_bundle_dir(bundle)
    restored_order = [
        {"index": index, "id": sequence.id, "sequence": sequence.to_str()}
        for index, sequence in enumerate(restored.sequences)
    ]
    restored_fingerprint = hashlib.sha256(
        json.dumps(restored_order, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert restored_fingerprint == stored_metadata["provenance"]["node_order_sha256"]
    assert restored.graph.number_of_edges() == landscape.graph.number_of_edges()

print(manifest["node_count"], len(manifest["files"]), node_fingerprint)
```

The manifest verifies stored bytes; the node-order fingerprint verifies the
user-level identity/order contract. Neither substitutes for source-data
versioning or a recorded preprocessing method.

## Common failures

- Only the package version is recorded, omitting graph and target definitions.
- A checksum is computed before the final file is written.
- Node order is treated as incidental even though layers and embeddings use it.
- Operational failure is recorded as a completed negative scientific result.
- Metadata claims a remote backup or source version that was never verified.
