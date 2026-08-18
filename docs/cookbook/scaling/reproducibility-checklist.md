# Record a reproducible run manifest

Create the run manifest before computation, then append outputs and terminal
status without erasing failures. A completed scientific negative result differs
from an operational failure that produced no estimable result.

## Input

Record environment versions, input hashes, sequence/node order, graph and edge
semantics, random state, independent unit, expected counts, output schema, and
resume state. Portable-bundle payload checksums make the result auditable.

## Worked example

```python
# cookbook: test
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.core import FitnessLandscape, NumericFitness

input_path = Path("docs/cookbook/data/toy_landscape.csv")
table = pd.read_csv(input_path, dtype={"sequence": "string"})
sequences = [BinarySequence(text, sequence_id=f"toy-{i}") for i, text in enumerate(table.sequence)]
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"assay": NumericFitness.from_scalars("assay", table.fitness)},
)
node_order = list(landscape.graph.nodes())
sequence_order = ["".join(map(str, sequence.to_array())) for sequence in landscape.sequences]
manifest = {
    "schema": "landscapy-run-manifest-1",
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "landscapy": version("landscapy"),
    },
    "input": {
        "path": str(input_path),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "rows": len(table),
    },
    "alignment": {
        "sequence_order_sha256": hashlib.sha256("\n".join(sequence_order).encode()).hexdigest(),
        "node_order": node_order,
    },
    "graph": {
        "constructor": "hamming",
        "parameters": {},
        "edge_schema": landscape.graph.graph["landscapy_edge_schema"],
        "expected": {"nodes": 8, "edges": 12},
    },
    "random_state": {"seed": None, "reason": "deterministic constructor"},
    "independent_unit": "one synthetic genotype row",
    "output_schema": {"bundle": "portable directory", "fitness_layer": "assay"},
    "stages": [
        {"name": "load_and_validate", "status": "completed"},
        {"name": "graph_construction", "status": "completed"},
        {"name": "analysis", "status": "completed_negative", "result": "no preregistered effect; example only"},
        {"name": "optional_secondary_backend", "status": "operational_failure", "error": "not run in this example", "resumable": True},
    ],
}

with TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "run_bundle"
    landscape.save_bundle_dir(bundle, metadata={"run_manifest": manifest})
    bundle_manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["bundle"] = {
        "manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        "payload_checksums": {
            name: record["sha256"] for name, record in bundle_manifest["files"].items()
        },
    }
    encoded = json.dumps(manifest, sort_keys=True)
    restored = FitnessLandscape.load_bundle_dir(bundle)
    assert restored.graph.number_of_nodes() == manifest["graph"]["expected"]["nodes"]
    assert restored.graph.number_of_edges() == manifest["graph"]["expected"]["edges"]
    assert len(manifest["bundle"]["payload_checksums"]) > 0
    assert json.loads(encoded)["stages"][2]["status"] == "completed_negative"
    assert json.loads(encoded)["stages"][3]["status"] == "operational_failure"
print(manifest)
```

Before launch, fill all input, alignment, graph, seed, independent-unit, and
expected-count fields. Afterward, record output checksums, observed counts, and
every completed, failed, or resumable stage without relabelling failures.

## Common failures

- Environment and input hashes are reconstructed after artifacts have changed.
- Sequence order is recorded but graph node-to-row alignment is not.
- Seeds are listed without the independent replicate or task they identify.
- Operational failures are counted as negative scientific results.
- Resumed output overwrites failure history or uses a different parameter manifest.
