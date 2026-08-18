# Use diffusion resource guards

Embedding diffusion computes exact sparse powers before applying the final
connectivity threshold. Bound both intermediate nonzeros and estimated scalar
products so infeasible work fails before exhausting memory.

## Input

Candidate storage begins on the order of the symmetric union of `n * k`
directed neighbours, but exact powers can fill in rapidly. Measure the actual
construction metadata on representative components.

## Worked example

```python
# cookbook: test
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_landscape import BinarySequence
from fitness_landscape.core import create_diffusion_emb_graph

table = pd.read_csv(
    Path("docs/cookbook/data/toy_landscape.csv"), dtype={"sequence": "string"}
)
sequences = [BinarySequence(text) for text in table.sequence]
embeddings = np.stack([sequence.to_one_hot().reshape(-1) for sequence in sequences])

try:
    create_diffusion_emb_graph(
        sequences,
        embeddings,
        k=3,
        t=3,
        backend="balltree",
        embedding_domain="plm",
        connectivity_threshold=0.9,
        max_diffusion_nnz=4,
        max_diffusion_work=1_000,
    )
except MemoryError as error:
    failure = str(error)
else:
    raise AssertionError("the deliberately small guard must fail")
assert "connectivity_threshold" in failure and "cannot reduce intermediate" in failure

reduced = create_diffusion_emb_graph(
    sequences,
    embeddings,
    k=1,
    t=1,
    backend="balltree",
    embedding_domain="plm",
    max_diffusion_nnz=30,
    max_diffusion_work=1_000,
)
diagnostics = reduced.graph["diffusion_construction"]
assert diagnostics["affinity_nnz"] <= diagnostics["max_diffusion_nnz"]
assert diagnostics["kernel_nnz"] <= diagnostics["max_diffusion_nnz"]
assert diagnostics["estimated_scalar_products"] <= diagnostics["max_diffusion_work"]
assert diagnostics["diffusion_accuracy"] == "exact"
print(failure, diagnostics)
```

The successful fallback changes `k` and `t`, so it defines a different graph
estimand. Partitioning into preregistered components is another option, but it
removes cross-component candidate pairs and must be labelled accordingly.

## Common failures

- Only final graph edges are used to estimate intermediate sparse-power memory.
- A high post-kernel threshold is expected to prevent exact multiplication work.
- `max_diffusion_nnz` is raised without provisioning peak working memory.
- Reduced `k`, `t`, or component scope is reported as the original analysis.
- A resource failure is recorded as a completed negative scientific result.
