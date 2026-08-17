# Map local Dirichlet contributions

Local contributions allocate half of each incident squared edge difference to
each endpoint. They locate graph boundaries in the observed signal.

## Input

Node annotations must share the landscape's sequence-index order. Use a finite
scalar active layer and declare whether edges are weighted.

## Worked example

```python
# cookbook: test
import numpy as np
import pandas as pd

from fitness_landscape.analysis import (
    calculate_ruggedness_dirichlet_energy,
    local_dirichlet_energy_contribution,
)
from fitness_landscape.core import AnnotationLayer
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=3, K=1, seed=17)
landscape.annotation_layers["design"] = AnnotationLayer(
    "design",
    pd.DataFrame(
        {
            "background": ["A"] * 4 + ["B"] * 4,
            "sequence_id": [sequence.id for sequence in landscape.sequences],
        }
    ),
)

local = local_dirichlet_energy_contribution(landscape, weight_key=None)
global_result = calculate_ruggedness_dirichlet_energy(landscape)
np.testing.assert_allclose(
    sum(local.values()), global_result["global_dirichlet_energy"]
)

node_order = list(landscape.graph.nodes)
table = landscape.annotation_layers["design"].to_dataframe()
table["node"] = node_order
table["local_dirichlet"] = [local[node] for node in node_order]
table = table.sort_values("local_dirichlet", ascending=False).reset_index(drop=True)

assert len(table) == len(landscape)
assert table["sequence_id"].is_unique
assert np.all(table["local_dirichlet"] >= 0.0)
print(table.head(3).to_dict(orient="records"))
```

The local values sum to the global energy because every edge contribution is
split into two halves. Joining by explicit node order prevents annotations from
silently shifting relative to the graph.

## Interpretation

A high value marks a node adjacent to large observed fitness differences. It
does not establish that the residue, background, or experimental class causes
that boundary. Degree, sampling, measurement scale, and graph choice are
alternative explanations.

## Common failures

- A sorted annotation table is joined by row position rather than node ID.
- High-degree nodes are compared without a degree-aware sensitivity analysis.
- A descriptive boundary is labeled a causal or epistatic site.