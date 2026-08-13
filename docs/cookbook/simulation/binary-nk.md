# Generate binary NK landscapes

The binary NK model combines contributions from `N` binary sites, each coupled
to `K` other sites. Increasing `K` changes the model's interaction structure;
it does not guarantee monotonic ruggedness in any one random realization.

## Install and input

```bash
python -m pip install landscapy
```

Choose `0 <= K < N` and a recorded seed. The factory enumerates the complete
binary state space and constructs its Hamming graph.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import (
    calculate_epistasis_walsh,
    calculate_ruggedness_dirichlet_energy,
    calculate_ruggedness_local_optima,
)
from fitness_landscape.models import create_nk_binary_landscape

report = {}
for K in (0, 1, 3):
    landscape = create_nk_binary_landscape(N=4, K=K, seed=12)
    layer_name = next(iter(landscape.fitness_layers))
    metadata = landscape.fitness_layers[layer_name].metadata
    epistasis = calculate_epistasis_walsh(landscape, order=4)
    highest_nonzero_order = max(
        order
        for order, terms in epistasis["by_order"].items()
        if any(abs(value) > 1e-12 for value in terms.values())
    )
    report[K] = {
        "nodes": landscape.graph.number_of_nodes(),
        "edges": landscape.graph.number_of_edges(),
        "layer": layer_name,
        "interaction_degrees": metadata["interaction_degrees"],
        "local_optima": calculate_ruggedness_local_optima(landscape)[
            "local_optima_count"
        ],
        "dirichlet_energy": calculate_ruggedness_dirichlet_energy(landscape)[
            "global_dirichlet_energy"
        ],
        "highest_nonzero_walsh_order": highest_nonzero_order,
    }

assert all(row["nodes"] == 2**4 for row in report.values())
assert all(row["edges"] == 4 * 2 ** (4 - 1) for row in report.values())
assert report[0]["interaction_degrees"] == [0, 0, 0, 0]
assert report[0]["highest_nonzero_walsh_order"] == 1
assert report[1]["highest_nonzero_walsh_order"] <= 2
assert report[3]["highest_nonzero_walsh_order"] <= 4
assert np.isfinite([row["dirichlet_energy"] for row in report.values()]).all()
print(report)
```

The node and edge counts are exact known answers. The upper bound of `K + 1`
on Walsh order checks the construction; counts of optima and energy are
seed-specific outcomes and need independent realizations for comparison.

## Common failures

- `K` is described as total sequence length rather than neighbours per site.
- One seed at each `K` is treated as a replicated simulation study.
- Active fitness-layer identity and model metadata are omitted from output.
- Complete-cube Walsh analysis is applied after filtering genotypes.
- A software known answer is presented as validation of the NK biological model.
