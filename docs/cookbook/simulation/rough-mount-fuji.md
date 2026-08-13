# Compare smooth and rugged Rough Mount Fuji landscapes

A Rough Mount Fuji (RMF) signal is a distance-dependent slope plus independent
Gaussian noise. `sigma=0` gives a deterministic additive landscape relative to
the specified optimum; positive noise can add local optima and block increasing
paths.

## Install and input

```bash
python -m pip install landscapy
```

Record `N`, `slope`, `sigma`, `optimum`, and seed. The seed affects the signal
only when `sigma > 0`.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import (
    calculate_ruggedness_autocorrelation_analytical,
    calculate_ruggedness_dirichlet_energy,
    calculate_ruggedness_local_optima,
    find_greedy_accessible_paths,
)
from fitness_landscape.models import create_rmf_landscape

optimum = [0, 0, 0, 0]
summary = {"smooth": [], "noisy": []}
for seed in range(6):
    for label, slope, sigma in (("smooth", 1.0, 0.0), ("noisy", 0.35, 1.5)):
        landscape = create_rmf_landscape(
            N=4, slope=slope, sigma=sigma, seed=seed, optimum=optimum
        )
        start = landscape.sequences[-1]  # 1111 in generated lexicographic order
        end = landscape.sequences[0]     # specified 0000 reference optimum
        summary[label].append(
            {
                "seed": seed,
                "optima": calculate_ruggedness_local_optima(landscape)[
                    "local_optima_count"
                ],
                "energy": calculate_ruggedness_dirichlet_energy(landscape)[
                    "global_dirichlet_energy"
                ],
                "lag1": calculate_ruggedness_autocorrelation_analytical(
                    landscape, lag_max=1
                )["autocorrelation"][1],
                "increasing_paths_to_reference": find_greedy_accessible_paths(
                    landscape, start, end
                )["path_count"],
            }
        )

assert all(row["optima"] == 1 for row in summary["smooth"])
assert all(row["increasing_paths_to_reference"] == 24 for row in summary["smooth"])
assert len({row["energy"] for row in summary["smooth"]}) == 1
assert any(row["optima"] > 1 for row in summary["noisy"])
assert np.isfinite([row["lag1"] for rows in summary.values() for row in rows]).all()
print(summary)
```

The smooth known answer has `4! = 24` shortest increasing paths from `1111` to
`0000`. The noisy comparison uses independent model realizations; summarize
their distribution rather than selecting the most illustrative seed.

## Common failures

- The reference sequence in `optimum` is assumed to remain the global maximum under noise.
- Repeated analyses of one realization are counted as independent replicates.
- Slope or noise scale is changed without recording both parameters.
- A larger raw energy is compared across graphs or fitness scales without normalization.
- An RMF pattern is interpreted as evidence for a particular biological mechanism.
