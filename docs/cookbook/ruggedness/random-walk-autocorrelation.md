# Compare random-walk autocorrelation estimators

Autocorrelation asks how quickly a centered fitness signal decorrelates under a
declared random walk. Discrete lag and continuous diffusion time are different
domains.

## Install and input

```bash
python -m pip install landscapy
```

Use one connected component with finite non-constant scalar fitness. State the
conductance key, stationary centering, stochastic walk count, length, and seed.
Read the [autocorrelation contract](autocorrelation-contract.md).

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import (
    calculate_ruggedness_autocorrelation_analytical,
    calculate_ruggedness_autocorrelation_stochastic,
    time_continuous_autocorrelation,
)
from fitness_landscape.models import create_nk_binary_landscape

landscape = create_nk_binary_landscape(N=3, K=1, seed=17)
analytical = calculate_ruggedness_autocorrelation_analytical(
    landscape, lag_max=3, weight_key="weight"
)
stochastic = calculate_ruggedness_autocorrelation_stochastic(
    landscape,
    n_walks=2000,
    steps=20,
    lag_max=3,
    seed=29,
    weight_key="weight",
)
continuous = time_continuous_autocorrelation(
    landscape, times=[0.0, 0.5, 1.0, 2.0], weight_key="weight"
)

np.testing.assert_allclose(analytical["autocorrelation"][0], 1.0)
np.testing.assert_allclose(stochastic["autocorrelation"][0], 1.0)
np.testing.assert_allclose(continuous["autocorrelation"][0], 1.0)
assert analytical["lags"].tolist() == [0, 1, 2, 3]
assert continuous["times"].tolist() == [0.0, 0.5, 1.0, 2.0]
assert np.all(stochastic["pair_counts"] > 0)

comparison = {
    "analytical_lag_1": analytical["autocorrelation"][1],
    "stochastic_lag_1": stochastic["autocorrelation"][1],
    "absolute_MC_difference": abs(
        analytical["autocorrelation"][1] - stochastic["autocorrelation"][1]
    ),
    "continuous_time_1": continuous["autocorrelation"][2],
    "walks": 2000,
    "seed": 29,
}
assert comparison["absolute_MC_difference"] < 0.08
print(comparison)
```

The analytical and stochastic values target the same discrete Markov-lag
quantity; their difference is Monte Carlo error. Continuous time uses a
semigroup and cannot be substituted for integer lag. Bipartite/periodic graphs
can show oscillating discrete correlations, so a single exponential length may
be unavailable.

## Common failures

- Disconnected components are combined into one finite correlation process.
- Arithmetic rather than stationary centering is assumed without checking.
- Continuous time `t=1` is called equivalent to one random-walk step.
- One seeded stochastic run is reported without pair counts or uncertainty.
- A missing correlation-length estimate is replaced by an arbitrary cutoff.
