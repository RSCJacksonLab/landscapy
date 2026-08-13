# Estimate basins of attraction

A basin assigns starting nodes to a declared local optimum under a particular
walk rule. Greedy and stochastic basins are different estimands.

## Install and input

```bash
python -m pip install landscapy
```

The stochastic implementation uses NumPy's global random state. Report the
seed, simulations per node, maximum steps, inverse temperature `beta`, and
membership probability threshold.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import (
    calculate_basin_of_attraction_greedy,
    calculate_basin_of_attraction_stochastic,
    calculate_ruggedness_local_optima,
)
from fitness_landscape.core import FitnessLandscape, NumericFitness

sequences = [BinarySequence(f"{value:03b}", sequence_id=f"g{value}") for value in range(8)]
fitness = [0, 3, 1, 0, 1, 0, 3, 0]
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"two_peaks": NumericFitness.from_scalars("two_peaks", fitness)},
)
landscape.view("two_peaks")

optima = calculate_ruggedness_local_optima(landscape)
assert optima["local_optima_indices"] == [1, 6]
greedy_1 = calculate_basin_of_attraction_greedy(landscape, sequences[1])
greedy_6 = calculate_basin_of_attraction_greedy(landscape, sequences[6])
assert set(greedy_1["basin_indices"]) == {0, 1, 3, 5}
assert set(greedy_6["basin_indices"]) == {2, 4, 6, 7}

np.random.seed(37)
stochastic = calculate_basin_of_attraction_stochastic(
    landscape,
    sequences[1],
    n_simulations=200,
    max_steps=50,
    beta=3.0,
    acceptance_threshold=0.5,
)
assert stochastic["basin_probabilities_by_index"][1] == 1.0
assert stochastic["parameters"] == {
    "n_simulations": 200,
    "max_steps": 50,
    "beta": 3.0,
    "acceptance_threshold": 0.5,
}

report = {
    "optima": optima["local_optima_indices"],
    "greedy_basin_sizes": [greedy_1["basin_size"], greedy_6["basin_size"]],
    "stochastic_basin_size": stochastic["basin_size"],
    "probabilities": stochastic["basin_probabilities_by_index"],
    "seed": 37,
}
print(report)
```

Greedy membership is determined by the best-neighbour rule and its tie order.
Stochastic membership is a finite-walk probability threshold; changing `beta`,
walk length, simulations, or threshold changes the basin definition. Basins for
different optima may overlap under stochastic rules.

## Common failures

- A non-optimum sequence is supplied as the target.
- Greedy and stochastic basin sizes share one unlabeled result.
- Monte Carlo probabilities are thresholded without reporting simulations.
- Component conditioning, ties, censoring, and the global RNG seed are omitted.
- Basin membership is described as an observed evolutionary fate.
