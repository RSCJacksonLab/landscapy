# Run stochastic adaptive walks

`adaptive_walk_stochastic` implements an algorithmic hill-climbing walk. Unlike the greedy adaptive walks, stochastic adaptive walks allow mutations that may deteriorate fitness, and are not deterministic.

## Input

The function uses NumPy's process-wide random state. Set and record it before
sampling starts or random-improvement steps, and avoid concurrent code that
mutates the same state.

## Worked example

```python
# cookbook: test
from collections import Counter

import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import adaptive_walk_stochastic
from fitness_landscape.core import FitnessLandscape, NumericFitness

sequences = [BinarySequence(f"{value:03b}", sequence_id=f"g{value}") for value in range(8)]
fitness = [sum(int(site) for site in f"{value:03b}") for value in range(8)]
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"additive": NumericFitness.from_scalars("additive", fitness)},
)
landscape.view("additive")

greedy = adaptive_walk_stochastic(
    landscape, start_sequence=sequences[0], max_steps=10, strategy="greedy"
)
assert greedy["steps_taken"] == 3
assert np.all(np.diff(greedy["walk_fitness"]) > 0)

np.random.seed(31)
random_walks = [
    adaptive_walk_stochastic(
        landscape,
        start_sequence=sequences[0],
        max_steps=10,
        strategy="random_improvement",
    )
    for _ in range(200)
]
endpoints = Counter(walk["walk_indices"][-1] for walk in random_walks)
assert endpoints == {7: 200}
assert {walk["steps_taken"] for walk in random_walks} == {3}

np.random.seed(31)
sampled_starts = [
    adaptive_walk_stochastic(landscape, start_sequence=None, max_steps=10)
    for _ in range(50)
]
start_counts = Counter(walk["walk_indices"][0] for walk in sampled_starts)
assert sum(start_counts.values()) == 50

report = {
    "graph": "complete three-site Hamming cube",
    "layer": "additive count, model units",
    "component_nodes": 8,
    "seed": 31,
    "replicates": 200,
    "max_steps": 10,
    "endpoint_counts": dict(endpoints),
    "sampled_start_counts": dict(start_counts),
}
print(report)
```

Report the endpoint distribution and stopping reason, not only a representative
path. `reached_optimum` means the implementation stopped before `max_steps`; it
does not prove a global optimum was reached unless independently checked.
