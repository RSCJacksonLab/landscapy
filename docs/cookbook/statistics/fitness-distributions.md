# Summarize a fitness distribution

Distribution summaries describe finite scalar values in the active layer.
Missingness, sample size, and degenerate cases remain part of the result.

## Install and input

```bash
python -m pip install landscapy
```

`nan_policy="raise"` is the default. Use `"omit"` only when omission is
scientifically justified and report both the input and retained sample sizes.

## Worked example

```python
# cookbook: test
import warnings

import networkx as nx
import numpy as np

from fitness_landscape.analysis import analyze_fitness_distribution
from fitness_landscape.core import BaseNumpySequence, FitnessLandscape, NumericFitness

def make_landscape(values):
    sequences = [
        BaseNumpySequence([index], sequence_id=f"row-{index}")
        for index in range(len(values))
    ]
    graph = nx.empty_graph(len(values))
    for node, sequence in enumerate(sequences):
        graph.nodes[node]["sequence"] = sequence
    landscape = FitnessLandscape(
        sequences,
        graph,
        fitness_layers={"assay": NumericFitness.from_scalars("assay", values)},
    )
    landscape.view("assay")
    return landscape

finite = analyze_fitness_distribution(
    make_landscape([0.1, 0.2, 0.3, 0.45, 0.5, 0.65, 0.8, 0.9]),
    alpha=0.05,
    nan_policy="raise",
)
omitted = analyze_fitness_distribution(
    make_landscape([0.1, np.nan, 0.3, 0.5]), nan_policy="omit"
)
small = analyze_fitness_distribution(make_landscape([0.1, 0.2]))
constant = analyze_fitness_distribution(make_landscape([1.0] * 4))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    large = analyze_fitness_distribution(make_landscape(np.arange(5001)))

assert finite["sample_size"] == finite["input_sample_size"] == 8
assert finite["normality_test"]["status"] == "performed"
assert omitted["sample_size"] == 3 and omitted["omitted_count"] == 1
assert small["normality_test"]["status"] == "not_run"
assert constant["normality_test"]["reason"] == "Shapiro-Wilk is undefined for a constant sample."
assert large["normality_test"]["status"] == "not_run" and caught

report = {
    "finite": {key: finite[key] for key in ["mean", "median", "std", "sample_size"]},
    "omitted": {key: omitted[key] for key in ["input_sample_size", "sample_size", "omitted_count"]},
    "small_normality": small["normality_test"],
    "constant_normality": constant["normality_test"],
    "large_normality": large["normality_test"],
}
print(report)
```

Shapiro-Wilk runs only for non-constant samples of 3–5000 values. Its p-value
tests one distributional null; it is not a universal decision rule for choosing
all downstream models, transformations, or parametric tests.

## Common failures

- NaNs are silently removed without a policy or denominator.
- Infinity is treated as an ordinary extreme value.
- A constant or undersized sample receives a fabricated normality p-value.
- `p >= 0.05` is reported as proof that the population is normal.
- Graph-dependent values are treated as independent assay replicates.
