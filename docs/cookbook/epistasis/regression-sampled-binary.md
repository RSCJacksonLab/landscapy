# Fit epistasis regression on sampled binary designs

Effect-coded regression supports sampled 0/1 genotypes, but unregularized
coefficients exist only when the intercept-plus-interaction design has full
column rank.

## Input

Report observation and parameter counts, rank, requested order, intercept,
penalty, alpha, sampling support, residual fit, and coefficient solution type.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import calculate_epistasis_regression
from fitness_landscape.core import FitnessLandscape, NumericFitness

def make_sample(n_rows):
    sequences, fitness = [], []
    for value in range(n_rows):
        bits = np.array([int(site) for site in f"{value:03b}"])
        z = 1 - 2 * bits
        sequences.append(BinarySequence(bits, sequence_id=f"g{value}"))
        fitness.append(1.0 + 2.0 * z[0] - 0.5 * z[1] * z[2])
    landscape = FitnessLandscape.build(
        sequences,
        graph="hamming",
        fitness_layers={"known": NumericFitness.from_scalars("known", fitness)},
    )
    landscape.view("known")
    return landscape

identified = calculate_epistasis_regression(make_sample(7), order=2)
assert identified["model"]["n_observations"] == 7
assert identified["model"]["n_parameters"] == 7
assert identified["model"]["design_rank"] == 7
assert identified["model"]["coefficient_solution"] == "data_identified"
np.testing.assert_allclose(identified["coefficients"]["pos0"], 2.0)
np.testing.assert_allclose(identified["coefficients"]["pos1*pos2"], -0.5)
assert identified["model"]["r2_score"] == 1.0

rank_deficient = make_sample(5)
try:
    calculate_epistasis_regression(rank_deficient, order=2)
except ValueError as error:
    assert "not identifiable" in str(error)
else:
    raise AssertionError("rank-deficient unregularized fit must fail")

penalized = {}
for penalty in ["l1", "l2", "elastic_net"]:
    result = calculate_epistasis_regression(
        rank_deficient,
        order=2,
        regularization=penalty,
        alpha=0.05,
        l1_ratio=0.5,
    )
    assert result["model"]["design_rank"] == 5
    assert result["model"]["coefficient_solution"] == "penalty_selected"
    penalized[penalty] = {
        "intercept": result["coefficients"]["intercept"],
        "r2": result["model"]["r2_score"],
        "alpha": result["model"]["alpha"],
        "coefficients": result["coefficients"],
    }

print(identified["model"], penalized)
```

`R^2` is an in-sample residual diagnostic, not evidence of coefficient
identifiability or predictive performance. Penalized solutions depend on
penalty, alpha, scaling, and observed genotypes even when fit is nearly exact.

## Common failures

- Observation count alone is used instead of augmented-design rank.
- A rank-deficient unregularized request is forced through another estimator.
- Alpha and elastic-net `l1_ratio` are omitted.
- Penalized coefficients are described as data-identified effects.
- In-sample `R^2` is reported as held-out performance.
