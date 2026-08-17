# Validate epistasis methods with known answers

Before empirical use, test the chosen method against a generative signal with
known additive or interaction terms, then perturb support without changing the
generative biology.

## Input

Use independent seeds for simulated landscapes and declare missingness or
imbalance mechanisms. Coefficient drift under changed design can be a design
effect rather than a changed biological interaction.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import (
    calculate_epistasis_ensemble,
    calculate_epistasis_regression,
    calculate_epistasis_walsh,
)
from fitness_landscape.core import FitnessLandscape, NumericFitness
from fitness_landscape.models import create_gnk_landscape, create_nk_binary_landscape

sequences, fitness = [], []
for value in range(8):
    bits = np.array([int(site) for site in f"{value:03b}"])
    z = 1 - 2 * bits
    sequences.append(BinarySequence(bits))
    fitness.append(1.0 + 2.0 * z[0] - 0.5 * z[1] * z[2])
full = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"known": NumericFitness.from_scalars("known", fitness)},
)
full.view("known")
walsh = calculate_epistasis_walsh(full, order=2)
np.testing.assert_allclose(walsh["coefficients"]["0"], 2.0)
np.testing.assert_allclose(walsh["coefficients"]["1,2"], -0.5)

sampled = FitnessLandscape.build(
    sequences[:5],
    graph="hamming",
    fitness_layers={"known": NumericFitness.from_scalars("known", fitness[:5])},
)
sampled.view("known")
ridge = calculate_epistasis_regression(
    sampled, order=2, regularization="l2", alpha=0.05
)
assert ridge["model"]["coefficient_solution"] == "penalty_selected"
assert not np.isclose(ridge["coefficients"]["pos0"], walsh["coefficients"]["0"])

nk = create_nk_binary_landscape(N=3, K=1, seed=43)
nk_result = calculate_epistasis_walsh(nk, order=2)
gnk = create_gnk_landscape(N=2, K=1, alphabet=["A", "B", "C"], seed=47)
gnk_result = calculate_epistasis_ensemble(gnk, order=2)
assert np.isfinite(list(nk_result["coefficients"].values())).all()
assert np.isfinite(list(gnk_result["coefficients"].values())).all()
assert gnk_result["domain"]["complete_factorial"] is True

audit = {
    "known_full": {"rows": 8, "expected_main": 2.0, "expected_pair": -0.5},
    "missingness": {"rows": 5, "penalty": "l2", "alpha": 0.05, "design_rank": ridge["model"]["design_rank"]},
    "nk": {"N": 3, "K": 1, "seed": 43},
    "gnk": {"N": 2, "K": 1, "alphabet": ["A", "B", "C"], "seed": 47},
}
print(audit)
```

The full known-answer coefficients recover exactly. Removing rows makes the
unregularized design rank-deficient, so the ridge coefficient changes even
though the generating formula did not. NK/GNK runs check finite execution over
the intended domains; their random coefficients are not empirical evidence.

## Common failures

- Visual similarity replaces a numerical expected value and tolerance.
- Missingness is introduced without recording which genotypes were removed.
- A penalty-induced coefficient change is called changed biology.
- Simulation seeds or model metadata are omitted.
- One simulated realization is treated as an independent replicate panel.
