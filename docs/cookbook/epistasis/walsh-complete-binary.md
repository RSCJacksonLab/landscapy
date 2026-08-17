# Compute Walsh epistasis on a complete binary cube

Walsh analysis requires every one of the `2^L` unique 0/1 genotypes. Row order
may vary, but completeness and genotype identity must be auditable.

## Input

This known-answer signal uses `z_i = 1 - 2x_i` and
`f(x) = 1 + 2 z_0 - 0.5 z_1 z_2`.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape import BinarySequence
from fitness_landscape.analysis import calculate_epistasis_walsh
from fitness_landscape.core import FitnessLandscape, NumericFitness
from fitness_landscape.transforms import walsh_coefficients, walsh_transform

sequences = []
fitness = []
for value in range(8):
    bits = np.array([int(site) for site in f"{value:03b}"])
    z = 1 - 2 * bits
    sequences.append(BinarySequence(bits, sequence_id=f"g{value}"))
    fitness.append(1.0 + 2.0 * z[0] - 0.5 * z[1] * z[2])

assert len({tuple(sequence.to_array()) for sequence in sequences}) == 8
landscape = FitnessLandscape.build(
    sequences,
    graph="hamming",
    fitness_layers={"known": NumericFitness.from_scalars("known", fitness)},
)
landscape.view("known")

result = calculate_epistasis_walsh(landscape, order=2)
transform = walsh_transform(landscape)
orthonormal = walsh_coefficients(landscape, order=2)

np.testing.assert_allclose(result["coefficients"]["intercept"], 1.0)
np.testing.assert_allclose(result["coefficients"]["0"], 2.0)
np.testing.assert_allclose(result["coefficients"]["1,2"], -0.5)
np.testing.assert_allclose(result["coefficients"]["0,1"], 0.0, atol=1e-12)
for term, coefficient in result["coefficients"].items():
    np.testing.assert_allclose(orthonormal[term], np.sqrt(8) * coefficient)
assert transform.shape == (8,)
assert result["normalization"]["binary_coding"] == "0 -> +1; 1 -> -1"

print(result["by_order"], result["variance_explained"])
```

The orthonormal transform value is `sqrt(2^L)` times the package's reported
uniform-measure coefficient. State this normalization and zero-based position
labels whenever coefficients leave the package.

## Common failures

- A genotype is missing or duplicated, or a sequence contains a non-binary state.
- Coefficients are assigned by input row rather than genotype identity.
- Orthonormal and uniform-measure values are mixed without the `sqrt(2^L)` factor.
- Position labels are reversed by assuming little-endian genotype order.
- A known-answer transform match is treated as empirical evidence.
