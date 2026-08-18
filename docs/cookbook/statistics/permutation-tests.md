# Run a reproducible permutation test

Permutation tests require exchangeable units under the null and a statistic
whose direction and null value are explicit.

## Input

This example permutes independent dataset-level scores. It uses a signed mean
difference, two-sided alternative, fixed seed, and Holm correction.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import permutation_test

groups = {
    "hamming": np.array([0.21, 0.25, 0.19, 0.30, 0.24, 0.27]),
    "knn": np.array([0.35, 0.32, 0.38, 0.29, 0.36, 0.34]),
    "diffusion": np.array([0.28, 0.31, 0.26, 0.30, 0.27, 0.29]),
}
settings = {
    "statistic_func": lambda left, right: float(np.mean(left) - np.mean(right)),
    "n_permutations": 999,
    "alternative": "two-sided",
    "random_state": 101,
    "correction_method": "holm",
    "nan_policy": "raise",
}
first = permutation_test(groups=groups, **settings)
second = permutation_test(groups=groups, **settings)

assert first == second
assert len(first) == 3
for comparison, record in first.items():
    expected_p = (record["extreme_count"] + 1) / (record["n_permutations"] + 1)
    np.testing.assert_allclose(record["p_value"], expected_p)
    np.testing.assert_allclose(record["p_value_resolution"], 1 / 1000)
    expected_se = np.sqrt(999 * record["p_value"] * (1 - record["p_value"])) / 1000
    np.testing.assert_allclose(record["monte_carlo_standard_error"], expected_se)
    assert record["random_state"]["kind"] == "seed"
    assert record["random_state"]["seed"] == 101
    assert record["random_state"]["bit_generator"] == "PCG64"
    assert record["correction_family_size"] == 3

print(first)
```

The smallest attainable estimate here is `1/(999+1) = 0.001`, never zero. The resolution of the test can be increased with more permutations.
Record the full pre-comparison generator state stored in each result when exact
comparison-level reconstruction matters.

