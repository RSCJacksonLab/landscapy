# Run pairwise reference tests

`hypothesis_testing` provides Welch t, Mann–Whitney, and Kolmogorov–Smirnov
tests with one explicit multiplicity family.

## Input

Groups must be non-empty finite samples. Define the independent experimental
unit and estimand before testing; this example treats dataset-level scores as
independent, not the nodes used to calculate each score.

## Worked example

```python
# cookbook: test
import numpy as np

from fitness_landscape.analysis import hypothesis_testing

groups = {
    "hamming": np.array([0.21, 0.25, 0.19, 0.30, 0.24, 0.27]),
    "knn": np.array([0.35, 0.32, 0.38, 0.29, 0.36, 0.34]),
    "diffusion": np.array([0.28, 0.31, 0.26, 0.30, 0.27, 0.29]),
}
result = hypothesis_testing(
    groups=groups,
    equal_var=False,
    run_tests=("ttest", "mannwhitney", "ks"),
    correction_method="holm",
    alpha=0.05,
    nan_policy="raise",
)

assert result["correction_family_size"] == 9
assert result["correction_method"] == "holm"
assert all(summary["n"] == 6 for summary in result["group_stats"].values())

records = []
for left, comparisons in result["pairwise_tests"].items():
    for right, tests in comparisons.items():
        effect = float(np.mean(groups[left]) - np.mean(groups[right]))
        for test_name, test in tests.items():
            assert test["adjusted_p_value"] >= test["p_value"]
            records.append(
                {
                    "contrast": f"{left} - {right}",
                    "estimand": "mean difference" if test_name == "t_test" else "distributional contrast",
                    "mean_difference": effect,
                    "test": test_name,
                    "statistic": test["statistic"],
                    "raw_p": test["p_value"],
                    "adjusted_p": test["adjusted_p_value"],
                    "significant": test["significant"],
                    "n": [len(groups[left]), len(groups[right])],
                }
            )

assert len(records) == 9
print(result["group_stats"], records)
```

Welch's test targets a mean contrast under its assumptions; Mann–Whitney and KS
do not estimate the same quantity. Holm controls family-wise error. Bonferroni
and `fdr_bh` are explicit alternatives; the choice must match the planned
family and error criterion.

## Common failures

- Millions of dependent nodes or edges are supplied as biological replicates.
- Test names, effect summary, family size, or missing-value policy are omitted.
- Mann–Whitney is automatically labeled a median test.
- Raw p-values are interpreted after selecting among many tests.
- Non-significance is described as equivalence without an equivalence design.
