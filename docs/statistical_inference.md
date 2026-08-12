# Statistical inference contract

The publication-facing helpers in `fitness_landscape.analysis.statistics`
make data cleaning, randomization, and multiplicity decisions explicit. They
do not silently convert an invalid analysis into a numerical result.

## Distribution analysis

`analyze_fitness_distribution` requires at least one scalar, finite fitness
value. Its default `nan_policy="raise"` rejects missing values. Passing
`nan_policy="omit"` explicitly removes NaNs and records both the input size and
the omitted count. Positive and negative infinity are rejected under either
policy, and an empty post-policy sample raises `ValueError`.

Shapiro-Wilk is run only for non-constant samples containing 3 through 5000
observations. Samples with fewer than 3 observations or no variation return a
`normality_test` record with `status="not_run"`, `is_normal=None`, and an
explanatory reason. Samples larger than 5000 follow the same result policy and
also emit `UserWarning`, because the SciPy p-value is not calibrated above that
range. The interpretation threshold is the validated `alpha` argument.

## Pairwise reference tests

`hypothesis_testing` supports exactly `ttest`, `mannwhitney`, and `ks`.
Unknown or repeated test names raise instead of being ignored. All groups must
be non-empty after the selected NaN policy; infinities are always rejected.
The independent t-test additionally requires at least two observations in
every group. Raw statistics and p-values use the corresponding SciPy routines:

- `scipy.stats.ttest_ind`, with Welch's test by default;
- `scipy.stats.mannwhitneyu`, with a two-sided alternative; and
- `scipy.stats.ks_2samp`, with a two-sided alternative.

Holm family-wise error control is the default across every finite p-value
returned by one call. `bonferroni` and Benjamini-Hochberg (`fdr_bh`) are also
available. Passing `correction_method=None` or `"none"` is the explicit opt-out.
Every test record preserves `p_value`, adds `adjusted_p_value`, and defines
`significant` from the adjusted value. The method and family size are returned
both with the call and with each comparison.

## Permutation tests

`permutation_test` requires a positive integer permutation count, a supported
alternative, at least two non-empty finite groups, and a statistic function
that returns one finite scalar. For a two-sided test, the statistic must be a
signed statistic whose null value is zero; the default difference in means has
that property.

If `b` of `B` sampled label permutations are at least as extreme as the
observed statistic, Landscapy reports

```text
p = (b + 1) / (B + 1)
```

This finite-sample correction treats the observed allocation as an additional
null draw and prevents an impossible Monte Carlo p-value of zero. Each result
also reports `extreme_count`, the resolution `1 / (B + 1)`, and the Monte Carlo
standard error of the corrected estimator,

```text
sqrt(B * p * (1 - p)) / (B + 1).
```

Holm correction across pairwise comparisons is the default; the same explicit
alternatives and opt-out as `hypothesis_testing` are available.

`random_state` accepts a non-negative integer seed, a NumPy `Generator`, or
`None`. An integer creates a fresh generator, a supplied generator is consumed
in place, and `None` draws operating-system entropy. Before each comparison,
the result records the source, bit-generator type, and complete pre-comparison
state. Supplying the same seed reproduces a complete call. The recorded state
can reconstruct an individual comparison with the matching NumPy bit-generator
class, including comparisons initially run from entropy.
