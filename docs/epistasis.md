# Epistasis domains and coefficient contracts

Landscapy exposes three distinct epistasis estimands. They are not
interchangeable, and each public result records its domain and normalization.
Every method requires equal-length sequences, a finite scalar active fitness
value for every observed sequence, and an interaction order from one through
the sequence length.

| Method | Supported sequence design | Coefficient definition |
| --- | --- | --- |
| `calculate_epistasis_walsh` | Complete, duplicate-free binary cube encoded as 0/1 | Uniform-measure Fourier-Walsh coefficients |
| `calculate_epistasis_regression` | Complete or sampled binary design encoded as 0/1 | Fitted effect-coded regression coefficients |
| `calculate_epistasis_ensemble` | Complete, incomplete, balanced, or unbalanced categorical/multi-allelic observed design | Empirical marginal Möbius coefficients |
| `calculate_epistasis_reference_free` | Same general categorical domain as `calculate_epistasis_ensemble` | The same empirical marginal Möbius coefficients, without selecting a reference allele |

Unsupported designs raise `ValueError`; no method silently returns `None`.

## Complete binary cubes

For a binary sequence `x` of length `L`, let `z_i(x) = 1 - 2 x_i`. The
coefficient reported for position set `S` by `calculate_epistasis_walsh` is

```text
c_S = 2**(-L) sum_x f(x) product_{i in S} z_i(x).
```

This is the uniform-measure Fourier-Walsh coefficient normalization used here.
Stating the formula is important: finite-difference conventions sometimes
attach additional order-dependent scale or sign factors to the word
"epistasis".
`walsh_coefficients` remains an orthonormal transform and returns
`w_S = sqrt(2**L) c_S`. `calculate_epistasis_walsh` includes those values under
`orthonormal_coefficients` and records the conversion in `normalization`.
Interaction labels use zero-based biological sequence positions in ascending
order; the implementation reverses the least-significant FWHT mask convention
when mapping masks back to the package's big-endian genotype order.

On a complete cube, unregularized `calculate_epistasis_regression` uses the
same `z_i` columns and therefore returns the same `c_S` values, with regression
labels such as `pos0*pos2` instead of Walsh labels such as `0,2`.

This normalization follows the `2**(-L)` Walsh coefficient convention described
by [Weinreich et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5986866/).
The extension of background-averaged epistasis from binary to multiallelic
landscapes is developed by
[Faure et al. (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11161127/).
Landscapy's observed-support behavior for incomplete designs is the explicit
empirical-marginal contract defined below; it is not an imputation method.

## Sampled binary regression

Regression fits an intercept and every effect-coded product column through the
requested order. For unregularized least squares, the augmented design matrix
must have full column rank. A rank-deficient request raises an error that asks
the caller to reduce the order, collect more observations, or select an
explicit penalty.

`l2`, `l1`, and `elastic_net` penalties accept rank-deficient designs when
`alpha` is finite and positive. Every penalized fit labels the returned
coefficient solution as `penalty_selected`, including when the corresponding
unpenalized design has full rank: the fitted coefficients are defined by the
chosen penalized optimization problem. The separate
`unregularized_coefficients_identifiable` field reports whether the observed
design alone would identify the unpenalized coefficients. The result also
reports observation count, parameter count, design rank, and penalty under
`model`.

On incomplete designs the fitted intercept is estimated jointly with the
effects and need not equal the raw fitness mean. Coefficients from penalized
and unpenalized fits, or from differently sampled genotype designs, should not
be compared as if they were the same estimand without accounting for those
design and penalty differences.

## General categorical and incomplete designs

The ensemble and reference-free methods implement the same explicitly defined
observed-support decomposition. For a subset of positions `S` and an observed
allele cell `a_S`, define its empirical marginal mean as

```text
m_S(a_S) = mean(y_r for observed rows r with X[r, S] = a_S).
```

Starting with `h_empty = m_empty`, Landscapy applies the full subset recursion

```text
h_S(a_S) = m_S(a_S) - sum_{T proper subset of S} h_T(a_T).
```

The sum includes *all* lower orders. A fourth-order coefficient therefore
subtracts the intercept, every main effect, every pairwise effect, and every
third-order effect. Summing the resulting hierarchy reconstructs each emitted
empirical marginal cell mean exactly.

On a complete balanced factorial design this is a functional-ANOVA
decomposition under the uniform empirical measure. On an incomplete or
unbalanced design it is a hierarchical decomposition of the observed empirical
marginals, not an orthogonal population ANOVA:

- observations have equal weight;
- missing genotype cells are not imputed or extrapolated;
- coefficients are emitted only for observed marginal cells; and
- sampling imbalance can move signal between nominal interaction orders.

The output reports alphabet levels, observed and possible genotype-cell counts,
factorial completeness, balance, weighting, orthogonality status, and the
missing-cell policy. Callers needing population-weighted contrasts, a specified
reference allele, or model-based estimates for absent genotypes must fit that
estimand explicitly rather than interpreting these empirical coefficients as
one of those alternatives.
