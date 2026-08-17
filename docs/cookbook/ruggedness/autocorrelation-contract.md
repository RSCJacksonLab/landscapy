# Stationary random-walk autocorrelation

<!-- cookbook: reference -->

Landscapy exposes separate discrete- and continuous-time autocorrelation
functions. They share one undirected conductance transition kernel and one
stationary centering convention, but they answer different questions.

For graph nodes in canonical node order, let `W` be the symmetric conductance
matrix, `d = W 1`, and

```text
P = diag(d)^-1 W
pi_i = d_i / sum_j d_j
x_c = x - (pi^T x) 1.
```

Passing `weight_key=None` makes `W` the unweighted adjacency matrix. An explicit
key uses its finite non-negative edge values as conductances. The default
`weight_key="auto"` resolves constructor-declared conductance metadata according
to the [edge-semantics contract](../graph-construction/edge-semantics.md). A valid input must be one
connected, non-trivial, undirected positive-conductance graph with a finite,
non-constant scalar signal. Use `FitnessLandscape.get_components()` before
analysis when the source landscape is disconnected.

## Discrete Markov lag

`calculate_ruggedness_autocorrelation_analytical` returns integer lags
`k = 0, ..., lag_max`, inclusive, and computes

```text
C_discrete(k) = x_c^T diag(pi) P^k x_c
                / (x_c^T diag(pi) x_c).
```

The stochastic function estimates this same quantity. It starts every walk in
`pi`, uses the global stationary mean and variance, and weights each lag by its
actual number of contributing time pairs.

No laziness is added. Consequently, a periodic or bipartite walk can have
negative or persistently oscillating discrete correlations. For a connected
unweighted `D`-regular graph, `pi` is uniform and the result is exactly

```text
C_discrete(k) = sum_j B_j (1 - lambda_j / D)^k,
```

the random-walk expression in Peter F. Stadler, *Landscapes and Their
Correlation Functions*, Journal of Mathematical Chemistry 20 (1996), 1-45,
[DOI 10.1007/BF01165154](https://doi.org/10.1007/BF01165154). Stationary
degree weighting is Landscapy's reversible irregular-graph generalization of
that regular-graph derivation. [Closed issue #45](https://github.com/RSCJacksonLab/landscapy/issues/45)
provides historical package context, but is not the mathematical validation of
this estimand.

`equivalent_single_exponential_length` is
`-1 / log(abs(C_discrete(1)))`, with zero and infinite limiting cases. It only
matches a single geometric envelope at lag one and is not a generic correlation
length, mixing time, or proof that a multimode landscape is elementary. The
legacy `correlation_length` field is therefore `None`.

## Continuous diffusion time

`time_continuous_autocorrelation` accepts finite non-negative real diffusion
times and computes

```text
L_rw = I - P
C_continuous(t) = x_c^T diag(pi) exp(-t L_rw) x_c
                  / (x_c^T diag(pi) x_c).
```

Here `t` is continuous, dimensionless diffusion time under a unit-rate
generator; it is not a count of graph steps. In the reversible spectral basis,
the same expression is `sum_j B_j exp(-t mu_j)`. It decays continuously even
when the corresponding discrete walk oscillates. `elementary_correlation_time`
is returned as `1 / mu` only when the centered signal is numerically verified
to occupy one transition mode. A general multi-exponential curve has no single
heat-decay time, so `correlation_time` remains `None`.

Both normalized functions give `C(0) = 1` and satisfy `|C| <= 1` up to floating-
point tolerance. Neither is numerically comparable to values from Landscapy's
former geodesic spectral-covariance implementation.
