# Reversible diffusion graph semantics

Both embedding and evolutionary diffusion graphs use one mathematical
contract. Let `W` be their symmetric, finite, non-negative pair-affinity
matrix. Embedding diffusion obtains `W` from an RBF kernel. Evolutionary
diffusion exponentiates the symmetric length-normalized log-odds scores with a
single global numerical shift; it does not apply row-dependent score shifts.

Self-affinities are removed before transition construction. An isolated state
receives an absorbing self transition. For all other states, define

```text
Q = D^-1 W
P = (I + Q) / 2
pi_i = D_ii / sum_j D_jj
```

where `D_ii = sum_j W_ij`. The one-half hold probability makes every component
aperiodic. Symmetry of `W` gives detailed balance,
`pi_i P_ij = pi_j P_ji`.

For a finite integer `t >= 1`, Landscapy stores the upper triangle of the
stationary-measure similarity kernel

```text
K_t = Pi^(1/2) P^t Pi^(-1/2).
```

The two numerical orientations are averaged before thresholding. Detailed
balance makes them equal analytically. Consequently `K_t` is symmetric and
permutation equivariant: applying a permutation matrix `R` to inputs produces
`R K_t R^T`, so edge values cannot depend on row or node order.

`t=None`, `t=0`, and `t=inf` all request the componentwise stationary limit.
For states `i` and `j` in the same communicating component `C`,

```text
K_inf(i, j) = sqrt(pi_i pi_j) / sum(k in C) pi_k.
```

For states in different components it is zero. This is a pairwise stationary
overlap, not an endpoint marginal, so reducible components and isolated states
cannot acquire artificial cross-component edges. The lazy transition ensures
the same limit exists for chains whose non-lazy form would be periodic.

`connectivity_threshold` is applied strictly after construction of `K_t`.
Edge `affinity`, `weight`, and legacy `kernel_weight` therefore have units of
dimensionless reversible diffusion amplitude. Self entries are never exported
as graph edges.
