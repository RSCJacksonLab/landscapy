# Reversible diffusion graph semantics

<!-- cookbook: reference -->

Both embedding and evolutionary diffusion graphs use one mathematical
contract. Let `W` be their symmetric, finite, non-negative pair-affinity
matrix. Embedding diffusion obtains sparse `W` from an RBF kernel evaluated
only on the symmetric union of directed kNN candidates. Evolutionary
diffusion exponentiates the symmetric length-normalized log-odds scores with a
single global numerical shift; it does not apply row-dependent score shifts.

For embedding diffusion, `k` is the number of non-self candidates requested
per row. `tiebuffer` asks the backend for additional hits, which are reranked
in the declared exact geometry; only extra hits tied at the kth exact distance
enter `W`. BallTree and flat FAISS define an exact candidate universe. HNSW
and IVF FAISS define an approximate candidate universe, so missed exact
neighbours can change the support of `W`; subsequent RBF evaluation and
diffusion are exact conditional on that returned universe. Graph metadata
records the backend, index, tie rule, and whether candidate selection was
approximate.

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

For a finite integer `t >= 1`, Landscapy computes the exact sparse matrix power
and stores the upper triangle of the stationary-measure similarity kernel

```text
K_t = Pi^(1/2) P^t Pi^(-1/2).
```

The two numerical orientations are averaged before thresholding. Detailed
balance makes them equal analytically. Consequently `K_t` is symmetric and
permutation equivariant: applying a permutation matrix `R` to inputs produces
`R K_t R^T`, so edge values cannot depend on row or node order.

Exact sparse powers can still become dense as `t` grows. During embedding-
diffusion construction, Landscapy counts the exact output structure and scalar
products before every matrix multiplication using O(n) marker storage. It
raises an actionable `MemoryError` before either `max_diffusion_nnz` (default
50,000,000) or `max_diffusion_work` (default 1,000,000,000) would be exceeded.
These are feasibility guards, not approximations: increasing them cannot
change a result that already fits. `connectivity_threshold` cannot reduce
intermediate work because thresholding remains post-kernel.

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

The componentwise stationary limit contains one dense block per communicating
component even when `W` is sparse. The embedding constructor represents those
blocks as sparse coordinate data only when their exact structural size fits
the same resource budgets; otherwise it raises before allocating a quadratic
component block.

## Scalability benchmark

`benchmarks/benchmark_embedding_diffusion.py` records elapsed time, edge count,
candidate count, sparse nonzeros, and estimated working bytes for explicit
`n,d,k,t` cases. Its default release cases cover finite powers one and two and
node counts from 1,000 to 5,000; `--case n,d,k,t` adds larger target workloads.
