# Component-wise effective resistance

Landscapy computes effective resistance only inside connected electrical
networks. For a `FitnessLandscape`, the analysis first calls
`FitnessLandscape.get_components()` and then refines those topological
components using positive conductance. An edge whose requested conductance is
zero does not connect an electrical component.

For nodes `u` and `v` in the same component, resistance is computed from that
component's Laplacian. For nodes in different components, the returned value is
`numpy.inf`; diagonal jitter is never allowed to turn disconnection into a
finite distance. `weight_epsilon` defaults to zero and, when requested, applies
only to already-positive edges inside a component.

Results report:

- `component_count`, `components`, and `component_ids`;
- `cross_component_resistance`, fixed to `numpy.inf`;
- `jitter`, `jitter_used`, and `jittered_components`; and
- `weight_epsilon` and the resolved `weight_key`.

Jitter is attempted only if factorization fails within a connected component.
For dense pseudoinverses it is applied to the zero-sum Laplacian subspace; for
sparse computation it is applied to the grounded component Laplacian.

## Category aggregation

Off-diagonal expected-pairwise category resistance is infinite whenever any
positive pair mass lies across electrical components; category self-distances
remain zero by convention. Optimal-transport distance is finite only when both
category distributions place equal total mass in every component. In that case
Landscapy solves transport independently within each component and sums the
component costs. Unequal component mass requires cross-component transport and
therefore returns infinity.

## Empty and singleton graphs

Resistance on an empty graph returns a `(0, 0)` matrix and zero components. A
singleton returns `[[0.0]]` and one component. Neither case uses jitter.

`graph_properties()` returns a stable empty schema: zero component count and
density, with `numpy.nan` for undefined degree, clustering, and path length. A
singleton has zero degree, clustering, path length, and density.

`graph_spectral_analysis()` returns empty spectral arrays for an empty graph
and a single zero eigenvalue for a singleton. Neither schema includes a
spectral gap.
