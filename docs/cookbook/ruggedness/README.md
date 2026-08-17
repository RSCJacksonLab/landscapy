# Ruggedness, autocorrelation, and spectral analysis

These recipes quantify variation of an active scalar fitness layer over a
declared graph. Report the graph, fitness units and scaling, component support,
weight semantics, and any spectral truncation before comparing estimates.

1. [Dirichlet energy](dirichlet-energy.md)
2. [Local Dirichlet contributions](local-dirichlet-contributions.md)
3. [Graph Fourier analysis](graph-fourier-analysis.md)
4. [Random-walk autocorrelation](random-walk-autocorrelation.md)
5. [Compare graph views](comparing-graph-views.md)

The diffusion-scale recipe is deliberately withheld. Its normalized-Laplacian
nullspace and degenerate-signal contract remains unresolved in
[#182](https://github.com/RSCJacksonLab/landscapy/issues/182), so publishing a
worked estimator example would normalize a scientifically disputed method.

## Reference contracts

- [Stationary random-walk autocorrelation](autocorrelation-contract.md)
- [Random-walk spectral operators](spectral-operators.md)
- [Edge attributes, distance, affinity, and conductance](../graph-construction/edge-semantics.md)
