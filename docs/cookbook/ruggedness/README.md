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

## Useful literature

- Stadler (1996), [*Landscapes and Their Correlation Functions*](https://doi.org/10.1007/BF01165154).
- Szendro et al. (2013), [*Quantitative analyses of empirical fitness landscapes*](https://doi.org/10.1088/1742-5468/2013/01/P01005).
- Shuman et al. (2013), [*The Emerging Field of Signal Processing on Graphs: Extending High-Dimensional Data Analysis to Networks and Other Irregular Domains*](https://doi.org/10.1109/MSP.2012.2235192).
- Matthews et al. (2024), [*Leveraging ancestral sequence reconstruction for protein representation learning*](https://doi.org/10.1038/s42256-024-00935-2).
