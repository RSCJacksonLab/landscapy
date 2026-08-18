# Scaling, backend selection, and reproducible execution

Scaling changes more than elapsed time. Approximate neighbour candidates,
truncated spectra, component eligibility, and resource guards can change the
estimand or the population represented by an analysis. Record those choices as
scientific inputs.

## Recipes

- [Benchmark kNN backends](knn-backends.md)
- [Batch, cache, and audit PLM embeddings](plm-embeddings.md)
- [Configure Ray and parallel construction](ray-and-parallel-construction.md)
- [Use diffusion resource guards](diffusion-resource-guards.md)
- [Audit a truncated sparse spectral analysis](sparse-spectral-analysis.md)
- [Run component-wise pipelines with honest denominators](component-wise-pipelines.md)
- [Record a reproducible run manifest](reproducibility-checklist.md)

## Reference contract

- [ESM embedding inputs, tokenization, pooling, and outputs](esm-embeddings.md)

Examples use fixed synthetic inputs and small resource limits so they remain
executable in CI. Replace them only after measuring the intended empirical
workload on its deployment platform.
