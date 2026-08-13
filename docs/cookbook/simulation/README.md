# Simulation models and method validation

Simulated landscapes expose the data-generating assumptions and provide known
answers for checking an analysis pipeline. They are controls for software and
estimators, not evidence that an empirical landscape follows the same model.

## Recipes

- [Generate binary NK landscapes](binary-nk.md)
- [Generate generalized and multiallelic NK landscapes](generalized-nk.md)
- [Compare smooth and rugged Rough Mount Fuji landscapes](rough-mount-fuji.md)
- [Construct elementary landscapes on Hamming and kNN graphs](elementary-landscapes.md)
- [Build a reusable known-answer validation suite](method-validation.md)

Every stochastic recipe fixes the seed, records the parameters stored in the
fitness-layer metadata, and distinguishes repeated model realizations from
replicate measurements of one genotype.
