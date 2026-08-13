# Landscapy cookbook

The cookbook is the worked-example companion to the [0.9 public API
contract](../public_api.md). Each page is a complete recipe: it names the
minimum installation, defines its input schema, checks alignment and graph
invariants, interprets the output, and lists common failure modes.

## Sections

- [Foundations: empirical data and the landscape data model](foundations/README.md)
- [Components, graph topology, communities, and annotated groups](topology/README.md)
- [Graph construction and representation choice](graph-construction/README.md)
- [Saving, sharing, CLI use, and external visualization](io/README.md)
- [Ruggedness, autocorrelation, and spectral analysis](ruggedness/README.md)
- [Adaptive walks, accessibility, basins, optima, and neutral networks](accessibility/README.md)
- [Epistasis on complete, sampled, and categorical landscapes](epistasis/README.md)
- [Statistical inference and robustness analysis](statistics/README.md)
- [Simulation models and known-answer validation](simulation/README.md)
- [Validated exports for downstream machine learning](ml/README.md)
- [Scaling, backend selection, and reproducible execution](scaling/README.md)

## Shared example data

Recipes use the versioned [synthetic binary landscape](data/README.md). It is
small enough to audit by hand and deliberately carries no biological claim.
Stochastic recipes use fixed seeds. Examples assume the repository root is the
working directory when run from a checkout.

## What a recipe establishes

A successful example establishes that the inputs satisfy the stated software
contract and that the reported quantity was computed. It does not establish
that the chosen graph is biologically correct or that an estimator is evidence
for a biological mechanism. Representation choice, empirical sampling, and
component support remain part of the scientific model.

All fenced blocks beginning with `# cookbook: test` are executed by
`python scripts/check_cookbook_examples.py` in CI.
