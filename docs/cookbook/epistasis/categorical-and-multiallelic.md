# Decompose categorical and multiallelic designs

The ensemble and reference-free APIs compute the same hierarchical Möbius
decomposition of observed empirical marginal means. They support complete,
incomplete, balanced, and unbalanced observed designs without imputation. For
broader treatments linking epistasis parameterizations and extending
Walsh-Hadamard ideas beyond complete binary designs, see
[Poelwijk, Krishna, and Ranganathan (2016)](https://doi.org/10.1371/journal.pcbi.1004771)
and [Faure et al. (2024)](https://doi.org/10.1371/journal.pcbi.1012132).

## Input

## Worked example

```python
# cookbook: test
import warnings

import networkx as nx
import numpy as np

from fitness_landscape.analysis import calculate_epistasis_ensemble, calculate_epistasis_reference_free
from fitness_landscape.core import BaseNumpySequence, FitnessLandscape, NumericFitness

def make_landscape(texts, values):
    sequences = [
        BaseNumpySequence(list(text), alphabet=["A", "B"], sequence_id=f"row-{i}")
        for i, text in enumerate(texts)
    ]
    graph = nx.empty_graph(len(sequences))
    for node, sequence in enumerate(sequences):
        graph.nodes[node]["sequence"] = sequence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        landscape = FitnessLandscape(
            sequences,
            graph,
            fitness_layers={"fitness": NumericFitness.from_scalars("fitness", values)},
        )
    landscape.view("fitness")
    return landscape

designs = {
    "complete_balanced": (["AA", "AB", "BA", "BB"], [1.0, 2.0, 3.0, 6.0]),
    "incomplete": (["AA", "AB", "BA"], [1.0, 2.0, 3.0]),
    "complete_unbalanced": (["AA", "AA", "AB", "BA", "BB"], [1.0, 1.2, 2.0, 3.0, 6.0]),
}

report = {}
for name, (texts, values) in designs.items():
    landscape = make_landscape(texts, values)
    ensemble = calculate_epistasis_ensemble(landscape, order=2)
    reference = calculate_epistasis_reference_free(landscape, order=2)
    assert ensemble["coefficients"] == reference["coefficients"]

    cell_means = {}
    for text in sorted(set(texts)):
        observed = [value for row, value in zip(texts, values) if row == text]
        coefficients = ensemble["coefficients"]
        reconstructed = (
            coefficients["intercept"]
            + coefficients[f"0:{text[0]}"]
            + coefficients[f"1:{text[1]}"]
            + coefficients[f"0:{text[0]},1:{text[1]}"]
        )
        np.testing.assert_allclose(reconstructed, np.mean(observed))
        cell_means[text] = reconstructed

    report[name] = {
        "domain": ensemble["domain"],
        "decomposition": ensemble["decomposition"],
        "emitted_pair_cells": len(ensemble["by_order"][2]),
        "reconstructed_cell_means": cell_means,
    }

assert report["complete_balanced"]["decomposition"]["orthogonal_anova"] is True
assert report["incomplete"]["domain"]["n_observed_genotype_cells"] == 3
assert report["incomplete"]["emitted_pair_cells"] == 3
assert report["complete_unbalanced"]["domain"]["balanced_genotype_counts"] is False
print(report)
```

Every emitted observed cell mean reconstructs exactly. In incomplete designs,
absent cells are not emitted. In unbalanced designs, equal observation weights
change empirical marginals and the decomposition is not orthogonal.
